"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.causal = config.causal
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            ones_mask =torch.ones(config.block_size, config.block_size)
            if self.causal:
                # causal mask to ensure that attention is only applied to the left in the input sequence
                self.register_buffer("bias", torch.tril(ones_mask)
                                            .view(1, 1, config.block_size, config.block_size))
            else:
                self.register_buffer("bias", ones_mask.view(1, 1, config.block_size, config.block_size))
                

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=self.causal)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) # diagonal if causal, the same otherwise
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # context size (T)
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    stoi: dict = None
    itos: dict = None
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    prefix_size: int = 0
    example_size: int = 0
    calibrate: bool = False
    multiplier: float = 1.0
    cross_calibrate: bool = False
    cross_multiplier: float = 1.0
    confidence: bool = True
    causal: bool = True

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)        
    
    def ECE(self, logits, targets, K=5, smooth=False, sigma=0.1, sample_weights=None, confidence=True):
        """
        Computes ECE loss on argmax predictions and targets. 
        If smooth is False, uses ECE loss (K buckets).
        If smooth is True, uses smooth ECE loss (Gaussian weights centered at K points with std dev sigma).
        sample_weights optionally weights each example (B x T): weights must be non-negative.
        """
        probs = F.softmax(logits, dim=-1)
        if sample_weights is None:
            N = probs.size(0) * probs.size(1) # number of examples (B*T)
        else:
            sample_weights = sample_weights.to(dtype=probs.dtype)
            N = sample_weights.sum()
            if N <= 0:
                return torch.tensor(0.0, device=logits.device, dtype=probs.dtype)

        # max probs is a tensor of size batch size x context size with max_probs[b,t] = max(probs[b,t,:])
        # max_probs = torch.max(probs, dim=-1)
        if confidence:
            # sample predictions from probs
            sampled_predictions = torch.distributions.Categorical(probs=probs).sample()
            # or set deterministically
            # sampled_predictions = torch.argmax(probs, dim=-1)
            # label is 1 if sampled prediction is correct, 0 otherwise
            labels = (sampled_predictions == targets)
            predictions = torch.max(probs, dim=-1).values # use max probabilities as predictions
        else:
            # labels are target tokens
            # map target indices to tokens
            flat_targets = targets.reshape(-1).tolist()
            try:
                labels_list = [float(self.config.itos[int(target_idx)]) for target_idx in flat_targets]
            except KeyError as e:
                raise ValueError(f"Target index {e.args[0]} not found in config.itos") from e
            except ValueError as e:
                raise ValueError("Target tokens must be numeric for ECE") from e
            labels = torch.tensor(labels_list, device=targets.device, dtype=probs.dtype).view_as(targets)
            # binary tasks: predictions are probabilities place on token 1
            idx = self.config.stoi["1"]
            predictions = probs[:, :, idx]

        if smooth:
            # define soft buckets using Gaussian kernels
            centers = torch.linspace(0, 1, K, device=logits.device, dtype=probs.dtype)
            # first compute unnormalized weights
            diff = predictions.unsqueeze(-1) - centers 
            unnorm_weights = torch.exp(- (diff ** 2) / (2 * sigma ** 2))
            # then normalize across K dimension
            weights = unnorm_weights / unnorm_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        else:
            # define hard buckets using uniform bins
            centers = torch.linspace(0, 1, K+1, device=logits.device, dtype=probs.dtype)
            # weights[b,t,k] = 1 if predictions[b,t] is in the k-th bucket, 0 otherwise
            weights = torch.zeros(probs.size(0), probs.size(1), K, device=logits.device, dtype=probs.dtype)
            left = centers[:-1] 
            left = left.view(1, 1, -1).expand(probs.size(0), probs.size(1), -1) # expand to size B x T x K  
            right = centers[1:] 
            right = right.view(1, 1, -1).expand(probs.size(0), probs.size(1), -1) # expand to size B x T x K
            for k in range(K):
                if k == K-1:
                    # compare each entry predictions[b,t] with left[b,t,k] and right[b,t,k]
                    weights[:, :, k] = (predictions >= left[:, :, k]) & (predictions <= right[:, :, k])
                else:
                    weights[:, :, k] = (predictions >= left[:, :, k]) & (predictions < right[:, :, k])

        if sample_weights is not None:
            weights = weights * sample_weights.unsqueeze(-1)
        S = weights.sum(dim=(0, 1)) # sum of weights for each bucket

        p_bar = torch.zeros(K, device=logits.device, dtype=probs.dtype) # mean probability for each bucket
        for k in range(K):
            p_bar[k] = ((weights[:, :, k] * predictions).sum()) / S[k] if S[k] > 0 else 0.0
        y_bar = torch.zeros(K, device=logits.device, dtype=probs.dtype) # mean label for each bucket
        for k in range(K):
            y_bar[k] = ((weights[:, :, k] * labels).sum()) / S[k] if S[k] > 0 else 0.0

        return torch.sum(S * torch.abs(p_bar - y_bar)) / N

    def cross_ECE(self, logits, targets, collaborator_predictions, collaborator_probs=None, K=5, smooth_collaborator=False, smooth_own=False, sigma=0.1, confidence=True):
        """
        Computes cross ECE between the model and the collaborator: 
        sum_{collaborator_predictions} ECE of the model conditional on collaborator's prediction.
        """
        # SMOOTH CROSS ECE ONLY MAKES SENSE IF PREDICTIONS ARE CONTINUOUS/ORDERED
        if collaborator_predictions is not None: # use predictions 
            # re-scale collaborator's predictions to lie between 0 and 1
            # collaborator_probs = (collaborator_predictions - collaborator_predictions.min()) / (collaborator_predictions.max() - collaborator_predictions.min())
            L = len(torch.unique(collaborator_predictions)) # number of unique predictions
            collaborator_min = collaborator_predictions.min()
            collaborator_max = collaborator_predictions.max()
            collaborator_probs = collaborator_predictions
        else: # use probabilities
            L = K
            collaborator_min = 0.0
            collaborator_max = 1.0


        if smooth_collaborator:
            # define soft buckets using Gaussian kernels
            centers_collaborator = torch.linspace(collaborator_min, collaborator_max, L, device=logits.device, dtype=logits.dtype)
            diff = collaborator_probs.unsqueeze(-1) - centers_collaborator
            unnorm_weights = torch.exp(- (diff ** 2) / (2 * sigma ** 2))
            weights_collaborator = unnorm_weights / unnorm_weights.sum(dim=-1, keepdim=True)
        else:
            # define hard buckets using uniform bins
            centers_collaborator = torch.linspace(collaborator_min, collaborator_max, L+1, device=logits.device, dtype=logits.dtype)
            left_collaborator = centers_collaborator[:-1]
            left_collaborator = left_collaborator.view(1, 1, -1).expand(logits.size(0), logits.size(1), -1)
            right_collaborator = centers_collaborator[1:]
            right_collaborator = right_collaborator.view(1, 1, -1).expand(logits.size(0), logits.size(1), -1)
            weights_collaborator = torch.zeros(logits.size(0), logits.size(1), L, device=logits.device, dtype=logits.dtype)
            for l in range(L):
                if l == L-1:
                    weights_collaborator[:, :, l] = (collaborator_probs >= left_collaborator[:, :, l]) & (collaborator_probs <= right_collaborator[:, :, l])
                else:
                    weights_collaborator[:, :, l] = (collaborator_probs >= left_collaborator[:, :, l]) & (collaborator_probs < right_collaborator[:, :, l])
            weights_collaborator = weights_collaborator.to(dtype=logits.dtype)

        N = logits.size(0) * logits.size(1) # number of examples (B*T)
        cross_ece = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        for l in range(L):
            bucket_weights = weights_collaborator[:, :, l]
            S_l = bucket_weights.sum()
            if S_l <= 0:
                continue
            bucket_ece = self.ECE(logits, targets, K=K, smooth=smooth_own, sigma=sigma, sample_weights=bucket_weights, confidence=confidence)
            cross_ece = cross_ece + bucket_ece * S_l

        return cross_ece / N  
    
    def brier_score(self, logits, targets):
        """
        Computes Brier score on argmax probabilities and accuracy.
        """
        probs = F.softmax(logits, dim=-1)
        N = probs.size(0) * probs.size(1) # number of examples (B*T)
        max_prob_indices = torch.argmax(probs, dim=-1)
        max_probs = torch.amax(probs, dim=-1)
        max_prob_labels = (max_prob_indices == targets).float()
        return torch.sum((max_probs - max_prob_labels) ** 2) / N

    def zero_one_loss(self, logits, targets):
        """
        Computes 0/1 loss on sampled predictions and targets.
        """
        probs = F.softmax(logits, dim=-1)
        N = probs.size(0) * probs.size(1) # number of examples (B*T) 
        # max_prob_indices = torch.argmax(logits, dim=-1)
        predictions = torch.distributions.Categorical(probs=probs).sample()
        return torch.sum(predictions != targets) / N

    def entropy(self, logits):
        """
        Computes entropy of the logits.
        """
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs), dim=-1)
        return entropy.mean(dim=(0,1))

    def forward(self, idx, targets=None, collaborator_predictions=None, confidence=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
        if confidence is None:
            confidence = self.config.confidence

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x) # shape (b, t, n_embd)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            if self.config.prefix_size > 0: # if we are using a prefix, crop the logits and targets
                logits = logits[:, self.config.prefix_size:, :]
                targets = targets[:, self.config.prefix_size:]
                probs = F.softmax(logits, dim=-1)
                # probs0 = probs[:,:,1]
                # probs1 = probs[:,:,2]
                # print("probs0:", probs0)
                # print("probs1:", probs1)
            # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
            #loss = 0
            ece_loss = self.ECE(logits, targets, smooth=False, confidence=confidence)
            sm_ece_loss = self.ECE(logits, targets, smooth=True, confidence=confidence)
            brier_score = self.brier_score(logits, targets)
            zero_one_loss = self.zero_one_loss(logits, targets)
            
            if collaborator_predictions is not None:
                # calculate cross calibration error wrt collaborator predictions
                cross_ece_loss = self.cross_ECE(logits, targets, collaborator_predictions, smooth_collaborator=False, smooth_own=False, confidence=confidence)
                sm_cross_ece_loss = self.cross_ECE(logits, targets, collaborator_predictions, smooth_collaborator=True, smooth_own=True, confidence=confidence)
                # if cross_ece_loss + 1e-6 < ece_loss:
                #     print("self ECE loss:", ece_loss)
                #     print("cross ECE loss:", cross_ece_loss)
                #     raise ValueError("Cross ECE loss is less than self ECE loss. This should not happen.")
                # if sm_cross_ece_loss + 1e-6 < sm_ece_loss:
                #     print("smooth self ECE loss:", sm_ece_loss)
                #     print("smooth cross ECE loss:", sm_cross_ece_loss)
                #     raise ValueError("Smoothed cross ECE loss is less than smoothed self ECE loss. This should not happen.")

                if self.config.cross_calibrate:
                    loss = loss + sm_cross_ece_loss * self.config.cross_multiplier
            else:
                cross_ece_loss = None
                sm_cross_ece_loss = None

            if self.config.calibrate == 'smECE':
                if collaborator_predictions is None:
                    loss = loss + sm_ece_loss * self.config.multiplier
                else:
                    if self.config.cross_calibrate is False:
                        loss = loss + sm_ece_loss * self.config.multiplier
            elif self.config.calibrate == 'brier':
                if collaborator_predictions is None:
                    loss = loss + brier_score * self.config.multiplier
                else:
                    if self.config.cross_calibrate is False:
                        loss = loss + brier_score * self.config.multiplier
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
            ece_loss = None
            sm_ece_loss = None
            cross_ece_loss = None
            sm_cross_ece_loss = None
            brier_score = None
            zero_one_loss = None
        entropy = self.entropy(logits)
        return logits, loss, ece_loss, sm_ece_loss, cross_ece_loss, sm_cross_ece_loss, brier_score, zero_one_loss, entropy

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _, _, _, _, _, _, _, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    @torch.no_grad()
    def generate_with_probs(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        Returns probability vector at each time. 
        """
        probs_list =[]
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _, _, _, _, _, _, _, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            probs_list.append(probs)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        probs_list = torch.stack(probs_list, dim=1)
        return idx, probs_list
