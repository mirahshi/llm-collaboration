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
from dataclasses import dataclass, field

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

class CalibrateMLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        dim = len(config.answer_tokens)
        self.c_fc    = nn.Linear(dim * 2, 8 * dim, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_out   = nn.Linear(8 * dim, dim, bias=config.bias)

    def forward(self, p1, p2):
        x = torch.cat((p1, p2), dim=-1)
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_out(x)
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
    cross_probabilities: bool = False
    K: int = 5 # number of buckets for self ECE
    K_cross: int = 5 # number of buckets for cross ECE
    compute_smooth_calibration: bool = False # compute smooth calibration losses (memory intensive)
    append_probabilities_temperature: float = 1.0 # temperature for appended probabilities (default is 1; here it is used for logging ECE loss on scaled probabilities)
    post_hoc_calibrate: bool = False # post-hoc calibrate the model using the predictions of the previous round
    post_hoc_calibrate_multiplier: float = 1.0 # multiplier for post-hoc cross calibration loss
    post_hoc_calibrate_use_smECE: bool = True # use sm cross ece loss for post-hoc calibration; if False, use CE loss
    post_hoc_calibrate_use_smECE_bins: int = 4 # number of bins for sm cross ece loss
    m_lookahead: int = 1 # number of autoregressive lookahead predictions to generate
    autoregressive_lookahead: bool = True # use autoregressive lookahead (otherwise, use ground truth targets)
    answer_tokens: list = field(default_factory=lambda: [''])
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

        # init post-hoc calibrate MLP
        self.calibrate_mlp = CalibrateMLP(config)

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
        
        expand_predictions = predictions.unsqueeze(-1).expand(-1, -1, K) # expand to size B x T x K
       
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
            weights = (expand_predictions >= left) & (expand_predictions < right)
            weights[:, :, -1] = (predictions >= left[:, :, -1]) & (predictions <= right[:, :, -1]) # for last bucket, include right endpoint

        if sample_weights is not None:
            weights = weights * sample_weights.unsqueeze(-1)
        S = weights.sum(dim=(0, 1)) # sum of weights for each bucket

        p_bar = torch.zeros(K, device=logits.device, dtype=probs.dtype) # mean probability for each bucket
        p_bar = (weights * expand_predictions).sum(dim=(0, 1)) / S.clamp_min(1e-12)

        y_bar = torch.zeros(K, device=logits.device, dtype=probs.dtype) # mean label for each bucket
        expand_labels = labels.unsqueeze(-1).expand(-1, -1, K)
        y_bar = (weights * expand_labels).sum(dim=(0, 1)) / S.clamp_min(1e-12)

        return torch.sum(S * torch.abs(p_bar - y_bar)) / N
    
    def ECE_multidim(self, probs, labels, K=5, smooth=False, sigma=0.1, weights_collaborator=None, confidence=True):
        """
        Computes ECE loss on probabilities and labels. 
        If smooth is False, uses ECE loss (K buckets).
        If smooth is True, uses smooth ECE loss (Gaussian weights centered at K points with std dev sigma).
        if weights_collaborator is not None, use it to compute cross ECE.
        """

        if confidence:
            # sample predictions from probs
            sampled_predictions = torch.distributions.Categorical(probs=probs).sample()
            # or set deterministically
            # sampled_predictions = torch.argmax(probs, dim=-1)
            # labels = (sampled_predictions == targets) # label is 1 if sampled prediction is correct, 0 otherwise
            # conf_labels[b,t] = 1 if labels[b, t, sample_predictions[b,t]] == 1, 0 otherwise
            conf_labels = torch.gather(labels, dim=-1, index=sampled_predictions.unsqueeze(-1))
            conf_labels = conf_labels.squeeze(-1)
            labels = conf_labels.unsqueeze(-1) # shape B x T x 1 (D = 1)
            # use max probabilities as predictions
            predictions = torch.max(probs, dim=-1).values.unsqueeze(-1) # shape B x T x 1 (D = 1)
        else:
            # labels are one-hot vectors of targets
            # labels = torch.zeros(probs.size(0), probs.size(1), D, device=logits.device, dtype=probs.dtype)
            # labels.scatter_(-1, targets.unsqueeze(-1), 1)
            predictions = probs
        
        D = predictions.size(-1) # prediction dimension
        N = predictions.size(0) * predictions.size(1) # number of examples (B*T)
        num_buckets = K ** D # number of buckets
              
        if smooth:
            # define D-dimensional soft buckets using Gaussian kernels
            centers_1d = torch.linspace(0, 1, K, device=probs.device, dtype=probs.dtype)
            grids = torch.meshgrid([centers_1d] * D, indexing="ij")
            centers = torch.stack(grids, dim=-1).reshape(-1, D)  # shape K^D x D
            diff = predictions.unsqueeze(-2) - centers.unsqueeze(0).unsqueeze(0)  # shape B x T x K^D x D
            diff_norm = (torch.pow(diff, 2)).sum(dim=-1)  # shape B x T x K^D; L2 norm squared across D
            unnorm_weights = torch.exp(- (diff_norm) / (2 * sigma ** 2))
            weights = unnorm_weights / unnorm_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        else:
            edges_1d = torch.linspace(0, 1, K+1, device=probs.device, dtype=probs.dtype)
            idx_per_coord = torch.bucketize(predictions, edges_1d, right=True)-1 # shape B x T x D: index of 1-dim bucket that prediction[b,t,d] is in
            idx_per_coord.clamp_(min=0, max=K-1)
            powers = (K**torch.arange(D, device=probs.device, dtype=torch.int64)) # shape D
            idx = (idx_per_coord.to(dtype=torch.int64) * powers).sum(dim=-1) # shape B x T: index of D-dim bucket that prediction[b,t] is in
            weights = F.one_hot(idx, num_classes=num_buckets).to(dtype=probs.dtype)
        
        if weights_collaborator is not None: # cross ECE buckets are joint buckets of the model and the collaborator
            joint_weights = weights_collaborator.unsqueeze(-1) * weights.unsqueeze(-2) # shape B x T x L x K^D
            joint_weights = joint_weights.reshape(probs.size(0), probs.size(1), -1) # flatten to shape B x T x (K^D*L)
            weights = joint_weights

        S = weights.sum(dim=(0, 1)) # sum of weights for each bucket

        p_bar = (weights.unsqueeze(-1) * predictions.unsqueeze(-2)).sum(dim=(0, 1)) / S.clamp_min(1e-12).unsqueeze(-1)
        y_bar = (weights.unsqueeze(-1) * labels.unsqueeze(-2)).sum(dim=(0, 1)) / S.clamp_min(1e-12).unsqueeze(-1)

        ece_loss = (S * torch.linalg.norm(p_bar - y_bar, ord=2, dim=-1)).sum() / N

        return ece_loss   

    def ECE_multidim_new(self, probs, labels, K=5, K_cross=5, smooth=False, sigma=0.1, collaborator_probs=None, confidence=True):
        """
        Computes ECE loss on probabilities and labels. 
        If smooth is False, uses ECE loss (K buckets).
        If smooth is True, uses smooth ECE loss (Gaussian weights centered at K points with std dev sigma).
        if collaborator_probs is not None, use it to compute cross ECE.
        """

        if confidence:
            # sample predictions from probs
            sampled_predictions = torch.distributions.Categorical(probs=probs).sample()
            conf_labels = torch.gather(labels, dim=-1, index=sampled_predictions.unsqueeze(-1))
            conf_labels = conf_labels.squeeze(-1)
            labels = conf_labels.unsqueeze(-1) # shape B x T x 1 (D = 1)
            # use max probabilities as predictions
            predictions = torch.max(probs, dim=-1).values.unsqueeze(-1) # shape B x T x 1 (D = 1)
            # TODO: specify collaborator predictions
        else:
            predictions = probs
            if collaborator_probs is not None:
                collaborator_predictions = collaborator_probs
        
        B, T, D = predictions.shape # prediction dimension
        N = B * T # number of examples
        num_buckets = K ** D # number of buckets

        flat_predictions = predictions.reshape(N, D)
        flat_labels = labels.to(dtype=probs.dtype).reshape(N, D)
              
        if smooth: # memory intensive
            # soft bucket assignment: define D-dimensional soft buckets using Gaussian kernels
            centers_1d = torch.linspace(0, 1, K, device=probs.device, dtype=probs.dtype)
            grids = torch.meshgrid([centers_1d] * D, indexing="ij")
            centers = torch.stack(grids, dim=-1).reshape(-1, D)  # shape K^D x D
            diff = predictions.unsqueeze(-2) - centers.unsqueeze(0).unsqueeze(0)  # shape B x T x K^D x D
            diff_norm = (torch.pow(diff, 2)).sum(dim=-1)  # shape B x T x K^D; L2 norm squared across D
            unnorm_weights = torch.exp(- (diff_norm) / (2 * sigma ** 2))
            weights = unnorm_weights / unnorm_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)

            if collaborator_probs is not None: # cross ECE buckets are joint buckets of the model and the collaborator
                # define D-dimensional soft buckets using Gaussian kernels
                centers_1d_collab = torch.linspace(0, 1, K_cross, device=probs.device, dtype=probs.dtype)
                grids_collab = torch.meshgrid([centers_1d_collab] * D, indexing="ij")
                centers_collab = torch.stack(grids_collab, dim=-1).reshape(-1, D)  # shape K^D x D
                diff_collab = collaborator_predictions.unsqueeze(-2) - centers_collab.unsqueeze(0).unsqueeze(0)  # shape B x T x K_cross^D x D
                diff_norm_collab = (torch.pow(diff_collab, 2)).sum(dim=-1)  # shape B x T x K_cross^D; L2 norm across D
                unnorm_weights_collab = torch.exp(- (diff_norm_collab) / (2 * sigma ** 2))
                weights_collaborator = unnorm_weights_collab / unnorm_weights_collab.sum(dim=-1, keepdim=True).clamp_min(1e-12) 

                joint_weights = weights_collaborator.unsqueeze(-1) * weights.unsqueeze(-2) # shape B x T x K_cross^D x K^D
                joint_weights = joint_weights.reshape(probs.size(0), probs.size(1), -1) # flatten to shape B x T x (K_cross^D*K^D)
                weights = joint_weights

            S = weights.sum(dim=(0, 1)) # sum of weights for each bucket

            p_bar = (weights.unsqueeze(-1) * predictions.unsqueeze(-2)).sum(dim=(0, 1)) / S.clamp_min(1e-12).unsqueeze(-1)
            y_bar = (weights.unsqueeze(-1) * labels.unsqueeze(-2)).sum(dim=(0, 1)) / S.clamp_min(1e-12).unsqueeze(-1)
        else: # memory efficient version
            # hard bucket assignment: each example gets one D-dimensional bucket
            edges_1d = torch.linspace(0, 1, K + 1, device=probs.device, dtype=probs.dtype)
            idx_per_coord = torch.bucketize(flat_predictions, edges_1d, right=True) - 1  # N x D: index of 1-dim bucket that prediction[n,d] is in
            idx_per_coord.clamp_(min=0, max=K - 1)
            powers = (K ** torch.arange(D, device=probs.device, dtype=torch.int64))  # D
            own_idx = (idx_per_coord.to(torch.int64) * powers).sum(dim=-1)  # N: index of D-dim bucket that prediction[n] is in
            if collaborator_probs is None:
                # use active buckets only
                active_bucket_ids, inverse = torch.unique(own_idx, sorted=False, return_inverse=True)
                M = active_bucket_ids.numel()  # M <= N

                S = torch.zeros(M, device=probs.device, dtype=probs.dtype)
                wp = torch.zeros(M, D, device=probs.device, dtype=probs.dtype)
                wl = torch.zeros(M, D, device=probs.device, dtype=probs.dtype)

                ones = torch.ones(N, device=probs.device, dtype=probs.dtype)
                S.scatter_add_(0, inverse, ones) # number of examples in each bucket
                inverse_exp = inverse.unsqueeze(-1).expand(-1, D) 
                wp.scatter_add_(0, inverse_exp, flat_predictions) # weighted sum of predictions in each bucket
                wl.scatter_add_(0, inverse_exp, flat_labels) # weighted sum of labels in each bucket
            else:
                flat_collab_predictions = collaborator_predictions.reshape(N, D)
                edges_1d_collab = torch.linspace(0, 1, K_cross + 1, device=probs.device, dtype=probs.dtype)
                collab_coord_idx = torch.bucketize(flat_collab_predictions, edges_1d_collab, right=True) - 1
                collab_coord_idx.clamp_(0, K_cross - 1)
                powers_collab = (K_cross ** torch.arange(D, device=probs.device, dtype=torch.int64))
                collab_idx = (collab_coord_idx.to(torch.int64) * powers_collab).sum(dim=-1)

                joint_idx_raw = own_idx + (K ** D) * collab_idx
                _, joint_inverse = torch.unique(joint_idx_raw, sorted=False, return_inverse=True) # index of joint bucket that prediction[n] is in

                M = int(joint_inverse.max().item()) + 1 # number of joint buckets

                S = torch.zeros(M, device=probs.device, dtype=probs.dtype)
                wp = torch.zeros(M, D, device=probs.device, dtype=probs.dtype)
                wl = torch.zeros(M, D, device=probs.device, dtype=probs.dtype)

                ones = torch.ones(N, device=probs.device, dtype=probs.dtype)
                S.scatter_add_(0, joint_inverse, ones)
                joint_exp = joint_inverse.unsqueeze(-1).expand(-1, D)
                wp.scatter_add_(0, joint_exp, flat_predictions)
                wl.scatter_add_(0, joint_exp, flat_labels)

            p_bar = wp / S.clamp_min(1e-12).unsqueeze(-1)
            y_bar = wl / S.clamp_min(1e-12).unsqueeze(-1)
           
        ece_loss = (S * torch.linalg.norm(p_bar - y_bar, ord=2, dim=-1)).sum() / N

        return ece_loss

    def cross_ECE(self, logits, targets, collaborator_predictions, collaborator_probs=None, K=5, cross_probabilities=False, smooth_collaborator=False, smooth_own=False, sigma=0.1, confidence=True):
        """
        Computes cross ECE between the model and the collaborator: 
        sum_{collaborator_predictions/probs} ECE of the model conditional on collaborator's predictions/probs.
        If cross_probabilities is True, use collaborator's probabilities to compute cross ECE. Otherwise, use predictions.
        """
        if cross_probabilities: # use probabilities
            L = K # number of buckets
            # binary tasks: predictions are probabilities place on token 1
            idx = self.config.stoi["1"]
            collaborator_predictions = collaborator_probs[:, :, idx]
        else: # use predictions (smoothing only makes sense if predictions are continuous/ordered)
            L = len(torch.unique(collaborator_predictions)) # number of unique predictions

        collaborator_min = collaborator_predictions.min()
        collaborator_max = collaborator_predictions.max()

        expand_collaborator_predictions = collaborator_predictions.unsqueeze(-1).expand(-1, -1, L) # expand to size B x T x L

        if smooth_collaborator:
            # define soft buckets using Gaussian kernels
            centers_collaborator = torch.linspace(collaborator_min, collaborator_max, L, device=logits.device, dtype=logits.dtype)
            diff = collaborator_predictions.unsqueeze(-1) - centers_collaborator
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
            weights_collaborator = (expand_collaborator_predictions >= left_collaborator) & (expand_collaborator_predictions < right_collaborator)
            weights_collaborator[:, :, -1] = (collaborator_predictions >= left_collaborator[:, :, -1]) & (collaborator_predictions <= right_collaborator[:, :, -1]) # for last bucket, include right endpoint

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

    def cross_ECE_multidim(self, probs, labels, collaborator_predictions, collaborator_probs=None, K=5, cross_probabilities=False, smooth_collaborator=False, smooth_own=False, sigma=0.1, confidence=True):
        """
        Computes cross ECE between the model and the collaborator: 
        sum_{collaborator_predictions/probs} ECE of the model conditional on collaborator's predictions/probs.
        If cross_probabilities is True, use collaborator's probabilities to compute cross ECE. Otherwise, use predictions.
        """
        if cross_probabilities: # use probability vectors
            D = collaborator_probs.size(-1) # dimension
            collaborator_predictions = collaborator_probs
        else: # use predictions (smoothing only makes sense if predictions are continuous/ordered)
            D = 1
            # TODO: generalize to multidimensional setting
            K = len(torch.unique(collaborator_predictions)) # number of unique predictions
        
        num_buckets = K**D # number of buckets

        if smooth_collaborator:
            # define D-dimensional soft buckets using Gaussian kernels
            centers_1d = torch.linspace(0, 1, K, device=probs.device, dtype=probs.dtype)
            grids = torch.meshgrid([centers_1d] * D, indexing="ij")
            centers = torch.stack(grids, dim=-1).reshape(-1, D)  # shape K^D x D
            diff = collaborator_predictions.unsqueeze(-2) - centers.unsqueeze(0).unsqueeze(0)  # shape B x T x K^D x D
            diff_norm = (torch.pow(diff, 2)).sum(dim=-1)  # shape B x T x K^D; L2 norm across D
            unnorm_weights = torch.exp(- (diff_norm) / (2 * sigma ** 2))
            weights_collaborator = unnorm_weights / unnorm_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12) 
        else:
            # define D-dimensional hard buckets using uniform cells
            edges_1d = torch.linspace(0, 1, K+1, device=probs.device, dtype=probs.dtype)
            idx_per_coord = torch.bucketize(collaborator_predictions, edges_1d, right=True)-1 # shape B x T x D: index of 1-dim bucket that prediction[b,t,d] is in
            idx_per_coord.clamp_(min=0, max=K-1)
            powers = (K**torch.arange(D, device=probs.device, dtype=probs.dtype)) # shape D
            idx = (idx_per_coord * powers).sum(dim=-1) # shape B x T: index of D-dim bucket that prediction[b,t] is in
            weights_collaborator = F.one_hot(idx.to(dtype=torch.int64), num_classes=num_buckets) 

        cross_ece_loss = self.ECE_multidim(probs, labels, K=K, smooth=smooth_own, sigma=sigma, weights_collaborator=weights_collaborator, confidence=confidence)
        
        return cross_ece_loss         

    def brier_score(self, probs, labels):
        """
        Computes Brier score: sum_{i=1}^D (p_i - y_i)^2 / N.
        """
        B, T, D = probs.shape # prediction dimension
        N = B * T # number of examples
        flat_probs = probs.reshape(N, D)
        flat_labels = labels.to(dtype=probs.dtype).reshape(N, D)
        return torch.sum((flat_probs - flat_labels) ** 2) / N

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

    def agreement(self, probs, collaborator_probs):
        """
        Computes l2 distance between probabilities of the model and the collaborator.
        """
        # N = probs.size(0) * probs.size(1) # number of examples (B*T)
        # predictions = torch.distributions.Categorical(probs=probs).sample()
        # collaborator_predictions = torch.distributions.Categorical(probs=collaborator_probs).sample()
        # return torch.sum(predictions == collaborator_predictions) / N
        return torch.norm(probs - collaborator_probs, p=2, dim=-1).mean(dim=(0,1))

    def forward(self, idx, targets=None, collaborator_predictions=None, collaborator_probs=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # multi-step autoregressive generation loop
        if targets is not None:
            targets = targets[:, -self.config.m_lookahead:] # shape (b, m_lookahead): crop targets to the last m_lookahead steps
        input_idx = idx
        logits_sequence = []
        CE_loss_sequence = []
        for m in range(self.config.m_lookahead):
            # forward the GPT model itself
            tok_emb = self.transformer.wte(input_idx) # token embeddings of shape (b, t, n_embd)
            pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)
            for block in self.transformer.h:
                x = block(x)
            x = self.transformer.ln_f(x) # shape (b, t, n_embd)  
            if m == 0:
                x_first = x
                if targets is None:
                    # if no targets are provided, we are in inference mode and only need the first step
                    break 

            if targets is not None: 
                # if we are given some desired targets also calculate the loss
                logits = self.lm_head(x)
                if self.config.prefix_size > 0: # if we are using a prefix, crop the logits and targets
                    logits = logits[:, self.config.prefix_size:, :] # shape (b, t-prefix_size, vocab_size)
                    targets_next = targets[:, m:m+1] # shape (b, 1): get target for next step
                logits_sequence.append(logits)
                
                # record CE loss for this step
                CE_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets_next.reshape(-1), ignore_index=-1)
                CE_loss_sequence.append(CE_loss)
                
                if self.config.m_lookahead == 1:
                    break # if we are only generating one step, we are done
                else:
                    # generate next token and append to the running sequence
                    logits_cropped = logits[:, -1, :] # shape (b, vocab_size): last token logits
                    probs = F.softmax(logits_cropped, dim=-1)
                    if self.config.autoregressive_lookahead:
                        # sample from the distribution to get the next token
                        idx_next = torch.multinomial(probs, num_samples=1) # shape (b, 1)
                    else:
                        # use ground truth for next token
                        idx_next = targets[:, m:m+1] # shape (b, 1)
                    # append sampled index
                    input_idx = torch.cat((input_idx, idx_next), dim=1)
                    # shift input_idx by one step
                    input_idx = input_idx[:, 1:]



        # # forward the GPT model itself
        # tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        # pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        # x = self.transformer.drop(tok_emb + pos_emb)
        # for block in self.transformer.h:
        #     x = block(x)
        # x = self.transformer.ln_f(x) # shape (b, t, n_embd)  

        if targets is not None: 
            # # if we are given some desired targets also calculate the loss
            # logits = self.lm_head(x)
            # if self.config.prefix_size > 0: # if we are using a prefix, crop the logits and targets
            #     logits = logits[:, self.config.prefix_size:, :] # shape (b, t-prefix_size, vocab_size)
            #     targets = targets[:, self.config.prefix_size:] # shape (b, t-prefix_size)
            # probs = F.softmax(logits, dim=-1)

            # # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            # loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
            # # loss = torch.tensor(0.0, device=device, dtype=torch.float32)
            # CE_loss = loss
            loss = sum(CE_loss_sequence) # objective is sum of CE losses for all steps

            # record metrics for the first step
            CE_loss = CE_loss_sequence[0] # CE loss for the first step

            logits = logits_sequence[0] # logits for the first step
            probs = F.softmax(logits, dim=-1)

            targets = targets[:, 0:1] # shape (b, 1): targets for the first step

            answer_indices = torch.tensor([self.config.stoi[token] for token in self.config.answer_tokens], device=device)
            answer_probs = torch.index_select(probs, dim=-1, index=answer_indices)
            labels = (answer_indices.unsqueeze(0).unsqueeze(0) == targets.unsqueeze(-1)) 

            # set whether to use smECE/brier score in loss
            use_sm_ece = (self.config.calibrate == "smECE"
                and ((collaborator_predictions is None and collaborator_probs is None)
                    or (not self.config.cross_calibrate)))
            use_brier_score = (self.config.calibrate == "brier"
                and ((collaborator_predictions is None and collaborator_probs is None)
                    or (not self.config.cross_calibrate)))

            with torch.set_grad_enabled(use_sm_ece):
                if self.config.compute_smooth_calibration:
                    sm_ece_loss_multidim_new = self.ECE_multidim_new(answer_probs, labels, K=self.config.K, K_cross=self.config.K_cross, smooth=True, sigma=0.1, confidence=self.config.confidence)            
                else:
                    sm_ece_loss_multidim_new = torch.tensor(0.0, device=device, dtype=torch.float32)
            with torch.set_grad_enabled(use_brier_score):
                brier_score = self.brier_score(answer_probs, labels)
            
            # eval metrics
            with torch.no_grad():           
                # ece_loss_multidim = self.ECE_multidim(answer_probs, labels, K=self.config.K, smooth=False, confidence=self.config.confidence)
                # sm_ece_loss_multidim = self.ECE_multidim(answer_probs, labels, K=self.config.K, smooth=True, confidence=self.config.confidence)
                ece_loss_multidim_new = self.ECE_multidim_new(answer_probs, labels, K=self.config.K, K_cross=self.config.K_cross, smooth=False, sigma=0.1, confidence=self.config.confidence)
                zero_one_loss = self.zero_one_loss(logits, targets)
                entropy = self.entropy(logits)

                # also compute ECE loss on scaled probabilities (same as ECE loss if temperature is 1)
                answer_probs_scaled = torch.softmax(torch.log(answer_probs.clamp_min(1e-12)) / self.config.append_probabilities_temperature, dim=-1)
                ece_loss_temperature_scaled = self.ECE_multidim_new(answer_probs_scaled, labels, K=self.config.K, K_cross=self.config.K_cross, smooth=False, sigma=0.1, confidence=self.config.confidence)
            
            if collaborator_predictions is not None or collaborator_probs is not None:
                # calculate cross calibration error wrt collaborator predictions
                with torch.set_grad_enabled(self.config.cross_calibrate):
                    if self.config.compute_smooth_calibration:
                        sm_cross_ece_loss_multidim_new = self.ECE_multidim_new(answer_probs, labels, K=self.config.K, K_cross=self.config.K_cross, smooth=True, sigma=0.1, collaborator_probs=collaborator_probs, confidence=self.config.confidence)
                    else:
                        sm_cross_ece_loss_multidim_new = torch.tensor(0.0, device=device, dtype=torch.float32)
                
                with torch.no_grad():
                    # cross_ece_loss_multidim = self.cross_ECE_multidim(answer_probs, labels, collaborator_predictions, collaborator_probs, K=self.config.K, cross_probabilities=self.config.cross_probabilities, smooth_collaborator=False, smooth_own=False, confidence=self.config.confidence)
                    # sm_cross_ece_loss_multidim = self.cross_ECE_multidim(answer_probs, labels, collaborator_predictions, collaborator_probs, K=self.config.K, cross_probabilities=self.config.cross_probabilities, smooth_collaborator=True, smooth_own=True, confidence=self.config.confidence)                
                    cross_ece_loss_multidim_new = self.ECE_multidim_new(answer_probs, labels, K=self.config.K, K_cross=self.config.K_cross, smooth=False, sigma=0.1, collaborator_probs=collaborator_probs, confidence=self.config.confidence)
                    agreement = self.agreement(answer_probs, collaborator_probs)
                
                if self.config.cross_calibrate:
                    loss = loss + sm_cross_ece_loss_multidim_new * self.config.cross_multiplier
                
                if self.config.post_hoc_calibrate:
                    answer_probs = answer_probs.detach()
                    # feed own probabilities and collaborator probabilities to CalibrateMLP to get conversation calibrated probabilities
                    calibrator_logits = self.calibrate_mlp(answer_probs, collaborator_probs)
                    calibrator_probs = F.softmax(calibrator_logits, dim=-1)

                    targets_answer_idx = labels.int().argmax(dim=-1) # (B, T) index of correct answer token

                    if self.config.post_hoc_calibrate_use_smECE:
                        calibrator_loss = self.ECE_multidim_new(calibrator_probs, labels, K=self.config.post_hoc_calibrate_use_smECE_bins, K_cross=self.config.post_hoc_calibrate_use_smECE_bins, smooth=True, sigma=0.1, collaborator_probs=collaborator_probs, confidence=self.config.confidence)
                        loss = loss + calibrator_loss * self.config.post_hoc_calibrate_multiplier
                    else:
                        # use CE loss for post-hoc calibration
                        D_answer = calibrator_logits.size(-1)
                        calibrator_loss = F.cross_entropy(calibrator_logits.reshape(-1, D_answer), targets_answer_idx.reshape(-1))
                        loss = loss + calibrator_loss * self.config.post_hoc_calibrate_multiplier

                    with torch.no_grad():
                        if not self.config.post_hoc_calibrate_use_smECE:
                            # still report smECE for monitoring even when not used in the loss
                            calibrator_sm_cross_ece_loss = self.ECE_multidim_new(calibrator_probs, labels, K=self.config.post_hoc_calibrate_use_smECE_bins, K_cross=self.config.post_hoc_calibrate_use_smECE_bins, smooth=True, sigma=0.1, collaborator_probs=collaborator_probs, confidence=self.config.confidence)
                        calibrator_cross_ece_loss = self.ECE_multidim_new(calibrator_probs, labels, K=self.config.K, K_cross=self.config.K_cross, smooth=False, sigma=0.1, collaborator_probs=collaborator_probs, confidence=self.config.confidence)
                        calibrator_ece_loss = self.ECE_multidim_new(calibrator_probs, labels, K=self.config.K, K_cross=self.config.K_cross, smooth=False, sigma=0.1, confidence=self.config.confidence)
                        calibrator_entropy = self.entropy(calibrator_logits)
                        calibrator_agreement = self.agreement(calibrator_probs, collaborator_probs)
                        calibrator_zero_one_loss = self.zero_one_loss(calibrator_logits, targets_answer_idx)
                else:
                    calibrator_logits = None
                    calibrator_loss = None
                    calibrator_cross_ece_loss = None
                    calibrator_ece_loss = None
                    calibrator_entropy = None
                    calibrator_agreement = None
                    calibrator_zero_one_loss = None
            else:
                cross_ece_loss_multidim_new = None
                sm_cross_ece_loss_multidim_new = None
                agreement = None
                calibrator_logits = None
                calibrator_loss = None
                calibrator_cross_ece_loss = None
                calibrator_ece_loss = None
                calibrator_entropy = None
                calibrator_agreement = None
                calibrator_zero_one_loss = None
            
            if use_sm_ece:
                loss = loss + sm_ece_loss_multidim_new * self.config.multiplier
            if use_brier_score:
                loss = loss + brier_score * self.config.multiplier

        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x_first[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
            CE_loss = None
            brier_score = None
            zero_one_loss = None
            entropy = None
            agreement = None
            ece_loss_multidim_new = None
            sm_ece_loss_multidim_new = None
            cross_ece_loss_multidim_new = None
            sm_cross_ece_loss_multidim_new = None
            calibrator_loss = None
            calibrator_cross_ece_loss = None
            calibrator_ece_loss = None
            calibrator_entropy = None
            calibrator_agreement = None
            calibrator_zero_one_loss = None
            ece_loss_temperature_scaled = None
            # compute calibrator_logits for inference if post_hoc_calibrate is enabled and collaborator_probs provided
            if self.config.post_hoc_calibrate and collaborator_probs is not None:
                probs = F.softmax(logits, dim=-1)
                answer_indices = torch.tensor([self.config.stoi[token] for token in self.config.answer_tokens], device=device)
                answer_probs = torch.index_select(probs, dim=-1, index=answer_indices)
                calibrator_logits = self.calibrate_mlp(answer_probs, collaborator_probs)
            else:
                calibrator_logits = None

        return logits, calibrator_logits, CE_loss_sequence, loss, ece_loss_multidim_new, sm_ece_loss_multidim_new, cross_ece_loss_multidim_new, sm_cross_ece_loss_multidim_new, calibrator_loss, calibrator_cross_ece_loss, calibrator_ece_loss, calibrator_entropy, calibrator_agreement, calibrator_zero_one_loss, brier_score, zero_one_loss, entropy, agreement, ece_loss_temperature_scaled

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
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, return_probs=False, sample_from_answers=False, use_calibrator_logits=False, collaborator_probs=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        probs_list = []
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, calibrator_logits, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = self(idx_cond, collaborator_probs=collaborator_probs)
            if use_calibrator_logits:
                assert calibrator_logits is not None, "calibrator_logits must be provided when use_calibrator_logits is True"
                logits = calibrator_logits
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            probs_list.append(probs)
            answer_indices = torch.tensor([self.config.stoi[token] for token in self.config.answer_tokens], device=idx.device)
            if sample_from_answers: 
                # sample from only the answer tokens
                probs_answer = torch.index_select(probs, dim=-1, index=answer_indices)
                idx_next = torch.multinomial(probs_answer, num_samples=1)
                idx_next = answer_indices[idx_next]
            elif use_calibrator_logits:
                # calibrator_logits are already for answer tokens only, so sample and convert back to vocab indices
                idx_next = torch.multinomial(probs, num_samples=1)
                idx_next = answer_indices[idx_next]
            else:
                # sample from the entire distribution
                idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
        probs_list = torch.stack(probs_list, dim=1)
        if return_probs:
            return idx, probs_list
        else:
            return idx
