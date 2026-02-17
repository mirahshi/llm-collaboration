"""
Implementation of iteratively training agents using collaborator models.

This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
import wandb
from termcolor import colored

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
wandb_group_name = 'exp'
# number of agents
num_agents = 1 # 1 or 2
num_rounds = 1 # number of rounds of conversation
pre_load_rounds = [] # rounds to pre-load collaborator models from
save_models = False # save models after each round
# data
datasets = ['openwebtext'] * num_agents # one dataset per agent
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# prefix length to ignore in loss (overridden by config files like config/train_arithmetic.py)
prefix_size = 0
example_size = 0
# calibrate?
calibrate = 'smECE' # self-calibrate: None, 'smECE', 'brier'
multiplier = 1 # multiplier for calibration loss
cross_calibrate = True # cross-calibrate using smooth cross ECE
cross_multiplier = 1 # multiplier for cross calibration loss
confidence = True # use confidence calibration; otherwise use prediction calibration
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
causal = True
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, list))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

class Agent():
    def __init__(self, config, round, id, collaborator_model=None):
        self.config = config
        self.id = id
        self.round = round
        self.collaborator_model = collaborator_model
        if self.collaborator_model is not None:
            self.collaborator_id = (self.id - 1) % self.config['num_agents'] # collaborator is the previous agent

        # various inits, derived attributes, I/O setup
        self.ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
        if self.ddp:
            init_process_group(backend=backend)
            ddp_rank = int(os.environ['RANK'])
            ddp_local_rank = int(os.environ['LOCAL_RANK'])
            ddp_world_size = int(os.environ['WORLD_SIZE'])
            device = f'cuda:{ddp_local_rank}'
            torch.cuda.set_device(device)
            self.master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
            seed_offset = ddp_rank # each process gets a different seed
            # world_size number of processes will be training simultaneously, so we can scale
            # down the desired gradient accumulation iterations per process proportionally
            assert self.config['gradient_accumulation_steps'] % ddp_world_size == 0
            self.config['gradient_accumulation_steps'] //= ddp_world_size
        else:
            # if not ddp, we are running on a single gpu, and one process
            self.master_process = True
            seed_offset = 0
            ddp_world_size = 1
        tokens_per_iter = self.config['gradient_accumulation_steps'] * ddp_world_size * self.config['batch_size'] * self.config['block_size']
        print(f"tokens per iteration will be: {tokens_per_iter:,}")

        if self.master_process:
            os.makedirs(self.config['out_dir'], exist_ok=True)
        torch.manual_seed(1337 + seed_offset)
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
        device_type = 'cuda' if 'cuda' in self.config['device'] else 'cpu' # for later use in torch.autocast
        # note: float16 data type will automatically use a GradScaler
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
        self.ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

        self.data_dir = os.path.join('data', self.config['datasets'][id])
        # attempt to derive vocab_size and char-to-int mapping from the dataset
        meta_path = os.path.join(self.data_dir, f'meta{self.id}.pkl')
        self.meta_vocab_size = None
        self.meta_stoi = None
        self.meta_itos = None
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            self.meta_vocab_size = meta['vocab_size']
            self.meta_stoi = meta['stoi']
            self.meta_itos = meta['itos']
            print(f"found agent {self.id} vocab_size = {self.meta_vocab_size} (inside {meta_path})")
            print(f"agent {self.id} stoi: {self.meta_stoi}")
            print(f"agent {self.id} itos: {self.meta_itos}")

        # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
        self.iter_num = 0
        self.best_val_loss = 1e9
        self.base_prefix_size = self.config['prefix_size']
        self.base_block_size = self.config['block_size']

        # model init
        self.model_args = dict(n_layer=self.config['n_layer'], n_head=self.config['n_head'], n_embd=self.config['n_embd'], block_size=self.config['block_size'],
                        bias=self.config['bias'], vocab_size=None, stoi=self.meta_stoi, itos=self.meta_itos, dropout=self.config['dropout'], prefix_size=self.config['prefix_size'], example_size=self.config['example_size'], calibrate=self.config['calibrate'], multiplier=self.config['multiplier'], 
                        cross_calibrate=self.config['cross_calibrate'], cross_multiplier=self.config['cross_multiplier'], confidence=self.config['confidence'], causal=self.config['causal']) # start with model_args from command line
        # update prefix size and block size if collaborator model is used
        if self.collaborator_model is not None:
            self.config['prefix_size'] += 1
            self.config['block_size'] += 1
            self.model_args['prefix_size'] = self.config['prefix_size']
            self.model_args['block_size'] = self.config['block_size']
        
        if self.config['init_from'] == 'scratch':
            # init a new model from scratch
            print("Initializing a new model from scratch")
            # determine the vocab size we'll use for from-scratch training
            if self.meta_vocab_size is None:
                print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
            self.model_args['vocab_size'] = self.meta_vocab_size if self.meta_vocab_size is not None else 50304
            gptconf = GPTConfig(**self.model_args)
            self.model = GPT(gptconf)
        elif self.config['init_from'] == 'resume':
            print(f"Resuming training from {self.config['out_dir']}")
            # resume training from a checkpoint.
            ckpt_path = os.path.join(self.config['out_dir'], 'ckpt.pt')
            checkpoint = torch.load(ckpt_path, map_location=self.config['device'])
            checkpoint_model_args = checkpoint['model_args']
            # force these config attributes to be equal otherwise we can't even resume training
            # the rest of the attributes (e.g. dropout) can stay as desired from command line
            for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'prefix_size', 'causal', 'example_size']:
                self.model_args[k] = checkpoint_model_args[k]
            # create the model
            gptconf = GPTConfig(**self.model_args)
            self.model = GPT(gptconf)
            state_dict = checkpoint['model']
            # fix the keys of the state dictionary :(
            # honestly no idea how checkpoints sometimes get this prefix, have to debug more
            unwanted_prefix = '_orig_mod.'
            for k,v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            self.model.load_state_dict(state_dict)
            self.iter_num = checkpoint['iter_num']
            self.best_val_loss = checkpoint['best_val_loss']
        elif self.config['init_from'].startswith('gpt2'):
            print(f"Initializing from OpenAI GPT-2 weights: {self.config['init_from']}")
            # initialize from OpenAI GPT-2 weights
            override_args = dict(dropout=dropout)
            self.model = GPT.from_pretrained(self.config['init_from'], override_args)
            # read off the created config params, so we can store them into checkpoint correctly
            for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'prefix_size', 'causal', 'example_size']:
                self.model_args[k] = getattr(self.model.config, k)
        # crop down the model block size if desired, using model surgery
        if self.config['block_size'] < self.model.config.block_size:
            self.model.crop_block_size(self.config['block_size'])
            self.model_args['block_size'] = self.config['block_size'] # so that the checkpoint will have the right value
        self.model.to(self.config['device'])

        # initialize a GradScaler. If enabled=False scaler is a no-op
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.config['dtype'] == 'float16'))

        # optimizer
        self.optimizer = self.model.configure_optimizers(self.config['weight_decay'], self.config['learning_rate'], (self.config['beta1'], self.config['beta2']), device_type)
        if self.config['init_from'] == 'resume':
            self.optimizer.load_state_dict(self.config['checkpoint']['optimizer'])
        checkpoint = None # free up memory

        # compile the model
        if self.config['compile']:
            print("compiling the model... (takes a ~minute)")
            unoptimized_model = self.model
            self.model = torch.compile(self.model) # requires PyTorch 2.0

        # wrap model into DDP container
        if self.ddp:
            self.model = DDP(self.model, device_ids=[ddp_local_rank])


    def get_batch(self, split):
        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        if split == 'train':
            data = np.memmap(os.path.join(self.data_dir, f'train{self.id}.bin'), dtype=np.uint16, mode='r')
        else:
            data = np.memmap(os.path.join(self.data_dir, f'val{self.id}.bin'), dtype=np.uint16, mode='r')
        # ix = torch.randint(len(data) - block_size, (batch_size,))
        # ix is multiples of example_size and has batch size batch_size
        starting_indices = torch.arange(0, len(data), example_size)
        ix = torch.randint(len(starting_indices), (self.config['batch_size'],))
        ix = starting_indices[ix]
        x = torch.stack([torch.from_numpy((data[i:i+self.base_block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+self.base_block_size]).astype(np.int64)) for i in ix])
        if self.config['device'] == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(self.config['device'], non_blocking=True), y.pin_memory().to(self.config['device'], non_blocking=True)
        else:
            x, y = x.to(self.config['device']), y.to(self.config['device'])
        # if collaborator is True, get x batch (same indices) for collaborator model
        if self.collaborator_model is not None:
            if split == 'train':
                collaborator_data = np.memmap(os.path.join(self.data_dir, f'train{self.collaborator_id}.bin'), dtype=np.uint16, mode='r')
            else:
                collaborator_data = np.memmap(os.path.join(self.data_dir, f'val{self.collaborator_id}.bin'), dtype=np.uint16, mode='r')
            collaborator_x = torch.stack([torch.from_numpy((collaborator_data[i:i+self.base_block_size]).astype(np.int64)) for i in ix])
            if self.config['device'] == 'cuda':
                collaborator_x = collaborator_x.pin_memory().to(self.config['device'], non_blocking=True)
            else:
                collaborator_x = collaborator_x.to(self.config['device'])
        else:
            collaborator_x = None
        return x, y, collaborator_x

    def get_collaborator_predictions(self, collaborator_x):
        with self.ctx:
            # get predictions of collaborator model on inputs
            collaborator_predictions = self.collaborator_model.generate(collaborator_x, max_new_tokens=1).detach()
            # crop outputs to retrieve only the answer token
            collaborator_predictions = collaborator_predictions[:, self.base_prefix_size+1:]
            return collaborator_predictions
    
    def append_collaborator_predictions(self, x, y, collaborator_predictions):
        # append collaborator predictions to end of prefix in x
        x = torch.cat((x[:, :self.base_prefix_size], collaborator_predictions, x[:, self.base_prefix_size:]), dim=-1)
        y = torch.cat((y[:, :self.base_prefix_size-1], collaborator_predictions, y[:, self.base_prefix_size-1:]), dim=-1)
        return x, y

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.config['eval_iters'])
            ece_losses = torch.zeros(self.config['eval_iters'])
            sm_ece_losses = torch.zeros(self.config['eval_iters'])
            if self.collaborator_model is not None:
                cross_ece_losses = torch.zeros(self.config['eval_iters'])
                sm_cross_ece_losses = torch.zeros(self.config['eval_iters'])
            brier_scores = torch.zeros(self.config['eval_iters'])
            zero_one_losses = torch.zeros(self.config['eval_iters'])
            entropies = torch.zeros(self.config['eval_iters'])
            for k in range(self.config['eval_iters']):
                X, Y, collaborator_X = self.get_batch(split)
                if self.collaborator_model is not None: # get collaborator predictions and append to x
                    collaborator_predictions = self.get_collaborator_predictions(collaborator_X)
                    X, Y = self.append_collaborator_predictions(X, Y, collaborator_predictions)
                else:
                    collaborator_predictions = None
                with self.ctx:
                    logits, loss, ece_loss, sm_ece_loss, cross_ece_loss, sm_cross_ece_loss, brier_score, zero_one_loss, entropy = self.model(X, Y, collaborator_predictions, self.config['confidence']) # call foward function
                losses[k] = loss.item()
                ece_losses[k] = ece_loss.item()
                sm_ece_losses[k] = sm_ece_loss.item()
                if self.collaborator_model is not None:
                    cross_ece_losses[k] = cross_ece_loss.item()
                    sm_cross_ece_losses[k] = sm_cross_ece_loss.item()
                brier_scores[k] = brier_score.item()
                zero_one_losses[k] = zero_one_loss.item()
                entropies[k] = entropy.item()
            out[split] = losses.mean()
            out[split + '_ece_loss'] = ece_losses.mean()
            out[split + '_sm_ece_loss'] = sm_ece_losses.mean()
            if self.collaborator_model is not None:
                out[split + '_cross_ece_loss'] = cross_ece_losses.mean()
                out[split + '_sm_cross_ece_loss'] = sm_cross_ece_losses.mean()
            out[split + '_brier_score'] = brier_scores.mean()
            out[split + '_zero_one_loss'] = zero_one_losses.mean()
            out[split + '_entropy'] = entropies.mean()
        self.model.train()
        return out

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(self, it):
        # 1) linear warmup for warmup_iters steps
        if it < self.config['warmup_iters']:
            return self.config['learning_rate'] * (it + 1) / (self.config['warmup_iters'] + 1)
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.config['lr_decay_iters']:
            return self.config['min_lr']
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.config['warmup_iters']) / (self.config['lr_decay_iters'] - self.config['warmup_iters'])
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return self.config['min_lr'] + coeff * (self.config['learning_rate'] - self.config['min_lr'])

    def train(self):
        # training loop
        X, Y, collaborator_X = self.get_batch('train') # fetch the very first batch
        if self.collaborator_model is not None: # get collaborator predictions and append to x
            collaborator_predictions = self.get_collaborator_predictions(collaborator_X)
            X, Y = self.append_collaborator_predictions(X, Y, collaborator_predictions)
        else:
            collaborator_predictions = None
        t0 = time.time()
        local_iter_num = 0 # number of iterations in the lifetime of this process
        raw_model = self.model.module if self.ddp else self.model # unwrap DDP container if needed
        running_mfu = -1.0
        while True:

            # determine and set the learning rate for this iteration
            lr = self.get_lr(self.iter_num) if self.config['decay_lr'] else self.config['learning_rate']
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # evaluate the loss on train/val sets and write checkpoints
            if self.iter_num % self.config['eval_interval'] == 0 and self.master_process:
                losses = self.estimate_loss()
                print(f"step {self.iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                if self.config['wandb_log']:
                    log_data = {
                        "iter": self.iter_num,
                        "train/loss": losses['train'],
                        "val/loss": losses['val'],
                        "train/ece_loss": losses['train_ece_loss'],
                        "val/ece_loss": losses['val_ece_loss'],
                        "train/sm_ece_loss": losses['train_sm_ece_loss'],
                        "val/sm_ece_loss": losses['val_sm_ece_loss'],
                        "train/brier_score": losses['train_brier_score'],
                        "val/brier_score": losses['val_brier_score'],
                        "train/zero_one_loss": losses['train_zero_one_loss'],
                        "val/zero_one_loss": losses['val_zero_one_loss'],
                        "train/entropy": losses['train_entropy'],
                        "val/entropy": losses['val_entropy'],
                        "lr": lr,
                        "mfu": running_mfu*100, # convert to percentage
                    }
                    if self.collaborator_model is not None:
                        log_data.update({
                            "train/cross_ece_loss": losses['train_cross_ece_loss'],
                            "val/cross_ece_loss": losses['val_cross_ece_loss'],
                            "train/sm_cross_ece_loss": losses['train_sm_cross_ece_loss'],
                            "val/sm_cross_ece_loss": losses['val_sm_cross_ece_loss'],
                        })
                    wandb.log(log_data)

                if losses['val'] < self.best_val_loss or self.config['always_save_checkpoint']:
                    self.best_val_loss = losses['val']
                    if self.iter_num > 0:
                        checkpoint = {
                            'model': raw_model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'model_args': self.model_args,
                            'iter_num': self.iter_num,
                            'best_val_loss': self.best_val_loss,
                            'config': config,
                        }
                        print(f"saving checkpoint to {self.config['out_dir']}")
                        torch.save(checkpoint, os.path.join(self.config['out_dir'], f'ckpt_round{self.round}.pt'))
            if self.master_process and self.config['save_models']:    
                if self.iter_num == self.config['max_iters']:
                    # save current model
                    checkpoint = {
                            'model': raw_model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'model_args': self.model_args,
                            'iter_num': self.iter_num,
                            'best_val_loss': self.best_val_loss,
                            'config': config,
                        }
                    save_path = os.path.join(self.config['out_dir'], f'ckpt_round{self.round}_agent{self.id}.pt')
                    torch.save(checkpoint, save_path)
                    print(colored(f"saved round {self.round} agent {self.id} checkpoint to {save_path}", 'light_green'))
            if self.iter_num == 0 and self.config['eval_only']:
                break

            # forward backward update, with optional gradient accumulation to simulate larger batch size
            # and using the GradScaler if data type is float16
            for micro_step in range(self.config['gradient_accumulation_steps']):
                if self.ddp:
                    # in DDP training we only need to sync gradients at the last micro step.
                    # the official way to do this is with model.no_sync() context manager, but
                    # I really dislike that this bloats the code and forces us to repeat code
                    # looking at the source of that context manager, it just toggles this variable
                    self.model.require_backward_grad_sync = (micro_step == self.config['gradient_accumulation_steps'] - 1)
                with self.ctx:
                    logits, loss, ece_loss, sm_ece_loss, cross_ece_loss, sm_cross_ece_loss, brier_score, zero_one_loss, entropy = self.model(X, Y, collaborator_predictions, self.config['confidence'])
                    loss = loss / self.config['gradient_accumulation_steps'] # scale the loss to account for gradient accumulation
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                X, Y, collaborator_X = self.get_batch('train')
                if self.collaborator_model is not None: # get collaborator predictions and append to x
                    collaborator_predictions = self.get_collaborator_predictions(collaborator_X)
                    X, Y = self.append_collaborator_predictions(X, Y, collaborator_predictions)
                else:
                    collaborator_predictions = None
                # backward pass, with gradient scaling if training in fp16
                self.scaler.scale(loss).backward()
            # clip the gradient
            if self.config['grad_clip'] != 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            # step the optimizer and scaler if training in fp16
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            self.optimizer.zero_grad(set_to_none=True)

            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if self.iter_num % self.config['log_interval'] == 0 and self.master_process:
                # get loss as float. note: this is a CPU-GPU sync point
                # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
                lossf = loss.item() * self.config['gradient_accumulation_steps']
                if local_iter_num >= 5: # let the training loop settle a bit
                    mfu = raw_model.estimate_mfu(self.config['batch_size'] * self.config['gradient_accumulation_steps'], dt)
                    running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
                print(f"iter {self.iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
            self.iter_num += 1
            local_iter_num += 1

            # termination conditions
            if self.iter_num > self.config['max_iters']:
                break

        if self.ddp:
            destroy_process_group() 

def load_model(round, agent_id):
    """Load a saved round/agent checkpoint as a ready-to-use collaborator model."""
    ckpt_path = os.path.join(config['out_dir'], f'ckpt_round{round}_agent{agent_id}.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=config['device'])
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)

    state_dict = checkpoint['model']
    # Some checkpoints may carry this wrapper prefix.
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.to(config['device'])
    model.eval()
    return model


def train_converse(config):
    """Iteratively train agents using collaborator models."""
    # training loop: agents take turns training in rounds
    current_model = None # collaborator model to be used by the next agent. in the first round, the first agent trains by itself.
    for r in range(config['num_rounds']):
        agent_id = r % config['num_agents'] # determine which agent trains this round

        if r in config['pre_load_rounds']:
            current_model = load_model(r, agent_id)
            print(colored(f"Round {r}: loaded agent {agent_id} model ==========================================", 'light_yellow'))
        else:
            print(colored(f"Round {r}: agent {agent_id} trains =================================================", 'light_yellow'))
            
            agent = Agent(config, r, agent_id, current_model)
                
            # logging
            if config['wandb_log'] and agent.master_process:
                wandb.init(project=config['wandb_project'], group=config['wandb_group_name'], name=config['wandb_run_name']+f"-round{r}-agent{agent.id}", config=config, reinit="finish_previous")
                
            agent.train()
            current_model = agent.model

    return current_model

if __name__ == "__main__":
    final_model = train_converse(config)



