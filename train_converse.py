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
import ast
import random

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from prepare import prepare
import wandb
from termcolor import colored
import tqdm

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
start_from_round = 0 # start from round 
label_starting_round = False # label the starting round dataset using the model of the previous round
save_models = False # save models after each round
# data
datasets = ['openwebtext'] * num_agents # one dataset per agent
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# prefix length to ignore in loss (overridden by config files like config/train_arithmetic.py)
prefix_size = 0
example_size = 0
# maze data formatting
maze_loc_formatting = False # format the maze locations in the data
seed = 0
# calibrate?
calibrate = 'smECE' # self-calibrate: None, 'smECE', 'brier'
multiplier = 1 # multiplier for calibration loss
cross_calibrate = True # cross-calibrate using smooth cross ECE
cross_multiplier = 1 # multiplier for cross calibration loss
confidence = False # use confidence calibration; otherwise use probability calibration
cross_probabilities = True # use collaborator's probabilities for cross calibration
K = 5 # number of buckets for self ECE
K_cross = 5 # number of buckets for cross ECE
compute_smooth_calibration = False # compute smooth calibration losses (memory intensive)
post_hoc_calibrate = False # post-hoc cross calibrate the model using the predictions of the previous round
post_hoc_calibrate_multiplier = 1.0 # multiplier for post-hoc cross calibration loss
post_hoc_calibrate_use_smECE = True # use sm cross ece loss for post-hoc calibration; if False, use CE loss
post_hoc_calibrate_use_smECE_bins = 4 # number of bins for sm cross ece loss
m_lookahead = 1 # number of autoregressive lookahead predictions to generate
autoregressive_lookahead = True # use autoregressive lookahead (otherwise, use ground truth targets)
answer_tokens = ['0', '1'] # possible answer tokens
append_predictions = True # append predictions to output file
append_argmax_predictions = False # if append_predictions is True, append the argmax prediction instead of the sampled prediction
append_probabilities = True # append probabilities to output file
append_probabilities_precision = 2 # decimal places for appended probabilities
append_probabilities_temperature = 1.0 # temperature for appended probabilities (default is 1)
prune_dataset = False # prune the dataset after each round to remove examples that are correct with 100% confidence
use_curriculum = False # if True, train on pruned train set (requires prune_dataset=True)
generate_hard_dataset = False # generate hard dataset
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
    def __init__(self, config, round, id, collaborator_model=None, seed=0):
        self.seed = seed
        self.config = config
        self.id = id
        self.round = round
        self.collaborator_model = collaborator_model
        if self.collaborator_model is not None:
            self.collaborator_id = (self.id - 1) % self.config['num_agents'] # collaborator is the previous agent
        
        # validate config
        if self.config['use_curriculum'] and not self.config['prune_dataset']:
            raise ValueError("use_curriculum=True requires prune_dataset=True to generate pruned training data")

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
        torch.manual_seed(self.seed + seed_offset)
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
        device_type = 'cuda' if 'cuda' in self.config['device'] else 'cpu' # for later use in torch.autocast
        # note: float16 data type will automatically use a GradScaler
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
        self.ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

        # self.data_dir = os.path.join('data', self.config['datasets'][id])
        self.data_dir = self.config['out_dir']
        # attempt to derive vocab_size and char-to-int mapping from the dataset
        # meta_path = os.path.join(self.data_dir, f'meta{self.id}.pkl')
        meta_path = os.path.join(self.config['out_dir'], f'meta{self.id}_round{self.round}.pkl')
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
            self.encode = lambda s: [self.meta_stoi[c] for c in s]
            self.decode = lambda l: ''.join([self.meta_itos[i] for i in l])
        self.answer_indices = torch.tensor([self.meta_stoi[token] for token in self.config['answer_tokens']], device=self.config['device'])
        print(f"answer indices: {self.answer_indices.tolist()}")

        # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
        self.iter_num = 0
        self.best_val_loss = 1e9
        
        # self.prefix_size = self.config['prefix_size']
        # self.block_size = self.config['block_size']
        # self.example_size = self.config['example_size']
        # get prefix size and answer length from input files
        with open(f'{self.data_dir}/input0_round0.txt', 'r') as f:
            lines = f.readlines()
            self.example_size = len(lines[0].split('\n')[0]) + 1 # number of characters in the input example (including '\n')
            self.block_size = self.example_size - 1 - self.config['m_lookahead']
            self.prefix_size = self.block_size - 1
            # prefix_size = len(lines[0].split('=')[0]) # number of characters in the input before the '='
            # target_size = len(lines[0].split('=')[1]) - 1 # number of characters in the target (excluding '\n')
            #assert target_size == m_lookahead, f"Target size {target_size} does not match m_lookahead {m_lookahead}"
            print(f"got example size: {self.example_size}")
            print(f"got block size: {self.block_size}")
            print(f"got prefix size: {self.prefix_size}")
        
        # update prefix size, block size, and example size if collaborator model is used
        if self.collaborator_model is not None:
            if self.config['append_predictions']:
                self.prefix_size += 2 # prediction and delimiter
                self.block_size += 2
                self.example_size += 2
            if self.config['append_probabilities']:
                num_answer_tokens = len(self.config['answer_tokens'])
                tokens_per_answer = 3 + self.config['append_probabilities_precision'] # e.g. 0.85,
                self.prefix_size += num_answer_tokens * tokens_per_answer
                self.block_size += num_answer_tokens * tokens_per_answer
                self.example_size += num_answer_tokens * tokens_per_answer
        print(f"prefix size: {self.prefix_size}, block size: {self.block_size}, example size: {self.example_size}")
        assert self.block_size == self.prefix_size + 1, f"block size {self.block_size} is not equal to prefix size {self.prefix_size} + 1"
        
        # model init
        self.model_args = dict(n_layer=self.config['n_layer'], n_head=self.config['n_head'], n_embd=self.config['n_embd'], block_size=self.block_size,
                        bias=self.config['bias'], vocab_size=None, stoi=self.meta_stoi, itos=self.meta_itos, dropout=self.config['dropout'], prefix_size=self.prefix_size, example_size=self.example_size, calibrate=self.config['calibrate'], multiplier=self.config['multiplier'], 
                        cross_calibrate=self.config['cross_calibrate'], cross_multiplier=self.config['cross_multiplier'], confidence=self.config['confidence'], causal=self.config['causal'], cross_probabilities=self.config['cross_probabilities'], 
                        K=self.config['K'], K_cross=self.config['K_cross'], compute_smooth_calibration=self.config['compute_smooth_calibration'], m_lookahead=self.config['m_lookahead'], 
                        autoregressive_lookahead=self.config['autoregressive_lookahead'], answer_tokens=self.config['answer_tokens'], append_probabilities_temperature=self.config['append_probabilities_temperature'],
                        post_hoc_calibrate=self.config['post_hoc_calibrate'], post_hoc_calibrate_multiplier=self.config['post_hoc_calibrate_multiplier'], post_hoc_calibrate_use_smECE=self.config['post_hoc_calibrate_use_smECE'], post_hoc_calibrate_use_smECE_bins=self.config['post_hoc_calibrate_use_smECE_bins']) # start with model_args from command line

        
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
            incompatible = self.model.load_state_dict(state_dict, strict=False)
            missing_keys = list(incompatible.missing_keys)
            unexpected_keys = list(incompatible.unexpected_keys)
            # Backward compatibility: older checkpoints may not have calibrator weights.
            allowed_missing_prefixes = ("calibrate_mlp.",)
            disallowed_missing = [k for k in missing_keys if not k.startswith(allowed_missing_prefixes)]
            if disallowed_missing or unexpected_keys:
                raise RuntimeError(
                    f"Error(s) in loading state_dict for GPT:\n"
                    f"  Missing key(s): {disallowed_missing}\n"
                    f"  Unexpected key(s): {unexpected_keys}"
                )
            if missing_keys:
                print(colored(f"ignoring missing checkpoint keys: {missing_keys}", "light_yellow"))
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
        # # crop down the model block size if desired, using model surgery
        # if self.config['block_size'] < self.model.config.block_size:
        #     self.model.crop_block_size(self.config['block_size'])
        #     self.model_args['block_size'] = self.config['block_size'] # so that the checkpoint will have the right value
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

        # parse and store collaborator probabilities
        if self.collaborator_model is not None:
            t0 = time.time()
            data_dir = self.config['out_dir']
            # always load from combined file (needed for original train/val evaluation)
            file_name_suffix = f'{self.id}_round{self.round}'
            input_file = os.path.join(data_dir, f'input{file_name_suffix}.txt')
            with open(input_file, 'r') as f:
                lines = f.readlines()
                delimiter = ';'
                collaborator_probs_list = [line.split(delimiter)[-1] for line in lines] # get last element
                collaborator_probs_list = [ast.literal_eval(prob) for prob in collaborator_probs_list]
            self.collaborator_probs = torch.tensor(collaborator_probs_list, device=self.config['device'], dtype=torch.float32)

            if self.config['prune_dataset'] and self.round > 0:
                # load collaborator probabilities from pruned train set
                pruned_train_file = os.path.join(data_dir, f'input{self.id}_round{self.round}_pruned_train.txt')
                with open(pruned_train_file, 'r') as f:
                    lines = f.readlines()
                    delimiter = ';'
                    collaborator_probs_list = [line.split(delimiter)[-1] for line in lines] # get last element
                    collaborator_probs_list = [ast.literal_eval(prob) for prob in collaborator_probs_list]
                self.collaborator_probs_train = torch.tensor(collaborator_probs_list, device=self.config['device'], dtype=torch.float32)
                print(f"Loaded collaborator probabilities from pruned train set: {pruned_train_file}")
                
                # load collaborator probabilities from pruned val set
                pruned_val_file = os.path.join(data_dir, f'input{self.id}_round{self.round}_pruned_val.txt')
                with open(pruned_val_file, 'r') as f:
                    lines = f.readlines()
                    delimiter = ';'
                    collaborator_probs_list = [line.split(delimiter)[-1] for line in lines] # get last element
                    collaborator_probs_list = [ast.literal_eval(prob) for prob in collaborator_probs_list]
                self.collaborator_probs_val = torch.tensor(collaborator_probs_list, device=self.config['device'], dtype=torch.float32)
                print(f"Loaded collaborator probabilities from pruned val set: {pruned_val_file}")
            t1 = time.time()
            print(f"Time taken for fetching collaborator probabilities: {t1 - t0} seconds")

    def get_batch(self, split, num_examples=None):
        """
        split: 'train' or 'val' or 'val_pruned' (to do eval on pruned dataset)
        """
        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122

        data_dir = self.config['out_dir']
        file_name_suffix = f'{self.id}_round{self.round}'
        
        if split == 'train':
            if self.config['use_curriculum'] and self.round > 0: # if use_curriculum, train on pruned dataset
                data = np.memmap(os.path.join(data_dir, f'train{file_name_suffix}_pruned.bin'), dtype=np.uint16, mode='r')
            else:
                data = np.memmap(os.path.join(data_dir, f'train{file_name_suffix}.bin'), dtype=np.uint16, mode='r')
        elif split == 'val': # original validation set
            data = np.memmap(os.path.join(data_dir, f'val{file_name_suffix}.bin'), dtype=np.uint16, mode='r')
        elif split == 'val_pruned': # pruned validation set
            data = np.memmap(os.path.join(data_dir, f'val{file_name_suffix}_pruned.bin'), dtype=np.uint16, mode='r')
        # ix = torch.randint(len(data) - block_size, (batch_size,))
        # ix_examples is randomly sampled example numbers in the batch
        # ix is the starting indices of the examples in the batch (ix is multiples of example_size)
        if num_examples is None:
            num_examples = self.config['batch_size']
        starting_indices = torch.arange(0, len(data), self.example_size)
        ix_examples = torch.randint(len(starting_indices), (num_examples,))
        ix = starting_indices[ix_examples]
        x = torch.stack([torch.from_numpy((data[i:i+self.block_size]).astype(np.int64)) for i in ix])
        # y = torch.stack([torch.from_numpy((data[i+1:i+1+self.block_size]).astype(np.int64)) for i in ix])
        # shift target by m_lookahead steps
        y = torch.stack([torch.from_numpy((data[i+self.config['m_lookahead']:i+self.config['m_lookahead']+self.block_size]).astype(np.int64)) for i in ix])
        if self.config['device'] == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(self.config['device'], non_blocking=True), y.pin_memory().to(self.config['device'], non_blocking=True)
        else:
            x, y = x.to(self.config['device']), y.to(self.config['device'])

        # if collaborator is True, get collaborator probabilities (stored in input file)
        if self.collaborator_model is not None:
            if split == 'train':
                if self.config['use_curriculum'] and self.round > 0:
                    # training on pruned train set, use pruned train collaborator probs
                    collaborator_probs = self.collaborator_probs_train[ix_examples]
                else:
                    # training on original train set, use first 90% of combined collaborator probs
                    collaborator_probs = self.collaborator_probs[ix_examples]
            elif split == 'val':
                # use original val probs (from combined file with offset)
                num_examples_combined = self.collaborator_probs.shape[0]
                split_idx = int(num_examples_combined*0.9)
                collaborator_probs = self.collaborator_probs[split_idx+ix_examples]
            elif split == 'val_pruned':
                # use pruned val collaborator probs
                collaborator_probs = self.collaborator_probs_val[ix_examples]
            collaborator_probs = collaborator_probs.unsqueeze(1) # shape (B, 1, V)
        else:
            collaborator_probs = None  
        return x, y, collaborator_probs
    
    @torch.no_grad()
    def label_dataset(self, input_path, label_path, output_path, batch_size=2000):
        """
        Label dataset specified by label_path by appending predictions generated by the model on inputs in input_path.
        The labelled dataset is saved to output_path.
        """
        if self.config['prune_dataset']:
            output_path_pruned_train = output_path.replace('.txt', '_pruned_train.txt')
            output_path_pruned_val = output_path.replace('.txt', '_pruned_val.txt')
            print(f"pruned dataset will be saved to {output_path_pruned_train} and {output_path_pruned_val}")
            with open(output_path_pruned_train, 'w') as f:
                pass
            with open(output_path_pruned_val, 'w') as f:
                pass
        with open(output_path, 'w') as f:
            pass
        with open(input_path, 'r', encoding='utf-8') as f_input, open(label_path, 'r', encoding='utf-8') as f_label:
            f_input_lines = f_input.readlines()
            f_label_lines = f_label.readlines()
            f_input.seek(0)  # reset file pointer to start
            f_label.seek(0)  # reset file pointer to start
            num_examples = len(f_input_lines)
            num_batches = num_examples // batch_size
            for batch_idx in tqdm.tqdm(range(num_batches)):
                start_line_idx = batch_idx * batch_size
                input_lines = f_input_lines[start_line_idx:start_line_idx + batch_size]
                starts = [input_line[:self.block_size] for input_line in input_lines]
                start_ids = [self.encode(start) for start in starts]
                start_ids = torch.tensor(start_ids, dtype=torch.long, device=self.config['device'])
                if self.config['post_hoc_calibrate'] and self.collaborator_model is not None:
                    # extract collaborator_probs from input file for this batch (collaborator probs are in input file, not label file)
                    delimiter = ';'
                    collaborator_probs_list = [line.split(delimiter)[-1] for line in input_lines]
                    collaborator_probs_list = [ast.literal_eval(prob) for prob in collaborator_probs_list]
                    collaborator_probs = torch.tensor(collaborator_probs_list, device=self.config['device'], dtype=torch.float32)
                    collaborator_probs = collaborator_probs.unsqueeze(1)  # shape (B, 1, num_answer_tokens)
                    y, probs = self.model.generate(start_ids, max_new_tokens=1, return_probs=True, sample_from_answers=False, use_calibrator_logits=True, collaborator_probs=collaborator_probs)
                    answer_probs = probs
                else:
                    y, probs = self.model.generate(start_ids, max_new_tokens=1, return_probs=True, sample_from_answers=True)
                    # get probabilities of answer tokens
                    answer_probs = torch.index_select(probs, dim=-1, index=self.answer_indices)
                # crop outputs to retrieve only the answer token
                y = y[:, self.block_size:]
                
                if not self.config['append_argmax_predictions']:
                    # sampled predictions
                    predictions = [self.decode(y[i].tolist()) for i in range(len(y))]   
                else:
                    # argmax predictions
                    max_indices = torch.argmax(answer_probs, dim=-1)
                    original_indices = self.answer_indices[max_indices]
                    predictions = [self.decode(original_indices[i].tolist()) for i in range(len(original_indices))]
                # apply temperature scaling to communicated probabilities
                T = self.config['append_probabilities_temperature']
                answer_probs_scaled = torch.softmax(torch.log(answer_probs.clamp_min(1e-12)) / T, dim=-1)
                # if T == 1.0:
                    # assert torch.allclose(answer_probs, answer_probs_scaled, atol=1e-4), f"answer_probs and answer_probs_scaled are not the same when T = 1.0:\nanswer_probs = {answer_probs}\nanswer_probs_scaled = {answer_probs_scaled}"

                # write predictions to label file
                label_lines = f_label_lines[start_line_idx:start_line_idx + batch_size]
                output_lines = []
                pruned_train_lines = []
                pruned_val_lines = []
                split_idx = int(num_examples * 0.9)
                if self.config['generate_hard_dataset']:
                    hard_dataset_lines0 = []
                    hard_dataset_lines1 = []
                    with open(os.path.join(self.config['out_dir'], 'input0_round0.txt'), 'r') as f:
                        lines0 = f.readlines()
                    with open(os.path.join(self.config['out_dir'], 'input1_round0.txt'), 'r') as f:
                        lines1 = f.readlines()

                
                for i, (label_line, prediction, prob, prob_scaled) in enumerate(zip(label_lines, predictions, answer_probs, answer_probs_scaled)):
                    remove_example = False
                    if self.config['prune_dataset']:
                        # if answer_probs has prob 1 on the correct answer, skip it
                        correct_answer = label_line.rstrip('\n')[-1]
                        correct_idx = self.config['answer_tokens'].index(correct_answer)
                        prob_correct_answer = prob[0, correct_idx].item()
                        if prob_correct_answer > 0.999:
                            if torch.rand(1).item() > 0.1: # keep easy examples with probability 0.1
                                remove_example = True
                    
                    output_line = label_line.rstrip('\n')
                    # write probabilities, prediction, and label to output file
                    if self.config['append_predictions']:
                        output_line = f"{prediction},{output_line}"
                    if self.config['append_probabilities']:
                        # round probabilities to specified number of decimal places
                        prob_string = ','.join(f"{p:.{self.config['append_probabilities_precision']}f}" for p in prob_scaled[0].tolist())
                        output_line = f"{prob_string},{output_line}"
                    
                    full_line = f"{output_line};{prob[0].tolist()};{prob_scaled[0].tolist()}\n"
                    output_lines.append(full_line)
                    
                    if self.config['prune_dataset'] and not remove_example:
                        example_idx = start_line_idx + i
                        if example_idx < split_idx:  # write to pruned train set
                            pruned_train_lines.append(full_line)
                        else:  # write to pruned val set
                            pruned_val_lines.append(full_line)
                        if self.config['generate_hard_dataset']:
                            # generate hard dataset
                            hard_dataset_lines0.append(lines0[example_idx])
                            hard_dataset_lines1.append(lines1[example_idx])
                
                # Batch write to files
                with open(output_path, "a", encoding="utf-8") as out:
                    out.writelines(output_lines)
                if self.config['prune_dataset']:
                    if pruned_train_lines:
                        with open(output_path_pruned_train, "a", encoding="utf-8") as out:
                            out.writelines(pruned_train_lines)
                    if pruned_val_lines:
                        with open(output_path_pruned_val, "a", encoding="utf-8") as out:
                            out.writelines(pruned_val_lines)
                if self.config['generate_hard_dataset']:
                    with open(os.path.join(self.config['out_dir'], 'hard_input0_round0.txt'), 'a', encoding="utf-8") as out:
                        out.writelines(hard_dataset_lines0)
                    with open(os.path.join(self.config['out_dir'], 'hard_input1_round0.txt'), 'a', encoding="utf-8") as out:
                        out.writelines(hard_dataset_lines1)
            
            if self.config['prune_dataset']:
                num_examples = len(f_label_lines)
                # print number of examples in output file after pruning
                with open(output_path_pruned_train, 'r') as f:
                    num_train_after_pruning = len(f.readlines())
                with open(output_path_pruned_val, 'r') as f:
                    num_val_after_pruning = len(f.readlines())
                fraction_train_after_pruning = num_train_after_pruning / num_examples
                fraction_val_after_pruning = num_val_after_pruning / num_examples
                print(colored(f"Number of examples in train dataset after pruning: {num_train_after_pruning}", 'light_blue'))
                print(colored(f"Number of examples in val dataset after pruning: {num_val_after_pruning}", 'light_blue'))
                print(colored(f"Percentage of examples remaining in train dataset: {fraction_train_after_pruning * 100}%", 'light_blue'))
                print(colored(f"Percentage of examples remaining in val dataset: {fraction_val_after_pruning * 100}%", 'light_blue'))
                if self.config['wandb_log'] and wandb.run is not None:
                    wandb.log({"data/fraction_train_after_pruning": fraction_train_after_pruning})
                    wandb.log({"data/fraction_val_after_pruning": fraction_val_after_pruning})

        next_agent_id = (self.id + 1) % self.config['num_agents']
        file_name_suffix = f'{next_agent_id}_round{self.round+1}'
        prepare(output_path, self.config['out_dir'], file_name_suffix)
        if self.config['prune_dataset']:
            file_name_suffix_pruned = file_name_suffix + '_pruned'
            prepare(output_path_pruned_train, self.config['out_dir'], file_name_suffix_pruned, split='train')
            prepare(output_path_pruned_val, self.config['out_dir'], file_name_suffix_pruned, split='val')
 
    @torch.no_grad()
    def maze_success_rate(self, num_mazes=1000, use_pruned=False):
        if use_pruned:
            input_path = os.path.join(self.data_dir, f'input{self.id}_round{self.round}_pruned_val.txt')
        else:
            input_path = os.path.join(self.data_dir, f'input{self.id}_round{self.round}.txt')
        with open(input_path, 'r') as input_file:
            lines = input_file.readlines()
            if use_pruned:
                # pruned val file already contains only validation examples
                pass
            else:
                # get validation examples from combined file
                num_examples = len(lines)
                split_idx = int(num_examples*0.9)
                lines = lines[split_idx:]
            examples = [line.split(';')[0].rstrip('\n') for line in lines]
            # maze_starting_indices = [i for i, example in enumerate(examples) if example[-2] == '=']
            if self.config['maze_loc_formatting']: # non-padded formatting
                if self.config['append_probabilities'] or self.config['append_predictions']:
                    maze_examples = [example.split(',')[-1] for example in examples]
                else:
                    maze_examples = examples
                maze_starting_indices = [i for i, example in enumerate(maze_examples) if example[0] == '@']
            else: # padded formatting
                maze_starting_indices = [i for i, example in enumerate(examples) if example[-self.config['m_lookahead']-1] == '=']
            num_available_mazes = len(maze_starting_indices)
            if num_available_mazes < num_mazes:
                print(colored(f"Warning: only {num_available_mazes} mazes available, requested {num_mazes}", 'light_red'))
                num_mazes = num_available_mazes
            sampled_mazes = random.sample(range(len(maze_starting_indices)), num_mazes) 

            num_successes = 0
            for i in sampled_mazes:
                if i == len(maze_starting_indices) - 1:
                    maze_lines = examples[maze_starting_indices[i]:]
                else:
                    maze_lines = examples[maze_starting_indices[i]:maze_starting_indices[i+1]] # get examples for current maze
                # generate predictions for sampled maze examples
                starts = [example[:self.block_size] for example in maze_lines]
                start_ids = [self.encode(start) for start in starts]
                start_ids = torch.tensor(start_ids, dtype=torch.long, device=self.config['device'])
                if self.config['post_hoc_calibrate'] and self.collaborator_model is not None:
                    # extract collaborator_probs from input lines for this batch
                    if i == len(maze_starting_indices) - 1:
                        collab_lines = lines[maze_starting_indices[i]:]
                    else:
                        collab_lines = lines[maze_starting_indices[i]:maze_starting_indices[i+1]]
                    delimiter = ';'
                    collaborator_probs_list = [line.split(delimiter)[-1] for line in collab_lines]
                    collaborator_probs_list = [ast.literal_eval(prob) for prob in collaborator_probs_list]
                    collaborator_probs = torch.tensor(collaborator_probs_list, device=self.config['device'], dtype=torch.float32)
                    collaborator_probs = collaborator_probs.unsqueeze(1)  # shape (B, 1, num_answer_tokens)
                    y, _ = self.model.generate(start_ids, max_new_tokens=1, return_probs=True, sample_from_answers=False, use_calibrator_logits=True, collaborator_probs=collaborator_probs)
                else:
                    y, _ = self.model.generate(start_ids, max_new_tokens=1, return_probs=True, sample_from_answers=True)
                # crop outputs to retrieve only the answer token
                y = y[:, self.block_size:]
                predictions = [self.decode(y[i].tolist()) for i in range(len(y))]
                # targets = [example[-1] for example in maze_lines]
                targets = [example[-self.config['m_lookahead']] for example in maze_lines] # get next token
                if all(pred == tgt for pred, tgt in zip(predictions, targets)):
                    num_successes += 1
            return num_successes / num_mazes


    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.config['eval_iters'])
            # CE_losses = torch.zeros(self.config['eval_iters'])
            CE_loss_sequences = [torch.zeros(self.config['eval_iters']) for m in range(0, self.config['m_lookahead'])]
            ece_loss_multidims_new = torch.zeros(self.config['eval_iters'])
            sm_ece_loss_multidims_new = torch.zeros(self.config['eval_iters'])
            ece_loss_temperatures_scaled = torch.zeros(self.config['eval_iters'])
            if self.collaborator_model is not None:
                cross_ece_loss_multidims_new = torch.zeros(self.config['eval_iters'])
                sm_cross_ece_loss_multidims_new = torch.zeros(self.config['eval_iters'])
                if self.config['post_hoc_calibrate']:
                    calibrator_sm_cross_ece_losses = torch.zeros(self.config['eval_iters'])
                    calibrator_cross_ece_losses = torch.zeros(self.config['eval_iters'])
                    calibrator_ece_losses = torch.zeros(self.config['eval_iters'])
                    calibrator_entropies = torch.zeros(self.config['eval_iters'])
                    calibrator_agreements = torch.zeros(self.config['eval_iters'])
                    calibrator_zero_one_losses = torch.zeros(self.config['eval_iters'])
                else:
                    calibrator_sm_cross_ece_losses = None
                    calibrator_cross_ece_losses = None
                    calibrator_ece_losses = None
                    calibrator_entropies = None
                    calibrator_agreements = None
                    calibrator_zero_one_losses = None
            brier_scores = torch.zeros(self.config['eval_iters'])
            zero_one_losses = torch.zeros(self.config['eval_iters'])
            entropies = torch.zeros(self.config['eval_iters'])
            agreements = torch.zeros(self.config['eval_iters'])
            for k in range(self.config['eval_iters']):
                X, Y, collaborator_probs = self.get_batch(split)
                collaborator_predictions = None

                with self.ctx:
                    logits, calibrator_logits, CE_loss_sequence, loss, ece_loss_multidim_new, sm_ece_loss_multidim_new, cross_ece_loss_multidim_new, sm_cross_ece_loss_multidim_new, calibrator_sm_cross_ece_loss, calibrator_cross_ece_loss, calibrator_ece_loss, calibrator_entropy, calibrator_agreement, calibrator_zero_one_loss, brier_score, zero_one_loss, entropy, agreement, ece_loss_temperature_scaled = self.model(X, Y, collaborator_predictions, collaborator_probs) # call foward function
                # CE_losses[k] = CE_loss.item()
                for m in range(0, self.config['m_lookahead']):
                    CE_loss_sequences[m][k] = CE_loss_sequence[m].item()
                losses[k] = loss.item()
                ece_loss_multidims_new[k] = ece_loss_multidim_new.item()
                sm_ece_loss_multidims_new[k] = sm_ece_loss_multidim_new.item()
                ece_loss_temperatures_scaled[k] = ece_loss_temperature_scaled.item()
                if self.collaborator_model is not None:
                    cross_ece_loss_multidims_new[k] = cross_ece_loss_multidim_new.item()
                    sm_cross_ece_loss_multidims_new[k] = sm_cross_ece_loss_multidim_new.item()
                    agreements[k] = agreement.item()
                    if self.config['post_hoc_calibrate']:
                        calibrator_sm_cross_ece_losses[k] = calibrator_sm_cross_ece_loss.item()
                        calibrator_cross_ece_losses[k] = calibrator_cross_ece_loss.item()
                        calibrator_ece_losses[k] = calibrator_ece_loss.item()
                        calibrator_entropies[k] = calibrator_entropy.item()
                        calibrator_agreements[k] = calibrator_agreement.item()
                        calibrator_zero_one_losses[k] = calibrator_zero_one_loss.item()
                brier_scores[k] = brier_score.item()
                zero_one_losses[k] = zero_one_loss.item()
                entropies[k] = entropy.item()
            out[split] = losses.mean()
            # out[split + '_CE_loss'] = CE_losses.mean()
            for m in range(0, self.config['m_lookahead']):
                out[split + f'_CE_loss_step{m+1}'] = CE_loss_sequences[m].mean()
            out[split + '_ece_loss_multidim_new'] = ece_loss_multidims_new.mean()
            out[split + '_sm_ece_loss_multidim_new'] = sm_ece_loss_multidims_new.mean()
            out[split + '_ece_loss_temperature_scaled'] = ece_loss_temperatures_scaled.mean()
            if self.collaborator_model is not None:
                out[split + '_cross_ece_loss_multidim_new'] = cross_ece_loss_multidims_new.mean()
                out[split + '_sm_cross_ece_loss_multidim_new'] = sm_cross_ece_loss_multidims_new.mean()
                out[split + '_agreement'] = agreements.mean()
                if self.config['post_hoc_calibrate']:
                    out[split + '_calibrator_sm_cross_ece_loss'] = calibrator_sm_cross_ece_losses.mean()
                    out[split + '_calibrator_cross_ece_loss'] = calibrator_cross_ece_losses.mean()
                    out[split + '_calibrator_ece_loss'] = calibrator_ece_losses.mean()
                    out[split + '_calibrator_entropy'] = calibrator_entropies.mean()
                    out[split + '_calibrator_agreement'] = calibrator_agreements.mean()
                    out[split + '_calibrator_zero_one_loss'] = calibrator_zero_one_losses.mean()
            out[split + '_brier_score'] = brier_scores.mean()
            out[split + '_zero_one_loss'] = zero_one_losses.mean()
            out[split + '_entropy'] = entropies.mean()
        
        # also compute calibration on larger val dataset
        t0 = time.time()
        X_val, Y_val, collaborator_probs_val = self.get_batch('val', num_examples=10000)
        collaborator_predictions_val = None
        with self.ctx:
            _,_,_,_,ece_loss_multidim_val_new, sm_ece_loss_multidim_val_new, cross_ece_loss_multidim_val_new, sm_cross_ece_loss_multidim_val_new,_,_,_,_,_,_,_,_,_,_,_ = self.model(X_val, Y_val, collaborator_predictions_val, collaborator_probs_val)
        ece_loss_val = ece_loss_multidim_val_new.item()
        sm_ece_loss_val = sm_ece_loss_multidim_val_new.item()
        out['val_ece_loss_val'] = ece_loss_val
        out['val_sm_ece_loss_val'] = sm_ece_loss_val
        if self.collaborator_model is not None:
            cross_ece_loss_multidim_val = cross_ece_loss_multidim_val_new.item()
            sm_cross_ece_loss_multidim_val = sm_cross_ece_loss_multidim_val_new.item()
            out['val_cross_ece_loss_val'] = cross_ece_loss_multidim_val
            out['val_sm_cross_ece_loss_val'] = sm_cross_ece_loss_multidim_val
        t1 = time.time()
        print(f"Time taken for computing calibration on larger val dataset: {t1 - t0} seconds")

        # compute maze success rate on val set
        if self.config['datasets'][self.id] == 'maze':
            t0 = time.time()
            maze_success_rate = self.maze_success_rate(num_mazes=1000)
            out['val_maze_success_rate'] = maze_success_rate
            t1 = time.time()
            print(f"Time taken for computing maze success rate: {t1 - t0} seconds")
        
        # also evaluate on pruned validation set if exists
        if self.config['prune_dataset'] and self.round > 0:
            losses_pruned = torch.zeros(self.config['eval_iters'])
            zero_one_losses_pruned = torch.zeros(self.config['eval_iters'])
            for k in range(self.config['eval_iters']):
                X, Y, collaborator_probs = self.get_batch('val_pruned')
                collaborator_predictions = None
                with self.ctx:
                    logits, calibrator_logits, CE_loss_sequence, loss, ece_loss_multidim_new, sm_ece_loss_multidim_new, cross_ece_loss_multidim_new, sm_cross_ece_loss_multidim_new, calibrator_sm_cross_ece_loss, calibrator_cross_ece_loss, calibrator_ece_loss, calibrator_entropy, calibrator_agreement, calibrator_zero_one_loss, brier_score, zero_one_loss, entropy, agreement, ece_loss_temperature_scaled = self.model(X, Y, collaborator_predictions, collaborator_probs)
                losses_pruned[k] = loss.item()
                zero_one_losses_pruned[k] = zero_one_loss.item()
            out['val_pruned'] = losses_pruned.mean()
            out['val_pruned_zero_one_loss'] = zero_one_losses_pruned.mean()
            
            # compute maze success rate on pruned val set
            if self.config['datasets'][self.id] == 'maze':
                t0 = time.time()
                maze_success_rate_pruned = self.maze_success_rate(num_mazes=1000, use_pruned=True)
                out['val_pruned_maze_success_rate'] = maze_success_rate_pruned
                t1 = time.time()
                print(f"Time taken for computing maze success rate on pruned val: {t1 - t0} seconds")

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
        X, Y, collaborator_probs = self.get_batch('train') # fetch the very first batch
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
                    # Keep x-axis aligned across runs that start at later rounds.
                    # Use log step count (not raw iters) so plots align correctly.
                    # +1 because we eval at both iter_num=0 and iter_num=max_iters
                    evals_per_round = (self.config['max_iters'] // self.config['eval_interval']) + 1
                    global_step = self.round * evals_per_round + (self.iter_num // self.config['eval_interval'])
                    global_iter = self.round * self.config['max_iters'] + self.iter_num
                    log_data = {
                        "iter": global_iter,
                        "step": global_step,
                        "round_iter": self.iter_num,
                        "round": self.round,
                        "train/loss": losses['train'],
                        "val/loss": losses['val'],
                        # "train/CE_loss": losses['train_CE_loss'],
                        # "val/CE_loss": losses['val_CE_loss'],
                        "train/ece_loss_multidim_new": losses['train_ece_loss_multidim_new'],
                        "val/ece_loss_multidim_new": losses['val_ece_loss_multidim_new'],
                        "train/sm_ece_loss_multidim_new": losses['train_sm_ece_loss_multidim_new'],
                        "val/sm_ece_loss_multidim_new": losses['val_sm_ece_loss_multidim_new'],
                        "train/ece_loss_temperature_scaled": losses['train_ece_loss_temperature_scaled'],
                        "val/ece_loss_temperature_scaled": losses['val_ece_loss_temperature_scaled'],
                        "train/brier_score": losses['train_brier_score'],
                        "val/brier_score": losses['val_brier_score'],
                        "train/zero_one_loss": losses['train_zero_one_loss'],
                        "val/zero_one_loss": losses['val_zero_one_loss'],
                        "train/entropy": losses['train_entropy'],
                        "val/entropy": losses['val_entropy'],
                        "val/ece_loss_val": losses['val_ece_loss_val'],
                        "val/sm_ece_loss_val": losses['val_sm_ece_loss_val'],
                        "lr": lr,
                        "mfu": running_mfu*100, # convert to percentage
                    }
                    for m in range(0, self.config['m_lookahead']):
                        log_data.update({
                            f"train/CE_loss_step{m+1}": losses[f'train_CE_loss_step{m+1}'],
                            f"val/CE_loss_step{m+1}": losses[f'val_CE_loss_step{m+1}'],
                        })
                    if self.collaborator_model is not None:
                        log_data.update({
                            # "train/cross_ece_loss": losses['train_cross_ece_loss'],
                            # "val/cross_ece_loss": losses['val_cross_ece_loss'],
                            # "train/sm_cross_ece_loss": losses['train_sm_cross_ece_loss'],
                            # "val/sm_cross_ece_loss": losses['val_sm_cross_ece_loss'],
                            "train/cross_ece_loss_multidim_new": losses['train_cross_ece_loss_multidim_new'],
                            "val/cross_ece_loss_multidim_new": losses['val_cross_ece_loss_multidim_new'],
                            "train/sm_cross_ece_loss_multidim_new": losses['train_sm_cross_ece_loss_multidim_new'],
                            "val/sm_cross_ece_loss_multidim_new": losses['val_sm_cross_ece_loss_multidim_new'],
                            "train/agreement": losses['train_agreement'],
                            "val/agreement": losses['val_agreement'],
                            "val/cross_ece_loss_val": losses['val_cross_ece_loss_val'],
                            "val/sm_cross_ece_loss_val": losses['val_sm_cross_ece_loss_val']
                        })
                        if self.config['post_hoc_calibrate']:
                            log_data.update({
                                "train/calibrator_sm_cross_ece_loss": losses['train_calibrator_sm_cross_ece_loss'],
                                "val/calibrator_sm_cross_ece_loss": losses['val_calibrator_sm_cross_ece_loss'],
                                "train/calibrator_cross_ece_loss": losses['train_calibrator_cross_ece_loss'],
                                "val/calibrator_cross_ece_loss": losses['val_calibrator_cross_ece_loss'],
                                "train/calibrator_ece_loss": losses['train_calibrator_ece_loss'],
                                "val/calibrator_ece_loss": losses['val_calibrator_ece_loss'],
                                "train/calibrator_entropy": losses['train_calibrator_entropy'],
                                "val/calibrator_entropy": losses['val_calibrator_entropy'],
                                "train/calibrator_agreement": losses['train_calibrator_agreement'],
                                "val/calibrator_agreement": losses['val_calibrator_agreement'],
                                "train/calibrator_zero_one_loss": losses['train_calibrator_zero_one_loss'],
                                "val/calibrator_zero_one_loss": losses['val_calibrator_zero_one_loss'],
                            })
                    if self.config['datasets'][self.id] == 'maze':
                        log_data.update({
                            "val/maze_success_rate": losses['val_maze_success_rate'],
                        })
                    # log pruned validation metrics if available
                    if 'val_pruned' in losses:
                        log_data.update({
                            "val_pruned/loss": losses['val_pruned'],
                            "val_pruned/zero_one_loss": losses['val_pruned_zero_one_loss'],
                        })
                        if 'val_pruned_maze_success_rate' in losses:
                            log_data.update({
                                "val_pruned/maze_success_rate": losses['val_pruned_maze_success_rate'],
                            })
                    wandb.log(log_data, step=global_step)

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
                    logits, calibrator_logits, CE_loss, loss, ece_loss_multidim_new, sm_ece_loss_multidim_new, cross_ece_loss_multidim_new, sm_cross_ece_loss_multidim_new, calibrator_sm_cross_ece_loss, calibrator_cross_ece_loss, calibrator_ece_loss, calibrator_entropy, calibrator_agreement, calibrator_zero_one_loss, brier_score, zero_one_loss, entropy, agreement, ece_loss_temperature_scaled = self.model(X, Y, collaborator_predictions, collaborator_probs)
                    loss = loss / self.config['gradient_accumulation_steps'] # scale the loss to account for gradient accumulation
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                X, Y, collaborator_probs = self.get_batch('train')
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

            # save on last iteration
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
            
            # label dataset for next round on last iteration
            if self.iter_num == self.config['max_iters']:
                next_agent_id = (self.id + 1) % config['num_agents']
                if self.round == 0:
                    input_path = os.path.join(self.data_dir, f'input{self.id}_round{self.round}.txt')
                else:
                    input_path = os.path.join(self.config['out_dir'], f'input{self.id}_round{self.round}.txt')
                label_path = os.path.join(self.data_dir, f'input{next_agent_id}_round{0}.txt') # label on original dataset
                output_path = os.path.join(self.config['out_dir'], f'input{next_agent_id}_round{self.round+1}.txt')
                print(colored(f"labeling dataset for round {self.round+1} agent {next_agent_id} =================================================", 'light_green'))
                start_time = time.time()
                self.label_dataset(input_path, label_path, output_path)
                end_time = time.time()
                print(colored(f"time taken: {end_time - start_time:.2f} seconds", 'light_green'))

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
    # gptconf = GPTConfig(**checkpoint['model_args'])
    # Start with current config values for all GPTConfig fields, then overlay
    # the checkpoint's saved model_args. This ensures that new fields added
    # after the checkpoint was created get their current config values instead
    # of falling back to GPTConfig defaults.
    saved_args = checkpoint['model_args']
    model_args = {}
    for field_name in GPTConfig.__dataclass_fields__:
        if field_name in saved_args:
            model_args[field_name] = saved_args[field_name]
        elif field_name in config:
            model_args[field_name] = config[field_name]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    state_dict = checkpoint['model']
    # Some checkpoints may carry this wrapper prefix.
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    incompatible = model.load_state_dict(state_dict, strict=False)
    missing_keys = list(incompatible.missing_keys)
    unexpected_keys = list(incompatible.unexpected_keys)
    # Backward compatibility: older checkpoints may not have calibrator weights.
    allowed_missing_prefixes = ("calibrate_mlp.",)
    disallowed_missing = [k for k in missing_keys if not k.startswith(allowed_missing_prefixes)]
    if disallowed_missing or unexpected_keys:
        raise RuntimeError(
            f"Error(s) in loading state_dict for GPT:\n"
            f"  Missing key(s): {disallowed_missing}\n"
            f"  Unexpected key(s): {unexpected_keys}"
        )
    if missing_keys:
        print(colored(f"ignoring missing checkpoint keys: {missing_keys}", "light_yellow"))
    model.to(config['device'])
    model.eval()
    print(colored(f"loaded checkpoint from {ckpt_path}", 'light_green'))
    return model

def label_starting_round_dataset(config, round, agent_id, model):
    """
    Label the starting round dataset for the next round and next agent, using given model.
    """
    next_agent_id = (agent_id + 1) % config['num_agents']
    input_path = os.path.join(config['out_dir'], f'input{agent_id}_round{round}.txt')
    label_path = os.path.join(config['out_dir'], f'input{next_agent_id}_round{0}.txt')
    output_path = os.path.join(config['out_dir'], f'input{next_agent_id}_round{round+1}.txt')

    # Build an Agent wrapper for round-specific shapes/tokenization and reuse the loaded GPT model.
    # For rounds > 0 the data includes collaborator annotations, so pass a non-None placeholder.
    label_config = dict(config)
    label_config['compile'] = False
    collaborator_placeholder = model if round > 0 else None
    labeler = Agent(label_config, round, agent_id, collaborator_placeholder)
    labeler.model = model
    labeler.model.eval()

    print(colored(f"labeling starting round dataset for round {round+1} agent {next_agent_id} =================================================", 'light_green'))
    start_time = time.time()
    labeler.label_dataset(input_path, label_path, output_path)
    end_time = time.time()    
    print(colored(f"time taken: {end_time - start_time:.2f} seconds", 'light_green'))


def train_converse(config):
    """Iteratively train agents using collaborator models."""
    # training loop: agents take turns training in rounds
    current_model = None # collaborator model to be used by the next agent. in the first round, the first agent trains by itself.
    
    # Check if master process (for DDP compatibility)
    ddp = int(os.environ.get('RANK', -1)) != -1
    master_process = (not ddp) or (int(os.environ['RANK']) == 0)
    
    # Initialize wandb early so logging works during label_starting_round
    if config['wandb_log'] and master_process:
        wandb.init(entity="mirahshi-university-of-pennsylvania",project=config['wandb_project'], group=config['wandb_group_name'], name=config['wandb_run_name'], config=config)
    
    for r in range(config['num_rounds']):
        
        agent_id = r % config['num_agents'] # determine which agent trains this round
        
        if r < config['start_from_round']-1:
            print(colored(f"Round {r}: agent {agent_id} skips ==================================================", 'light_yellow'))
            continue
        elif r == config['start_from_round']-1:
            current_model = load_model(r, agent_id)
            print(colored(f"Round {r}: loaded agent {agent_id} model ==========================================", 'light_yellow'))
            if config['label_starting_round']:
                label_starting_round_dataset(config, r, agent_id, current_model)
        else:
            print(colored(f"Round {r}: agent {agent_id} trains =================================================", 'light_yellow'))
            
            agent = Agent(config, r, agent_id, current_model, seed=int(config['seed']*10)+r)
            agent.train()
            current_model = agent.model

    return current_model

if __name__ == "__main__":
    final_model = train_converse(config)
