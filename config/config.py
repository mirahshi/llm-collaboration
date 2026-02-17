
num_agents = 2
num_rounds = 2
pre_load_rounds = [0] # rounds to pre-load collaborator models from
save_models = False # save models after each round

datasets = ['majority-mask'] * num_agents

wandb_log = True # override via command line if you like
wandb_project = 'parity'
wandb_group_name = 'collab_exp4'

out_dir = f'out-{wandb_group_name}'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# calibrate?
calibrate = None # self-calibrate: None, 'smECE', 'brier'
multiplier = 1 # multiplier for calibration loss
cross_calibrate = False # cross-calibrate: smECE conditioned on collaborator's predictions
cross_multiplier = 1 # multiplier for cross calibration loss
confidence = False # use confidence calibration; otherwise use prediction calibration

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_run_name = ''
if calibrate is not None:
    if not confidence:
        wandb_run_name = wandb_run_name + f'pred'
    wandb_run_name = wandb_run_name + f'cal-{calibrate}x{multiplier}'
if cross_calibrate: 
    wandb_run_name = wandb_run_name + f'crossx{cross_multiplier}'

# get prefix size from input.txt
dataset = datasets[0] # ASSUMING BOTH AGENTS' TASKS ARE IN THE SAME FORMAT
with open(f'data/{dataset}/input0.txt', 'r') as f:
    lines = f.readlines()
    example_size = len(lines[0].split('\n')[0]) + 1 # number of characters in the input example (including '\n')
    prefix_size = len(lines[0].split('=')[0]) # number of characters in the input before the '='
    print("prefix_size:", prefix_size)
# check that the prefix size is the same for all agents
for idx in range(num_agents):
    with open(f'data/{dataset}/input{idx}.txt', 'r') as f:
        lines = f.readlines()
        if len(lines[0].split('=')[0]) != prefix_size:
            raise ValueError(f"Prefix size for {dataset} is not the same as the first dataset")

gradient_accumulation_steps = 1
batch_size = 256
block_size = prefix_size+1 # 32 # 256 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 2
n_head = 6
n_embd = 240 # 384
dropout = 0.0 # 0.2
causal = True

learning_rate = 3e-5 # 1e-4 # 1e-3 # with baby networks can afford to go a bit higher
max_iters = 10000
lr_decay_iters = 40000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# wandb_run_name = wandb_run_name + f'-n_embd{n_embd}-dropout{dropout}-lr{learning_rate}-layers{n_layer}'
# wandb_run_name = wandb_run_name + f'-lr{learning_rate}'

# on macbook also add
# device = 'cpu'  # run on cpu only
compile = False # do not torch compile the model


