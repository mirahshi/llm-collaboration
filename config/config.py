
num_agents = 2
num_rounds = 3
start_from_round = 0 # which round to start from (begins at 0) 
save_models = False # save models after each round
save_name_suffix = ''
# save_name_suffix = '_crosscal' # include suffix for saved model names

datasets = ['majority-mask'] * num_agents

wandb_log = True # override via command line if you like
wandb_project = 'parity'
wandb_group_name = 'collab_exp5'

out_dir = f'out-{wandb_group_name}'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# calibrate?
calibrate = None # self-calibrate: None, 'smECE', 'brier'
multiplier = 1 # multiplier for calibration loss
cross_calibrate = False # cross-calibrate: smECE conditioned on collaborator's predictions
cross_multiplier = 1 # multiplier for cross calibration loss
confidence = False # use confidence calibration; otherwise use probability calibration
cross_probabilities = True # use collaborator's probabilities for cross calibration
K = 5 # number of buckets for (cross) ECE
answer_tokens = ['0', '1'] # possible answer tokens
top_k = 2 # number of top k predictions to use for ECE losses

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_run_name = ''
if calibrate is not None:
    wandb_run_name = wandb_run_name + f'cal-{calibrate}K{K}x{multiplier}'
if cross_calibrate: 
    wandb_run_name = wandb_run_name + f'crossK{K}x{cross_multiplier}'

# get prefix size and answer length from input.txt
dataset = datasets[0] # ASSUMING BOTH AGENTS' TASKS ARE IN THE SAME FORMAT
with open(f'data/{dataset}/input0_round0.txt', 'r') as f:
    lines = f.readlines()
    example_size = len(lines[0].split('\n')[0]) + 1 # number of characters in the input example (including '\n')
    prefix_size = len(lines[0].split('=')[0]) # number of characters in the input before the '='
    print("example size:", example_size)
    print("prefix_size:", prefix_size)
# check that the prefix size is the same for all agents
for idx in range(num_agents):
    with open(f'data/{dataset}/input{idx}_round0.txt', 'r') as f:
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
max_iters = 200# 10000
lr_decay_iters = 40000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# wandb_run_name = wandb_run_name + f'-n_embd{n_embd}-dropout{dropout}-lr{learning_rate}-layers{n_layer}'
# wandb_run_name = wandb_run_name + f'-lr{learning_rate}'

# on macbook also add
# device = 'cpu'  # run on cpu only
compile = False # do not torch compile the model


