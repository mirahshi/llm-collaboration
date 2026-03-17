num_agents = 2
num_rounds = 4
start_from_round = 0 # which round to start from (begins at 0) 
save_models = True # save models after each round
out_dir_suffix = 'scratch'

datasets = ['maze'] * num_agents

wandb_log = True # override via command line if you like
wandb_project = 'parity'
wandb_group_name = 'collab_exp14'

out_dir = f'out-{wandb_group_name}/{out_dir_suffix}'

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
K = 10 # number of buckets for self ECE
K_cross = 10 # number of buckets for cross ECE
answer_tokens = ['d', 'r', 'u', 'l'] # possible answer tokens
append_predictions = True # append predictions to output file
append_probabilities = True # append probabilities to output file

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_run_name = ''
if confidence:
    wandb_run_name = wandb_run_name + 'conf'
if calibrate is not None:
    wandb_run_name = wandb_run_name + f'cal-{calibrate}K{K}K_cross{K_cross}x{multiplier}'
else:
    wandb_run_name = wandb_run_name + f'nocal-K{K}K_cross{K_cross}'
if cross_calibrate: 
    wandb_run_name = wandb_run_name + f'crossK{K_cross}x{cross_multiplier}'
if append_probabilities:
    wandb_run_name += '-with-probs'

# get prefix size and answer length from input.txt
dataset = datasets[0] # ASSUMING BOTH AGENTS' TASKS ARE IN THE SAME FORMAT
with open(f'data/{dataset}/input0_round0.txt', 'r') as f:
    lines = f.readlines()
    example_size = len(lines[0].split('\n')[0]) + 1 # number of characters in the input example (including '\n')
    prefix_size = len(lines[0].split('=')[0]) # number of characters in the input before the '='
    print("config example size:", example_size)
    print("config prefix_size:", prefix_size)
# check that the prefix size is the same for all agents
for idx in range(num_agents):
    with open(f'data/{dataset}/input{idx}_round0.txt', 'r') as f:
        lines = f.readlines()
        if len(lines[0].split('=')[0]) != prefix_size:
            raise ValueError(f"Prefix size for {dataset} is not the same as the first dataset")

gradient_accumulation_steps = 1
batch_size = 1024
block_size = prefix_size+1 # 32 # 256 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 2 # TODO: TRY MORE LAYERS
n_head = 6
n_embd = 240 # 384
dropout = 0.0 # 0.2
causal = True

learning_rate = 1e-4 # 1e-3 # with baby networks can afford to go a bit higher
max_iters = 15000
lr_decay_iters = max_iters # make equal to max_iters usually
min_lr = learning_rate / 10 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# wandb_run_name = wandb_run_name + f'-n_embd{n_embd}-dropout{dropout}-lr{learning_rate}-layers{n_layer}'
# wandb_run_name = wandb_run_name + f'-lr{learning_rate}'

# on macbook also add
# device = 'cpu'  # run on cpu only
compile = False # do not torch compile the model


