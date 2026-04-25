num_agents = 2
num_rounds = 4
start_from_round = 0 # which round to start from (begins at 0) 
label_starting_round = False # label the starting round dataset using the model of the previous round
save_models = True # save models after each round
out_dir_suffix = 'scratch_lookahead_m1' #'n_layer4'

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
compute_smooth_calibration = False # compute smooth calibration losses (memory intensive)

post_hoc_calibrate = False # post-hoc cross calibrate the model using the predictions of the previous round
post_hoc_calibrate_multiplier = 1.0 # multiplier for post-hoc cross calibration loss

answer_tokens = ['d', 'r', 'u', 'l'] # possible answer tokens
append_predictions = False # append predictions to output file (default is sampled predictions unless append_argmax_predictions is True)
append_argmax_predictions = True # if append_predictions is True, append the argmax prediction instead of the sampled prediction
append_probabilities = False # append probabilities to output file
append_probabilities_precision = 4 # decimal places for appended probabilities
append_probabilities_temperature = 1.0 # temperature for appended probabilities (default is 1)

m_lookahead = 1 # number of autoregressive lookahead predictions to generate in one forward pass
autoregressive_lookahead = True # use autoregressive lookahead (otherwise, use ground truth targets)

prune_dataset = False # prune the dataset after each round to remove examples that are correct with 100% confidence
use_curriculum = False # train using curriculum (requires prune_dataset=True)

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_run_name = ''
if calibrate is not None:
    wandb_run_name = wandb_run_name + f'cal-{calibrate}K{K}K_cross{K_cross}x{multiplier}'
else:
    wandb_run_name = wandb_run_name + f'nocal-K{K}K_cross{K_cross}'
if cross_calibrate: 
    wandb_run_name = wandb_run_name + f'crossK{K_cross}x{cross_multiplier}'
if m_lookahead > 1:
    wandb_run_name += f'-m{m_lookahead}'
    if autoregressive_lookahead:
        wandb_run_name += 'autoregressive'

# import sys
# def _get_cli_arg(name, default):
#     prefix = f"--{name}="
#     for arg in sys.argv[1:]:
#         if arg.startswith(prefix):
#             return arg.split("=", 1)[1]
#     return default
# # override from command line if provided (e.g. --out_dir=out-collab_exp16/n_layer4)
# out_dir = _get_cli_arg("out_dir", out_dir)
# print(f"config out_dir: {out_dir}")

# # get prefix size and answer length from input files
# with open(f'{data_dir}/input0_round0.txt', 'r') as f:
#     lines = f.readlines()
#     example_size = len(lines[0].split('\n')[0]) + 1 # number of characters in the input example (including '\n')
#     block_size = example_size - 1 - m_lookahead
#     prefix_size = block_size - 1
#     # prefix_size = len(lines[0].split('=')[0]) # number of characters in the input before the '='
#     # target_size = len(lines[0].split('=')[1]) - 1 # number of characters in the target (excluding '\n')
#     #assert target_size == m_lookahead, f"Target size {target_size} does not match m_lookahead {m_lookahead}"
#     print(f"config example size: {example_size}")
#     print(f"config block size: {block_size}")
#     print(f"config prefix size: {prefix_size}")
# # check that the block size is the same for all agents
# for idx in range(num_agents):
#     with open(f'{data_dir}/input{idx}_round0.txt', 'r') as f:
#         lines = f.readlines()
#         if len(lines[0].split('\n')[0]) - m_lookahead != block_size:
#             raise ValueError(f"Block size is not the same as the first dataset")

gradient_accumulation_steps = 1
batch_size = 1024
# block_size = prefix_size+1 # 32 # 256 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 4
n_head = 6
n_embd = 240 # 384
dropout = 0.0 # 0.2
causal = True

learning_rate = 1e-4 # 1e-3 # with baby networks can afford to go a bit higher
max_iters = 25000
lr_decay_iters = max_iters # make equal to max_iters usually
min_lr = learning_rate / 10 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small
warmup_iters = 100 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
compile = False # do not torch compile the model


