#!/usr/bin/env bash
set -euo pipefail

wandb_group_name="out-api_exp4"
wandb_run_name="probs-rollouts"

# for calibrator.py
calibrator_out_dir="${wandb_group_name}/calibrator"
maze_conversation_log_path="${wandb_group_name}/${wandb_run_name}/conversations"

# for api_converse_multiprompt_calibrator.py
calibrator_path="${wandb_group_name}/calibrator"
data_dir="out-api_exp1"
converse_out_dir="${wandb_group_name}/${wandb_run_name}"
num_rounds=4
num_samples=2
verbose=True
start_maze=0
end_maze=1

# begin by training calibrator for round 1
round=1
python calibrator.py \
    --maze_conversation_log_path=${maze_conversation_log_path} \
    --out_dir=${calibrator_out_dir} \
    --round=${round}

# then run api converse for round 2 using calibrated probabilities
start_round=2
end_round=3
python api_converse_multiprompt_calibrator.py \
    --calibrator_path=${calibrator_path} \
    --data_dir=${data_dir} \
    --out_dir=${converse_out_dir} \
    --start_round=${start_round} \
    --end_round=${end_round} \
    --num_samples=${num_samples} \
    --verbose=${verbose} \
    --start_maze=${start_maze} \
    --end_maze=${end_maze}

# then train calibrator for round 2
round=2
python calibrator.py \
    --maze_conversation_log_path=${maze_conversation_log_path} \
    --out_dir=${calibrator_out_dir} \
    --round=${round}

# then run api converse for round 3 using calibrated probabilities
start_round=3
end_round=4
python api_converse_multiprompt_calibrator.py \
    --calibrator_path=${calibrator_path} \
    --data_dir=${data_dir} \
    --out_dir=${converse_out_dir} \
    --start_round=${start_round} \
    --end_round=${end_round} \
    --num_samples=${num_samples} \
    --verbose=${verbose} \
    --start_maze=${start_maze} \
    --end_maze=${end_maze}

# then train calibrator for round 3
round=3
python calibrator.py \
    --maze_conversation_log_path=${maze_conversation_log_path} \
    --out_dir=${calibrator_out_dir} \
    --round=${round}