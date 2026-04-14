#!/usr/bin/env bash
set -euo pipefail

wandb_group_name="out-api_exp3"
wandb_run_name="probs-rollouts-calibrator"

num_rounds=4
num_agents=2
num_samples=2
append_probabilities=True
verbalize_probabilities=True
append_full_response=False
calibrator_path="out-api_exp3/calibrator"
data_dir="out-api_exp1"
out_dir="${wandb_group_name}/${wandb_run_name}"
verbose=False
start_maze=0
end_maze=20

python api_converse_multiprompt_probs.py \
    --num_rounds=${num_rounds} \
    --num_agents=${num_agents} \
    --num_samples=${num_samples} \
    --calibrator_path=${calibrator_path} \
    --data_dir=${data_dir} \
    --out_dir=${out_dir} \
    --verbose=${verbose} \
    --start_maze=${start_maze} \
    --end_maze=${end_maze}