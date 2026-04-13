#!/usr/bin/env bash
set -euo pipefail

wandb_group_name="collab_exp22"
input_file0="out-collab_exp22/8x8-input-probs-curriculum-v2/input0_round0.txt"
input_file1="out-collab_exp22/8x8-input-probs-curriculum-v2/input1_round0.txt"

# generate hard dataset
out_dir_suffix="8x8-input-probs-curriculum-v2"
out_dir="out-${wandb_group_name}/${out_dir_suffix}"
append_predictions=False
append_probabilities=False
start_from_round=4
label_starting_round=True
prune_dataset=True
generate_hard_dataset=True
wandb_run_name="generate_hard_dataset"
mkdir -p ${out_dir}

# python prepare.py --input_file=${input_file0} --out_dir=${out_dir} --suffix="0_round0"
# python prepare.py --input_file=${input_file1} --out_dir=${out_dir} --suffix="1_round0"
python train_converse.py config/config_maze.py --wandb_group_name=${wandb_group_name} --wandb_run_name=${wandb_run_name} --out_dir=${out_dir} --start_from_round=${start_from_round} --label_starting_round=${label_starting_round} --prune_dataset=${prune_dataset} --append_predictions=${append_predictions} --append_probabilities=${append_probabilities} --generate_hard_dataset=${generate_hard_dataset}