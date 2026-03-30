#!/usr/bin/env bash
set -euo pipefail

wandb_group_name="collab_exp20"
input_file0="out-collab_exp17/m1/input0_round0.txt"
input_file1="out-collab_exp17/m1/input1_round0.txt"

# post hoc cross calibrate
out_dir_suffix="post_hoc_cc"
out_dir="out-${wandb_group_name}/${out_dir_suffix}"
append_predictions=False
append_probabilities=False
K=3
K_cross=3
post_hoc_calibrate=True
post_hoc_calibrate_multiplier=1.0
wandb_run_name="post_hoc_cc_K${K}_Kcross${K_cross}"
mkdir -p ${out_dir}
python prepare.py --input_file=${input_file0} --out_dir=${out_dir} --suffix="0_round0"
python prepare.py --input_file=${input_file1} --out_dir=${out_dir} --suffix="1_round0"
python train_converse.py config/config_maze.py --wandb_group_name=${wandb_group_name} --wandb_run_name=${wandb_run_name} --out_dir=${out_dir} --post_hoc_calibrate=${post_hoc_calibrate} --post_hoc_calibrate_multiplier=${post_hoc_calibrate_multiplier} --append_predictions=${append_predictions} --append_probabilities=${append_probabilities} --K=${K} --K_cross=${K_cross}