#!/usr/bin/env bash
set -euo pipefail

wandb_group_name="collab_exp20"

# post hoc cross calibrate
out_dir_suffix="post_hoc_cc"
out_dir="out-${wandb_group_name}/${out_dir_suffix}"
append_predictions=False
append_probabilities=False
K=5
K_cross=5
post_hoc_calibrate=True
post_hoc_calibrate_multiplier=1.0
wandb_run_name="post_hoc_cc_K${K}_Kcross${K_cross}"
python prepare.py --out_dir=${out_dir}
python train_converse.py config/config_maze.py --wandb_group_name=${wandb_group_name} --wandb_run_name=${wandb_run_name} --out_dir=${out_dir} --post_hoc_calibrate=${post_hoc_calibrate} --append_predictions=${append_predictions} --append_probabilities=${append_probabilities}