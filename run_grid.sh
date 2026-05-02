#!/usr/bin/env bash
set -euo pipefail

wandb_group_name="collab_exp22"
input_file0="out-collab_exp17/m1/input0_round0.txt"
input_file1="out-collab_exp17/m1/input1_round0.txt"

prune_dataset=True

# curriculum (append probabilities)
wandb_run_name="input-probs-curriculum-v2"
out_dir="out-${wandb_group_name}/${wandb_run_name}"
use_curriculum=True
append_probabilities=True
append_predictions=False
append_argmax_predictions=False
start_from_round=1
label_starting_round=True
mkdir -p ${out_dir}
cp -n out-collab_exp21/input-probs/ckpt_round0_agent0.pt ${out_dir}
python prepare.py --input_file=${input_file0} --out_dir=${out_dir} --suffix="0_round0"
python prepare.py --input_file=${input_file1} --out_dir=${out_dir} --suffix="1_round0"
python train_converse.py config/config_maze.py --wandb_group_name=${wandb_group_name} --wandb_run_name=${wandb_run_name} --out_dir=${out_dir} --append_probabilities=${append_probabilities} --append_predictions=${append_predictions} --append_argmax_predictions=${append_argmax_predictions} --start_from_round=${start_from_round} --label_starting_round=${label_starting_round} --prune_dataset=${prune_dataset} --use_curriculum=${use_curriculum}

# no curriculum (append probabilities)
wandb_run_name="input-probs-no-curriculum-v2"
out_dir="out-${wandb_group_name}/${wandb_run_name}"
use_curriculum=False
append_probabilities=True
append_predictions=False
append_argmax_predictions=False
start_from_round=1
label_starting_round=True
mkdir -p ${out_dir}
cp -n out-collab_exp21/input-probs/ckpt_round0_agent0.pt ${out_dir}
python prepare.py --input_file=${input_file0} --out_dir=${out_dir} --suffix="0_round0"
python prepare.py --input_file=${input_file1} --out_dir=${out_dir} --suffix="1_round0"
python train_converse.py config/config_maze.py --wandb_group_name=${wandb_group_name} --wandb_run_name=${wandb_run_name} --out_dir=${out_dir} --append_probabilities=${append_probabilities} --append_predictions=${append_predictions} --append_argmax_predictions=${append_argmax_predictions} --start_from_round=${start_from_round} --label_starting_round=${label_starting_round} --prune_dataset=${prune_dataset} --use_curriculum=${use_curriculum}

# # append argmax action
# wandb_run_name="input-argmax-actions"
# out_dir="out-${wandb_group_name}/${wandb_run_name}"
# append_probabilities=False
# append_predictions=True
# append_argmax_predictions=True
# start_from_round=1
# label_starting_round=True
# # mkdir -p ${out_dir}
# # cp -n out-collab_exp21/input-probs/ckpt_round0_agent0.pt ${out_dir}
# # python prepare.py --input_file=${input_file0} --out_dir=${out_dir} --suffix="0_round0"
# # python prepare.py --input_file=${input_file1} --out_dir=${out_dir} --suffix="1_round0"
# python train_converse.py config/config_maze.py --wandb_group_name=${wandb_group_name} --wandb_run_name=${wandb_run_name} --out_dir=${out_dir} --append_probabilities=${append_probabilities} --append_predictions=${append_predictions} --append_argmax_predictions=${append_argmax_predictions} --start_from_round=${start_from_round} --label_starting_round=${label_starting_round} --prune_dataset=${prune_dataset}

# # vary temperature for appended probabilities
# for append_probabilities_temperature in 0.2; do 
#     wandb_run_name="temp${append_probabilities_temperature}"
#     out_dir="out-${wandb_group_name}/${wandb_run_name}"
#     start_from_round=1
#     label_starting_round=True
#     mkdir -p ${out_dir}
#     cp -n out-collab_exp18/probs-only/ckpt_round0_agent0.pt ${out_dir}
#     python prepare.py --input_file=${input_file0} --out_dir=${out_dir} --suffix="0_round0"
#     python prepare.py --input_file=${input_file1} --out_dir=${out_dir} --suffix="1_round0"
#     python train_converse.py config/config_maze.py --wandb_group_name=${wandb_group_name} --wandb_run_name=${wandb_run_name} --out_dir=${out_dir} --append_probabilities_temperature=${append_probabilities_temperature} --start_from_round=${start_from_round} --label_starting_round=${label_starting_round}
# done

# for m_lookahead in 3 5; do
#     if (( m_lookahead > 1 )); then
#         for autoregressive_lookahead in False; do
#             wandb_group_name="collab_exp17"
#             wandb_run_name="m${m_lookahead}autoregressive${autoregressive_lookahead}"
#             out_dir="out-${wandb_group_name}/${wandb_run_name}"
#             python data/maze/generate.py --out_dir=${out_dir} --m_lookahead=${m_lookahead}
#             python prepare.py --input_file=${out_dir}/input0_round0.txt --out_dir=${out_dir} --suffix="0_round0"
#             python prepare.py --input_file=${out_dir}/input1_round0.txt --out_dir=${out_dir} --suffix="1_round0"
#             python train_converse.py config/config_maze.py --wandb_group_name=${wandb_group_name} --wandb_run_name=${wandb_run_name} --out_dir=${out_dir} --m_lookahead=${m_lookahead} --autoregressive_lookahead=${autoregressive_lookahead}
#         done
#     else
#         wandb_group_name="collab_exp17"
#         wandb_run_name="m${m_lookahead}"
#         out_dir="out-${wandb_group_name}/${wandb_run_name}"
#         python data/maze/generate.py --out_dir=${out_dir} --m_lookahead=${m_lookahead}
#         python prepare.py --input_file=${out_dir}/input0_round0.txt --out_dir=${out_dir} --suffix="0_round0"
#         python prepare.py --input_file=${out_dir}/input1_round0.txt --out_dir=${out_dir} --suffix="1_round0"
#         python train_converse.py config/config_maze.py --wandb_group_name=${wandb_group_name} --wandb_run_name=${wandb_run_name} --out_dir=${out_dir} --m_lookahead=${m_lookahead}
#     fi
# done