#!/usr/bin/env bash
set -euo pipefail

for m_lookahead in 1 3 5; do
    if (( m_lookahead > 1 )); then
        for autoregressive_lookahead in True False; do
            wandb_group_name="collab_exp17"
            wandb_run_name="m${m_lookahead}autoregressive${autoregressive_lookahead}"
            out_dir="out-${wandb_group_name}/${wandb_run_name}"
            python data/maze/generate.py --out_dir=${out_dir} --m_lookahead=${m_lookahead}
            python prepare.py --input_file=${out_dir}/input0_round0.txt --out_dir=${out_dir} --suffix="0_round0"
            python prepare.py --input_file=${out_dir}/input1_round0.txt --out_dir=${out_dir} --suffix="1_round0"
            python train_converse.py config/config_maze.py --wandb_group_name=${wandb_group_name} --wandb_run_name=${wandb_run_name} --out_dir=${out_dir} --m_lookahead=${m_lookahead} --autoregressive_lookahead=${autoregressive_lookahead}
        done
    else
        wandb_group_name="collab_exp17"
        wandb_run_name="m${m_lookahead}"
        out_dir="out-${wandb_group_name}/${wandb_run_name}"
        python data/maze/generate.py --out_dir=${out_dir} --m_lookahead=${m_lookahead}
        python prepare.py --input_file=${out_dir}/input0_round0.txt --out_dir=${out_dir} --suffix="0_round0"
        python prepare.py --input_file=${out_dir}/input1_round0.txt --out_dir=${out_dir} --suffix="1_round0"
        python train_converse.py config/config_maze.py --wandb_group_name=${wandb_group_name} --wandb_run_name=${wandb_run_name} --out_dir=${out_dir} --m_lookahead=${m_lookahead}
    fi
done