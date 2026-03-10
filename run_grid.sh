#!/usr/bin/env bash
set -euo pipefail
# hyperparameter sweep

for batch_size in 512,1024; do
    for n_layer in 2; do
        for dropout in 0.0; do
            for learning_rate in 3e-4,1e-4,7e-5,3e-5,1e-5; do
                for n_embd in 240; do
                    for causal in True; do
                        for max_iters in 25000; do
                        wandb_run_name="batch_size${batch_size}-lr${learning_rate}"
                        python train.py config/config_maze.py --wandb_run_name=$wandb_run_name --batch_size=$batch_size --n_layer=$n_layer --dropout=$dropout --learning_rate=$learning_rate --min_lr=$learning_rate --n_embd=$n_embd --causal=$causal --max_iters=$max_iters
                    done
                done
            done
        done
    done
done