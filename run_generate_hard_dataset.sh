#!/usr/bin/env bash
#SBATCH --job-name=collab_generate_hard_dataset
#SBATCH --partition=dgx-b200
#SBATCH --output=logs/generate_hard_dataset_%j.out
#SBATCH --error=logs/generate_hard_dataset_%j.err
#SBATCH --time=0:40:00
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /vast/home/m/mirahshi/envs/llm-collaboration

wandb_group_name="collab_exp25"
input_file0="out-collab_exp17/input0_round0.txt"
input_file1="out-collab_exp17/input1_round0.txt"

wandb_run_name="test_curriculum"
out_dir="/vast/projects/surbhig/multi-agent-collab/out-${wandb_group_name}/${wandb_run_name}"

generate_hard_dataset=True
append_probabilities=True
append_predictions=False
max_iters=500
num_agents=2
num_rounds=1
wandb_log=False

curriculum_steps=4
for step in $(seq 1 "${curriculum_steps}"); do
    current_out_dir="${out_dir}/step${step}"
    mkdir -p "${current_out_dir}"
    if [ "${step}" -eq 1 ]; then
        python prepare.py --input_file=${input_file0} --out_dir=${current_out_dir} --suffix="0_round0" --split="both"
        python prepare.py --input_file=${input_file1} --out_dir=${current_out_dir} --suffix="1_round0" --split="both"
    fi
    if [ "${step}" -gt 1 ]; then
        # copy meta files from step 1
        cp "${out_dir}/step1/meta0_round0.pkl" "${current_out_dir}/meta0_round0.pkl"
        cp "${out_dir}/step1/meta1_round0.pkl" "${current_out_dir}/meta1_round0.pkl"
    fi

    curriculum_next_out_dir="${out_dir}/step$((step + 1))"
    mkdir -p "${curriculum_next_out_dir}"

    python train_converse.py config/config_maze.py \
        --wandb_group_name="${wandb_group_name}" \
        --wandb_run_name="${wandb_run_name}" \
        --out_dir="${current_out_dir}" \
        --curriculum_next_out_dir="${curriculum_next_out_dir}" \
        --generate_hard_dataset="${generate_hard_dataset}" \
        --append_probabilities="${append_probabilities}" \
        --append_predictions="${append_predictions}" \
        --max_iters="${max_iters}" \
        --wandb_log="${wandb_log}" \
        --num_agents="${num_agents}" \
        --num_rounds="${num_rounds}"
done