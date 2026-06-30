#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH -A uTS26_Tornator
#SBATCH -t 04:00:00
##SBATCH --exclusive
#SBATCH --job-name=circuit_full
#SBATCH -o circuit_full.out

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# OLMoE-1B-7B: 16 layers, 64 experts (top-8). The model is
# small enough that the AtP backward pass fits at the full batch size, so atp-batch-size
# matches batch-size here (gpt-oss needs a smaller one). The gate-AtP grid cache is keyed by
# n-prompts, so changing --n-prompts just writes a new atp_grid_n<N>.npy (no cleanup needed).
source scripts/setup_env.sh
python scripts/cineca/circuit_runner.py \
  --model allenai/OLMoE-1B-7B-0924-Instruct \
  --batch-size 6 \
  --atp-batch-size 6 \
  --n-prompts 48 \
  --n-test 48
