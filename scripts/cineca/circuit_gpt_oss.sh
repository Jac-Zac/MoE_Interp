#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH -A uTS26_Tornator
#SBATCH -t 06:00:00
##SBATCH --exclusive
#SBATCH --job-name=circuit_gpt_oss
#SBATCH -o circuit_gpt_oss.out

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# gpt-oss-20b: 24 layers, 32 experts (top-4). atp-batch-size is smaller than batch-size to
# avoid OOM on the gate-AtP backward pass.
source scripts/setup_env.sh
python scripts/cineca/circuit_runner.py \
  --model openai/gpt-oss-20b \
  --batch-size 4 \
  --atp-batch-size 2 \
  --n-prompts 50 \
  --n-test 30
