#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH -A uTS25_Tornator
#SBATCH -t 06:00:00
##SBATCH --exclusive
#SBATCH --job-name=circuit_gpt_oss
#SBATCH -o circuit_gpt_oss.out

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# gpt-oss-20b: 24 layers, 32 experts (top-4). The patching grid is more forwards than
# OLMoE, hence the longer walltime; steer-layer 12 is mid-stack.
source scripts/setup_env.sh
python scripts/cineca/circuit_runner.py \
  --model openai/gpt-oss-20b \
  --batch-size 4 \
  --knockout-k 15 \
  --steer-layer 12 \
  --max-new-tokens 24 \
  --downweight-scale 0.5
