#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH -A uTS25_Tornator
#SBATCH -t 04:00:00
##SBATCH --exclusive
#SBATCH --job-name=circuit_full
#SBATCH -o circuit_full.out

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

source scripts/setup_env.sh
python scripts/cineca/circuit_runner.py \
  --model allenai/OLMoE-1B-7B-0924-Instruct \
  --batch-size 6 \
  --knockout-k 15 \
  --steer-layer 12 \
  --max-new-tokens 24 \
  --downweight-scale 0.5
