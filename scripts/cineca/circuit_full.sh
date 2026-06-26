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

# OLMoE-1B-7B: 16 layers, 64 experts (top-8). steer-layer 12 is ~75% depth. The model is
# small enough that the AtP backward pass fits at the full batch size, so atp-batch-size
# matches batch-size here (gpt-oss needs a smaller one). n-test sizes the held-out split —
# kept generous because the localized step's specificity check (Δelic vs Δneut) needs enough
# neutral prompts to be trustworthy. Changing --n-prompts? clear circuit/patching + attribution
# first (the patching grid cache is not keyed by n-prompts).
source scripts/setup_env.sh
python scripts/cineca/circuit_runner.py \
  --model allenai/OLMoE-1B-7B-0924-Instruct \
  --batch-size 6 \
  --atp-batch-size 6 \
  --knockout-k 15 \
  --steer-layer 12 \
  --max-new-tokens 24 \
  --n-prompts 48 \
  --n-test 48
