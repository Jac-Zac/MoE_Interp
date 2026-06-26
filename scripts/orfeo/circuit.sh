#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --partition=GPU
#SBATCH --gres=gpu:V100:1
#SBATCH -t 06:00:00
#SBATCH --job-name=circuit_full
#SBATCH -o circuit_full.out

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Full causal-circuit pipeline at held-out scale: patching grid, gate-AtP, faithfulness,
# the knockout/project-out intervention, and the localized project-out (step 5). The larger
# --n-test is what makes the localized *specificity* check (Δelic vs Δneut) trustworthy — at
# small n the neutral collateral is too noisy to tell blunt suppression from real localization.
#
# NOTE: the patching grid (patching_grid.npy) is cached and NOT keyed by --n-prompts. If you
# change --n-prompts, delete data/<model>/circuit/patching/ and circuit/attribution/ first, or
# the cheap steps will mix a stale grid with a freshly-sized AtP/steer split.
source scripts/setup_env.sh
module load cuda

python scripts/cineca/circuit_runner.py \
  --model allenai/OLMoE-1B-7B-0924-Instruct \
  --batch-size 6 \
  --atp-batch-size 6 \
  --knockout-k 15 \
  --steer-layer 12 \
  --max-new-tokens 24 \
  --n-prompts 100 \
  --n-test 64
