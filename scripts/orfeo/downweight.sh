#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --partition=GPU
#SBATCH --gres=gpu:V100:1
#SBATCH -t 02:00:00
#SBATCH --job-name=downweight
#SBATCH -o downweight.out

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Knockout / downweighting sweep (no steering). For two expert budgets (1% and 5% of all
# (layer,expert) slots; top-ranked per selector) it scales the router gate of the SOMP / gate-AtP /
# matched-random expert sets by 0.9, 0.5, 0.25, 0.0 (10%/50%/75% downweight + full knockout) during
# 24-token greedy generation, scoring concept propensity / word-rate / distinct-1 PER PROMPT so the
# bootstrap puts 95% CIs (error bars) on every point. Results -> $DATA_DIR/<model>/circuit/downweight/
# sweep_offensive.json (DATA_DIR from .env).
#
# Generation-heavy; the per-(budget,selector,scale,set) checkpoint makes it resume-safe if the 2h
# GPU cap is hit — just resubmit this same sbatch and it skips completed cells. Add finer
# downweight steps on a resubmit with e.g.  --scales 0.9 0.75 0.5 0.25 0.1 0.0
source scripts/setup_env.sh
module load cuda

# Extra args pass through (e.g. --budgets 0.01 0.05 0.1, --hi 0.8 --challenging, --n-test 96).
python scripts/cineca/downweight_runner.py \
  --model allenai/OLMoE-1B-7B-0924-Instruct \
  --batch-size 6 \
  --max-new-tokens 24 \
  --n-prompts 100 \
  --n-test 64 \
  "$@"
