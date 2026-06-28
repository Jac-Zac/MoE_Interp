#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --partition=GPU
#SBATCH --gres=gpu:V100:1
#SBATCH -t 02:00:00
#SBATCH --job-name=circuit_full
#SBATCH -o circuit_full.out

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Full causal-circuit pipeline at held-out scale: gate-AtP localization, the per-expert
# interventions on the SOMP and AtP experts vs a matched-random control (knockout + α
# expert-output DoM steering), and the dose-response curve. The larger --n-test is what makes
# the *specificity* check (Δelic vs Δneut, causal vs random) trustworthy — at small n the
# neutral collateral is too noisy. The GPU association caps walltime at 2h; the intervention
# step is generation-heavy, so this is the full budget.
#
# NOTE: the gate-AtP grid (atp_grid_n<N>.npy) is keyed by --n-prompts, so changing --n-prompts
# just writes a new file — no stale-cache cleanup needed.
source scripts/setup_env.sh
module load cuda

python scripts/cineca/circuit_runner.py \
  --model allenai/OLMoE-1B-7B-0924-Instruct \
  --batch-size 6 \
  --atp-batch-size 6 \
  --knockout-k 15 \
  --max-new-tokens 24 \
  --n-prompts 100 \
  --n-test 64
