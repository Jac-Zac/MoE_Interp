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

# gate-AtP localization + localization report. Computes the causal-effect grid (one backward
# pass) over the eliciting train prompts and renders the gate-AtP heatmap + faithfulness report.
# The grid is shared with the knockout/downweighting sweep (scripts/orfeo/downweight.sh), which is
# where the intervention results are produced.
#
# NOTE: the gate-AtP grid (atp_grid_n<N>.npy) is keyed by --n-prompts, so changing --n-prompts
# just writes a new file — no stale-cache cleanup needed.
source scripts/setup_env.sh
module load cuda

# Extra args pass through, e.g. a higher-toxicity regime (writes a regime-tagged grid so it
# won't clobber the default run):  sbatch scripts/orfeo/circuit.sh --hi 0.8 --challenging
python scripts/cineca/circuit_runner.py \
  --model allenai/OLMoE-1B-7B-0924-Instruct \
  --batch-size 6 \
  --atp-batch-size 6 \
  --n-prompts 100 \
  --n-test 64 \
  "$@"
