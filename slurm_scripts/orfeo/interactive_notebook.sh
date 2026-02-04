#!/bin/bash

# If not already running under SLURM, re-launch with srun
if [ -z "$SLURM_JOB_ID" ]; then
    echo "Not running under SLURM, launching with srun..."
    exec srun --pty --partition=GPU --gres=gpu:V100:1 --mem=50G --cpus-per-task=4 --time=00:10:00 "$0" "$@"
fi

export SCRATCH="/orfeo/scratch/dssc/$USER"
source "slurm_scripts/setup_cache.sh"
source secrets.txt
module load cuda
uv sync
source .venv/bin/activate

ipython -i notebook.py
