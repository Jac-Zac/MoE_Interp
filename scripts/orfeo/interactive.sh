#!/bin/bash

if [ -z "$SLURM_JOB_ID" ]; then
    echo "Not running under SLURM, launching with srun..."
    exec srun --pty --partition=GPU --gres=gpu:V100:1 --mem=50G --cpus-per-task=4 --time=00:10:00 "$0" "$@"
fi

source scripts/setup_env.sh
module load cuda

ipython -i notebook.py
