#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --partition=GPU
#SBATCH --gres=gpu:V100:1
#SBATCH -t 00:10:00
#SBATCH --job-name=interactive_notebook

export SCRATCH="/orfeo/scratch/dssc/$USER"
source "slurm_scripts/setup_cache.sh"
source secrets.txt
module load cuda
uv sync
source .venv/bin/activate

ipython -i notebook.py
