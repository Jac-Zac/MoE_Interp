#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --partition=GPU
#SBATCH --gres=gpu:V100:1
#SBATCH -t 00:05:00
#SBATCH --job-name=extract_moe_act
#SBATCH -o test.out  # SLURM stdout

# Source common cache setup
export SCRATCH="/orfeo/scratch/dssc/$USER"
source "slurm_scripts/setup_cache.sh"

# Export env variables
source secrets.txt

# Load the cuda module
module load cuda

# Set up python env
uv sync
source .venv/bin/activate

time ./main.py
