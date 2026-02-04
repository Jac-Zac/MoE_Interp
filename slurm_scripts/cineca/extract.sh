#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH -A uTS25_Tornator
#SBATCH -t 00:20:00
##SBATCH --exclusive
#SBATCH --job-name=self_attention_bench

# Source common cache setup
# export SCRATCH="/cineca/scratch/$USER"
source "slurm_scripts/setup_cache.sh"

# module load nvhpc
# module load profile/deeplrn
# module load cineca-ai

# Load the uv enviroment
uv sync
source .venv/bin/activate

./main.py
