#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=100G
#SBATCH --partition=GENOA
#SBATCH -t 00:30:00
#SBATCH --job-name=analyze_moe_act
#SBATCH -o analyze.out

# Worst alternative
##SBATCH --cpus-per-task=4
##SBATCH --partition=GPU
##SBATCH --gres=gpu:V100:1

source scripts/setup_env.sh
module load cuda

python main.py pursuit --k 100
