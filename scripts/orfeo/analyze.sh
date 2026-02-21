#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=4
##SBATCH --cpus-per-task=64
#SBATCH --mem=100G
##SBATCH --partition=GENOA
#SBATCH --partition=GPU
#SBATCH --gres=gpu:V100:1
#SBATCH -t 01:00:00
#SBATCH --job-name=analyze_moe_act
#SBATCH -o analyze.out

source scripts/setup_env.sh
module load cuda

python main.py pursuit --k 100
