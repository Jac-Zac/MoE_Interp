#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --partition=GPU
#SBATCH --gres=gpu:V100:1
#SBATCH -t 00:25:00
#SBATCH --job-name=extract_moe_act
#SBATCH -o test.out

source scripts/setup_env.sh
module load cuda

time python main.py pursuit --concept countries --k 50

