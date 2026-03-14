#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=GPU
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=100G
#SBATCH -t 01:00:00
#SBATCH --job-name=analyze_moe_act
#SBATCH -o analyze.out

# uv sync
source .venv/bin/activate
module load cuda

python main.py pursuit --k 50
# python main.py pursuit --k 50 --concept countries
# python main.py pursuit --k 50 --concept offensive
# python main.py pursuit --k 50 --concept numbers
