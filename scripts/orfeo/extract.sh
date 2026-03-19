#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --partition=GPU
#SBATCH --gres=gpu:V100:1
#SBATCH -t 01:30:00
#SBATCH --job-name=extract_moe_act
#SBATCH -o extract.out

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

source scripts/setup_env.sh
module load cuda

python main.py extract --n_docs 25000 --batch_size 128
