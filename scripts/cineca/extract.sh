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

source scripts/setup_env.sh

python main.py
