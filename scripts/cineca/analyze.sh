#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --partition=dcgp_usr_prod
#SBATCH -A uTS25_Tornator_0
#SBATCH -t 00:30:00
#SBATCH --job-name=analyze_moe_act
#SBATCH -o analyze.out

source scripts/setup_env.sh
python main.py pursuit --k 100
