#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH -A uTS25_Tornator
#SBATCH -t 00:30:00
##SBATCH --exclusive
#SBATCH --job-name=extract_moe_act
#SBATCH -o extract.out

source scripts/setup_env.sh
python main.py encode --n_docs 50000 --batch_size 64
