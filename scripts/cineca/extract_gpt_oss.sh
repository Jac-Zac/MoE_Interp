#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH -A uTS26_Tornator
#SBATCH -t 02:00:00
##SBATCH --exclusive
#SBATCH --job-name=extract_gpt_oss
#SBATCH -o extract_gpt_oss.out

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

source scripts/setup_env.sh
python main.py extract \
  --model openai/gpt-oss-20b \
  --n_docs 25000 \
  --batch_size 64 \
  --dataset pile10k
