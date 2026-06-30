#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --partition=GPU
#SBATCH --gres=gpu:V100:1
#SBATCH -t 02:00:00
#SBATCH --job-name=triviaqa_headline
#SBATCH -o triviaqa.out

# Regenerate the OLMoE report headline (tab:experts + median EVR): the TriviaQA
# extraction + full-dictionary Expert Pursuit that no longer exists on scratch.
# Done as one GPU job — at 10k docs the extraction is ~pile10k-scale (~minutes) and
# pursuit is fast, so both fit comfortably under the 2h GPU association cap.
#
# NOTE: extraction opens its HDF5 in append mode and is NOT partial-resumable, so it
# must finish in one window (10k docs does, easily). The report prose says "50,000
# TriviaQA"; this run uses 10k (per request) — update the wording when reading results.
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

source scripts/setup_env.sh
module load cuda

MODEL=allenai/OLMoE-1B-7B-0924-Instruct

# 1) extract gated expert activations on 10k TriviaQA questions
python main.py extract --model "$MODEL" --dataset triviaqa --n_docs 10000 --batch_size 64

# 2) full-dictionary pursuit (k=50 tokens/expert) -> pursuit/triviaqa/{results.jsonl,evr_matrix.npy}
python main.py pursuit --model "$MODEL" --dataset triviaqa --k 50
