#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --partition=GPU
#SBATCH --gres=gpu:V100:1
#SBATCH -t 04:00:00
#SBATCH --job-name=rigor
#SBATCH -o rigor.out

# Paper-rigor add-ons for the toxicity circuit (reuses the cached gate-AtP grid from circuit.sh,
# so run circuit.sh first with the same --n-prompts). Generation-heavy → 4h budget.
#   1. sufficiency curve  — concept removal vs #experts knocked out (AtP/SOMP/random)
#   2. co-firing group ablation — does removing the redundant group break what the sparse set can't
# Bootstrap CIs run afterwards on CPU (no GPU needed); see commands at the end.

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

source scripts/setup_env.sh
module load cuda

python scripts/rigor/sufficiency_curve.py --n-prompts 100 --n-test 64
python scripts/rigor/group_ablation.py  --n-prompts 100 --n-test 64

echo "GPU work done. Now bootstrap CIs (CPU):"
echo "  python scripts/rigor/bootstrap.py data/*/circuit/rigor/sufficiency_offensive.json"
echo "  python scripts/rigor/bootstrap.py data/*/circuit/rigor/group_ablation_offensive.json"
