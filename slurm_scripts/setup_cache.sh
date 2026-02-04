#!/bin/bash
# Common setup script for HuggingFace cache configuration on HPC clusters
# Source this script from cluster-specific job scripts

# Setup scratch directories for HuggingFace cache
# This prevents filling up the home directory with model downloads
SCRATCH_DIR="${SCRATCH}"

# Create cache directories
echo "Setting up cache directories in scratch: $SCRATCH_DIR"
mkdir -p "$SCRATCH_DIR/huggingface/cache/hub"
mkdir -p "$SCRATCH_DIR/huggingface/datasets"
mkdir -p "$SCRATCH_DIR/torch"

# Set environment variables for HuggingFace and PyTorch caches
export HF_HOME="$SCRATCH_DIR/huggingface/cache"
export HF_HUB_CACHE="$SCRATCH_DIR/huggingface/cache/hub"
export HF_DATASETS_CACHE="$SCRATCH_DIR/huggingface/datasets"
export TORCH_HOME="$SCRATCH_DIR/torch"

echo "HF_HOME: $HF_HOME"
echo "HF_HUB_CACHE: $HF_HUB_CACHE"
echo "HF_DATASETS_CACHE: $HF_DATASETS_CACHE"
echo "TORCH_HOME: $TORCH_HOME"
