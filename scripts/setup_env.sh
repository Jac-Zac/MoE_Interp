#!/bin/bash
# Setup script: sync deps, load env vars, activate venv
# Usage: source scripts/setup_env.sh

PROJECT_ROOT=$(git rev-parse --show-toplevel)
cd "$PROJECT_ROOT"

# Sync dependencies
uv sync

# Load environment variables
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Activate Python venv
source "$PROJECT_ROOT/.venv/bin/activate"

# Set up source to project root
export PYTHONPATH=$PWD
