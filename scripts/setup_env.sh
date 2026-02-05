#!/bin/bash
# Setup script: sync deps, load env vars, activate venv
# Usage: source scripts/setup_env.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

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
