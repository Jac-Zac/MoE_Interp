#!/bin/bash
# Setup script: load env vars, sync deps, activate venv
# Usage: source scripts/setup_env.sh

SOURCE="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="$(cd "$(dirname "$SOURCE")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load environment variables from .env (exported to shell)
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
    echo "Loaded environment from .env"
else
    echo "Warning: .env file not found at $PROJECT_ROOT/.env"
fi

# Sync dependencies if uv is available
if command -v uv &> /dev/null; then
    (cd "$PROJECT_ROOT" && uv sync)
fi

# Activate Python venv if it exists
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
fi
