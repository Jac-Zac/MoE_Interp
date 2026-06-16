# Setup

## Install

```bash
cp .env.example .env          # then edit it (HF token, paths)
source scripts/setup_env.sh   # load env vars + activate the project env
uv sync                       # install deps
```

Requires Python >=3.13 (see `.python-version`) and [uv](https://docs.astral.sh/uv/).

## Download model and datasets

```bash
python scripts/download.py --all
```

## Run tests

Tests use random tensors (no model load), so they are fast:

```bash
python -m pytest
```
