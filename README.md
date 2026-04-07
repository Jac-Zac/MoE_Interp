# Interpretability of Mixture-of-Experts (MoE)

**Expert Pursuit**: An adaptation of the [HeadPursuit](https://github.com/lorenzobasile/HeadPursuit) framework to MoE models.
Projects expert activations onto the unembedding dictionary to identify which experts specialize in which semantic concepts.

Target model: `allenai/OLMoE-1B-7B-0924-Instruct` (16 layers, 64 experts/layer, top-8 routing)

> [!WARNING]
> Work in progress and experimental

## Docs

See [docs/README.md](docs/README.md) for the docs index.

## Quickstart

### Install

```bash
# Create and edit environment config
cp .env.example .env

# Load env vars and activate the project env
source scripts/setup_env.sh

# Install deps
uv sync
```

### Run

```bash
# Download model and datasets
python scripts/download.py --all

# Run tests
python -m pytest

# Capture expert activations
python main.py extract --n_docs 5000

# Multi-GPU capture
torchrun --nproc_per_node=2 main.py extract --model "openai/gpt-oss-20b" --n_docs=10000

# Run pursuit analysis
python main.py pursuit --k 100
```

## Commands

### `extract` - Capture expert activations

```bash
python main.py extract [--model MODEL] [--n_docs N]
```

Saves per-layer HDF5 activations and metadata to `data/<model>/extractions/<dataset>/`.

**Multi-GPU Support:** For better GPU utilization with 2 GPUs, use tensor parallelism:

```bash
torchrun --nproc_per_node=2 main.py extract [--model MODEL] [--n_docs N]
```

This distributes each layer's computation across GPUs (tensor parallelism) instead of splitting layers (pipeline parallelism), resulting in more balanced GPU usage and improved throughput.

### `pursuit` - Run analysis and generate plots

> Concept can also not be passed to obtain the general projection on the entire unembedding matrix

```bash
python main.py pursuit [--k N] [--min_activations N] [--concept {offensive,countries,numbers}]
```

Outputs to `data/<model>/pursuit/<dataset>/` (or `.../<concept>/` when `--concept` is set):

- `results.jsonl` — per-expert top-k tokens with EVR scores
- `evr_heatmap.html` — EVR heatmap across all layers and experts

## Project Structure

```
.
├── main.py                            # CLI entry point
├── notebooks/
│   ├── notebook_extract.py            # Standalone extraction walkthrough
│   ├── notebook_pursuit.py            # Pursuit demo
│   └── notebook_clustering.py         # Experimental K-means clustering analysis
├── src/moe_interp/
│   ├── capture/
│   │   ├── capture.py                 # Expert activation extraction (nnsight)
│   │   ├── cache.py                   # HDF5 storage utilities
│   │   └── model_adapter.py           # Model-specific MoE trace adapters
│   ├── pursuit/
│   │   ├── pursuit.py                 # Projection pursuit orchestration
│   │   ├── decomposition.py           # PCA, OMP, SOMP implementations
│   │   ├── concepts.py                # Word lists (offensive, countries, numbers)
│   │   └── dictionary.py              # Dictionary augmentation utilities
│   ├── io/
│   │   ├── data.py                    # Dataset loading + chat-template formatting
│   │   └── plots.py                   # Plotly EVR/count heatmaps
│   ├── config.py                      # Env loading, device selection, seeds
│   └── parser.py                      # CLI argument parser
├── tests/
│   ├── test_data.py                   # Dataset prompt loading tests
│   ├── test_model_adapter.py          # Model adapter tests
│   └── test_pursuit.py                # Projection pursuit + SOMP unit tests
├── scripts/                           # Setup and cluster scripts
└── pyproject.toml
```

## Token Aggregation Strategies

1. **LAST-TOKEN CAPTURE** (default): Captures expert output at generation-ready position.
   Semantic: "What does this expert contribute to predicting the answer?"

2. **CONTENT-TOKEN AVERAGING** (not yet implemented): Average over question tokens.
   Semantic: "Which experts are consistently used for this question type?"
   MoE consideration: each token routes to only top-k experts (k=8).

## Links

- [HeadPursuit code](https://github.com/lorenzobasile/HeadPursuit)
