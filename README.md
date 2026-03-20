# Interpretability of Mixture-of-Experts (MoE)

**Expert Pursuit**: An adaptation of the [HeadPursuit](https://github.com/lorenzobasile/HeadPursuit) framework to MoE models.
Projects expert activations onto the unembedding dictionary to identify which experts specialize in which semantic concepts.

Target model: `allenai/OLMoE-1B-7B-0924-Instruct` (16 layers, 64 experts/layer, top-8 routing)

> [!WARNING]
> Work in progress and experimental

## Quickstart

```bash
# Setup: copy and edit .env
cp .env.example .env
# Edit .env with your HF_TOKEN and cache directories

# Load environment (run once per shell session)
source scripts/setup_env.sh

# Download model and datasets
python scripts/download.py --all

# Run tests
python -m pytest

# Run extraction (capture expert activations)
python main.py extract --n_docs 5000

# For tensor parallelism (2 GPUs), use torchrun for better GPU utilization:
torchrun --nproc_per_node=2 main.py extract --model "openai/gpt-oss-20b" --n_docs=10000

# Run pursuit analysis (generates plots)
python main.py pursuit --k 100
```

## Commands

### `extract` - Capture expert activations

```bash
python main.py extract [--model MODEL] [--n_docs N]
```

Saves per-layer HDF5 activations and metadata to `data/extractions/`.

**Multi-GPU Support:** For better GPU utilization with 2 GPUs, use tensor parallelism:

```bash
torchrun --nproc_per_node=2 python main.py extract [--model MODEL] [--n_docs N]
```

This distributes each layer's computation across GPUs (tensor parallelism) instead of splitting layers (pipeline parallelism), resulting in more balanced GPU usage and improved throughput.

### `pursuit` - Run analysis and generate plots

> Concept can also not be passed to obtain the general projection on the entire unembedding matrix

```bash
python main.py pursuit [--k N] [--min_activations N] [--concept {offensive,countries,numbers}]
```

Outputs to `data/pursuit/` (or `data/pursuit/<concept>/` when `--concept` is set):

- `results.jsonl` — per-expert top-k tokens with EVR scores
- `evr_heatmap.html` — EVR heatmap across all layers and experts

## Project Structure

```
.
├── main.py                        # CLI entry point
├── notebooks/
│   ├── notebook_extract.py         # Standalone extraction walkthrough
│   ├── notebook_pursuit.py        # Pursuit demo
│   └── notebook_pursuit_marimo.py # Interactive Marimo explorer
├── src/
│   ├── capture.py                 # Expert activation extraction (nnsight)
│   ├── cache.py                   # HDF5 storage utilities
│   ├── concepts.py                # Word lists (offensive, countries, numbers)
│   ├── data.py                    # TriviaQA loading + chat-template formatting
│   ├── environment.py             # Env loading, device selection, seeds
│   ├── plots.py                   # Plotly EVR heatmap
│   ├── pursuit.py                 # Projection pursuit (SOMP over unembedding dict)
│   └── sparse_decomposition.py   # PCA, OMP, SOMP implementations
├── tests/
│   ├── test_core.py               # HDF5 round-trip tests
│   └── test_pursuit.py            # Projection pursuit + SOMP unit tests
├── scripts/                       # SLURM scripts (Cineca, Orfeo)
└── pyproject.toml
```

## Token Aggregation Strategies

1. **LAST-TOKEN CAPTURE** (default): Captures expert output at generation-ready position.
   Semantic: "What does this expert contribute to predicting the answer?"

2. **CONTENT-TOKEN AVERAGING** (not yet implemented): Average over question tokens.
   Semantic: "Which experts are consistently used for this question type?"
   MoE consideration: each token routes to only top-k experts (k=8).

## Marimo Explorer

```bash
uv run marimo run notebooks/notebook_pursuit_marimo.py
```

## Links

- [HeadPursuit code](https://github.com/lorenzobasile/HeadPursuit)
