# Interpretability of Mixture-of-Experts (MoE)

**Expert Pursuit**: An adaptation of the [HeadPursuit](https://github.com/lorenzobasile/HeadPursuit) framework to MoE models.
Projects expert activations onto the unembedding dictionary to identify which experts specialize in which semantic concepts.

Target model: `allenai/OLMoE-1B-7B-0924-Instruct` (16 layers, 64 experts/layer, top-8 routing)

> [!WARNING]
> Work in progress and experimental

## Docs

See [docs/README.md](docs/README.md) for the docs index.

## Usage

Setup (install, downloads, tests) is in [docs/setup.md](docs/setup.md). The pipeline
runs in three stages:

```bash
python main.py extract --n_docs 5000       # capture expert activations
python main.py pursuit --k 100             # rank each expert's tokens
python main.py analysis --dataset pile10k  # logit-lens baseline vs SOMP
```

**`extract`** captures activations, **`pursuit`** ranks each expert's tokens, and
**`analysis`** sanity-checks pursuit against a logit-lens baseline. Multi-GPU capture,
token-selection modes, concept restriction, and all flags are documented in
[docs/](docs/README.md).

## Project Structure

```
.
├── main.py                            # CLI entry point
├── notebooks/
│   ├── notebook_extract.py            # Standalone extraction walkthrough
│   ├── notebook_pursuit.py            # Pursuit demo
│   └── notebook_analysis.py           # Logit-lens vs SOMP walkthrough
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
│   ├── analysis/
│   │   ├── common.py                  # Shared loaders for the post-hoc analyses
│   │   └── logit_lens.py              # Logit-lens baseline vs SOMP (EVR + Jaccard)
│   ├── io/
│   │   ├── data.py                    # Dataset loading + chat-template formatting
│   │   └── plots.py                   # Plotly EVR/count heatmaps
│   ├── config.py                      # Env loading, device selection, seeds
│   └── parser.py                      # CLI argument parser
├── tests/
│   ├── test_data.py                   # Dataset prompt loading tests
│   ├── test_model_adapter.py          # Model adapter tests
│   ├── test_pursuit.py                # Projection pursuit + SOMP unit tests
│   └── test_analysis.py               # Logit-lens cumulative-EVR tests
├── scripts/                           # Setup and cluster scripts
└── pyproject.toml
```

## Links

- [HeadPursuit code](https://github.com/lorenzobasile/HeadPursuit)
