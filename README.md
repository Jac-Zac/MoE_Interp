# Interpretability of Mixture-of-Experts (MoE)

Two complementary studies of expert specialization in MoE models:

1. **Expert Pursuit** — an adaptation of the [HeadPursuit](https://github.com/lorenzobasile/HeadPursuit)
   framework to MoE: projects expert activations onto the unembedding dictionary to identify which
   experts *associate* with which semantic concepts (descriptive / correlational).
2. **Causal toxic-expert circuit** — which experts *causally* drive toxic generations (activation
   patching + gate-AtP), and how to suppress them at generation time (knockout / project-out).

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

The causal circuit study reuses the captured activations and adds a model-in-the-loop
pipeline (see [docs/circuit.md](docs/circuit.md)):

```bash
python main.py circuit                      # causal activation-patching grid (ground truth)
python main.py circuit-compare              # gate-AtP (1 backward pass) vs the patching grid
python main.py toxic-dla                    # gradient-free DLA classifier (no model)
python main.py circuit-steer                # suppress toxicity: knockout / project-out vs baseline
python main.py circuit-report               # assemble all circuit artifacts into one HTML report
```

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
│   │   ├── logit_lens.py              # Logit-lens baseline vs SOMP (EVR + Jaccard)
│   │   └── toxic_dla.py               # Gradient-free toxic-expert classifier (no model)
│   ├── circuit/                       # Causal toxic-expert study (model in the loop)
│   │   ├── prompts.py                 # Toxic-eliciting + matched neutral prompts
│   │   ├── toxicity.py                # Toxic-logit probe + shared gate-ablation plumbing
│   │   ├── patching.py                # Per-(layer,expert) causal effect grid
│   │   ├── attribution.py             # gate-AtP: whole grid in one backward pass
│   │   ├── compare.py                 # Faithfulness of attributors vs patching
│   │   ├── direction.py               # Diff-of-means toxic direction
│   │   ├── intervene.py               # Generation-time knockout / project-out
│   │   ├── steer.py                   # circuit-steer orchestration
│   │   └── report.py                  # Self-contained HTML circuit report
│   ├── io/
│   │   ├── data.py                    # Dataset loading + chat-template formatting
│   │   └── plots.py                   # Plotly EVR/count heatmaps
│   ├── grids.py                       # top-k helper for layer×expert score grids
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
