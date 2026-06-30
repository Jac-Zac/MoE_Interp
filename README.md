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
pipeline. It is kept out of the CLI as `# %%` walkthroughs under `notebooks/circuits/`
(it was the most experimental part of the project; see [docs/circuit.md](docs/circuit.md)):

```bash
python main.py pursuit --concept offensive   # classify: SOMP experts whose atoms are offensive
python notebooks/circuits/localize.py         # localize: gate-AtP grid + faithfulness (+ report)
python scripts/cineca/downweight_runner.py    # intervene: knockout / downweighting sweep + CIs
```

## Project Structure

```
.
├── main.py                            # CLI entry point
├── notebooks/
│   ├── notebook_extract.py            # Standalone extraction walkthrough
│   ├── notebook_pursuit.py            # Pursuit demo
│   ├── notebook_analysis.py           # Logit-lens vs SOMP walkthrough
│   └── circuits/                      # Causal toxic-expert study (# %% walkthroughs)
│       └── localize.py               # localize: gate-AtP grid + faithfulness
├── src/moe_interp/
│   ├── capture/
│   │   ├── capture.py                 # Expert activation extraction (nnsight)
│   │   ├── cache.py                   # HDF5 storage utilities
│   │   └── model_adapter.py           # Model-specific MoE trace adapters
│   ├── pursuit/
│   │   ├── pursuit.py                 # Projection pursuit orchestration
│   │   ├── decomposition.py           # PCA, OMP, SOMP implementations
│   │   └── concepts.py                # Word lists (offensive, countries, numbers)
│   ├── analysis/
│   │   ├── common.py                  # Shared loaders for the post-hoc analyses
│   │   └── logit_lens.py              # Logit-lens baseline vs SOMP (EVR + Jaccard)
│   ├── circuit/                       # Causal toxic-expert study (model in the loop)
│   │   ├── prompts.py                 # RealToxicityPrompts eliciting/neutral split
│   │   ├── toxicity.py                # Toxic-logit probe + shared gate-ablation plumbing
│   │   ├── patching.py                # Per-(layer,expert) causal effect grid
│   │   ├── attribution.py             # gate-AtP: whole grid in one backward pass
│   │   ├── intervene.py               # Generation-time gate knockout / downweighting
│   │   ├── expert_sets.py             # SOMP / gate-AtP / matched-random expert sets
│   │   ├── downweight.py              # Knockout/downweighting sweep + bootstrap error bars
│   │   └── report.py                  # Self-contained HTML localization report
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
