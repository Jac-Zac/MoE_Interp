# Interpretability of Mixture-of-Experts (MoE)

**Expert Pursuit**: An adaptation of the [HeadPursuit](https://github.com/lorenzobasile/HeadPursuit) framework to MoE models.
Projects expert activations onto the unembedding dictionary to identify which experts specialize in which semantic concepts.

Target model: `allenai/OLMoE-1B-7B-0924-Instruct` (16 layers, 64 experts/layer, top-8 routing)

> [!WARNING]
> Work in progress and experimental

## Quickstart

```bash
# Setup
source scripts/setup_env.sh

# Run tests
python -m pytest

# Encode documents (capture expert activations)
python main.py encode --n_docs 5000 --batch_size 8

# Run pursuit analysis (generates plots)
python main.py pursuit --k 100
```

## Commands

### `encode` - Capture expert activations

```bash
python main.py encode [options]
```

Options:

- `--n_docs` (int, default: 5000): Number of documents to encode
- `--split` (str, default: "train"): Dataset split
- `--batch_size` (int, default: 8): Batch size for tracing

### `pursuit` - Run analysis and generate plots

```bash
python main.py pursuit [options]
```

Options:

- `--k` (int, default: 50): Top tokens per expert by explained variance

Outputs:

- `data/pursuit/pursuit_results.json` - Per-expert concept decompositions
- `data/plots/evr_heatmap.html` - EVR heatmap across all experts
- `data/plots/concept_frequency.html` - Most frequent concepts

## Project Structure

```
.
├── main.py                 # CLI entry point
├── notebooks/
│   └── notebook_pursuit.py # Jupyter demo
├── src/
│   ├── capture.py          # Expert activation extraction (nnsight)
│   ├── cache.py            # HDF5 storage
│   ├── constants.py        # Word lists (countries, colors, quantity)
│   ├── data.py             # TriviaQA loading
│   ├── environment.py      # Model loading, device, seeds
│   ├── plot.py             # Plotly visualizations
│   └── pursuit.py          # Projection pursuit analysis
├── tests/
│   └── test_core.py        # Pytest suite
├── scripts/                # SLURM cluster scripts
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
