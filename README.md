# Interpretability of Mixture-of-Experts (MoE)

**Expert Pursuit**: An adaptation of the [HeadPursuit](https://github.com/lorenzobasile/HeadPursuit) framework to MoE models.
Uses Simultaneous Orthogonal Matching Pursuit (SOMP) to identify which experts specialize in which semantic concepts by analyzing gated expert outputs against the model's unembedding matrix.

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
python main.py encode --n_docs 100 --max_tokens 512

# Run SOMP analysis
python main.py pursuit --concept countries --k 50
```

## Commands

### `encode` - Capture expert activations

```bash
python main.py encode [options]
```

Options:

- `--n_docs` (int, default: 100): Number of documents to encode
- `--max_tokens` (int, default: 512): Max tokens per document
- `--truncate`: Truncate long documents instead of skipping

### `pursuit` - Run SOMP analysis

```bash
python main.py pursuit --concept <name> [options]
```

Options:

- `--concept` (required): Concept name - `countries`, `colors`, `numbers`, or `full`
- `--k` (int, default: 50): Number of SOMP iterations

**Available concepts:** `countries`, `colors`, `numbers`

## Project Structure

```
.
├── main.py                 # CLI entry point
├── notebooks/
│   ├── notebook_base.py    # Jupyter demo (nnsight tracing + SOMP)
│   └── notebook_lens.py    # Expert Logit Lens visualization
├── src/
│   ├── capture.py         # Expert activation extraction
│   ├── cache.py           # HDF5-backed storage
│   ├── somp.py            # SOMP algorithm
│   ├── dictionary.py      # Unembedding/concept dictionaries
│   ├── pursuit.py         # Expert Pursuit analysis
│   ├── data.py            # Dataset loading
│   └── environment.py     # Utils (seeds, device)
├── tests/
│   └── test_core.py       # Pytest suite
├── scripts/               # SLURM cluster scripts
│   ├── orfeo/
│   └── cineca/
└── pyproject.toml         # Dependencies
```

# Personal Notes

- [Head pursuit code](https://github.com/lorenzobasile/HeadPursuit)

- MLP -> Molgeva (perhaps use the other techniques that you found)
- Weight interpetrability of the weight (and also ROME)

# NNsight

1. Phepras you can try to use vLLM infernece engine and test it with a small MoE model to actually see the basic performance
2. The next step would be to run inference on some prompts, eg 5 random prompts that you can import from a file or something. Or potentially laoding just one part of a dataset
3. Cache all activations and write them to disk every (tot time to avoid wasating things -> In this case I can write to the scratch partition)

Try also to only save one expert for example. Then read head persuit algorithm and try to do based on that

-> Be careful of using the correct template. If I need to add tokens or doing somehting with the prefix or something

# NOTE:

Review properties in python datacasses

- You can add alliasing like this https://github.com/ndif-team/nnsight/blob/4f2bdbe8aacb19b02a618ed222545b4d91e3b731/src/nnsight/intervention/envoy.py#L1339
