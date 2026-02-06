# Interpretability of Mixture-of-Experts (MoE)

This is a project on the Interpretability of MoE models.

> [!WARNING]
> This is a work in progress and it is extremily experimental

### Project Scope

1. Extracting activations from experts
2. Analyzing them with ... (head persuite perhaps)

## Usage

### Local Usage

```bash
source scripts/setup_env.sh
python main.py
```

### Cluster Usage

1. For running the extraction on the cluster

```bash
sbatch scripts/orfeo/extract.sh
```

2. For running the analyses

```bash
sbatch scripts/...
```

## Project Structure

```bash
.
├── main.py              # Entry point
├── notebook.py          # Jupyter notebook scripts
├── src/
│   ├── capture.py       # Expert activation extraction
│   ├── cache.py         # MoETrace dataclass
│   ├── checkpoint.py    # Batch saving/loading
│   ├── data.py          # Dataset loading
│   ├── analyze.py       # Analysis functions
│   └── environment.py   # Utils (seeds, device)
├── scripts/             # SLURM cluster scripts
│   ├── orfeo/
│   └── cineca/
├── pyproject.toml       # Dependencies and config
└── README.md
```
