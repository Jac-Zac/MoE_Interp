# Interpretability of Mixture-of-Experts (MoE)

This is a project on the Interpretability of MoE models.

> [!WARNING]
> This is a work in progress and it is extremily experimental

### Project Scope

1. Extracting activations from experts
2. Analyzing them with head pursuit

## Usage

### Local Usage

```bash
python main.py
```

### Cluster Usage

1. For running the extraction on the cluster

```bash
sbatch slurm_scripts/orfeo/extract.sh
```

2. For running the analyses

```bash
sbatch ...
```

## Project Structure

```bash
.
├── main.py              # Entry point
├── src/
│   ├── capture.py       # Expert activation extraction
│   ├── analyze.py       # Analysis functions
│   └── environment.py   # Utils (seeds, device)
├── pyproject.toml       # Dependencies and config
└── README.md
```
