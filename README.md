# MoE Interpretability

Expert Pursuit adapts
[HeadPursuit](https://github.com/lorenzobasile/HeadPursuit) to study semantic specialization in
Mixture-of-Experts language models. It projects gated expert outputs onto the normalized
unembedding dictionary and includes exploratory gate-attribution and intervention experiments.

Target model: `allenai/OLMoE-1B-7B-0924-Instruct`.

> Experimental research code. Interpret results with the limitations discussed in the report.

## Usage

```bash
uv sync
python main.py extract --n_docs 5000 --batch_size 8
python main.py pursuit --k 50
python -m pytest tests/ -v
```

Use `torchrun --nproc_per_node=2` before `python main.py extract ...` for two-GPU extraction.

## Documentation

- [Report](report/main.pdf)
- [Setup and usage](docs/README.md)
- [Causal circuit experiments](docs/circuit.md)
