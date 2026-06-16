# Analysis

Post-hoc analysis that reads only the stored activations (HDF5 extractions) and the SOMP
`results.jsonl` — no model forward pass.

```bash
python main.py analysis [--dataset DATASET] [--extractions_dir DIR] [--pursuit_dir DIR]
```

Inputs default to the standard local layout (`data/<model>/extractions/<dataset>` and
`data/<model>/pursuit/<dataset>`); pass `--extractions_dir` / `--pursuit_dir` to read from
elsewhere. Outputs `logit_lens_comparison.json` to `data/<model>/analysis/<dataset>/`.
Interactive version: `notebooks/notebook_analysis.py` (`# %%` cells).

## Logit-lens baseline vs SOMP (`analysis/logit_lens.py`)

The bulk logit lens ranks tokens by the expert's **mean** activation, `topk(D @ mean(A))`.
SOMP instead selects a **basis** of atoms that explain the **variance** of the (centred)
activations. To compare fairly, both methods' atoms reconstruct the same centred `A` via
least squares and we read the cumulative explained-variance ratio (EVR) — the identical
estimator used in `pursuit.decomposition.somp`, so it is comparable to the stored EVR.

Reported per expert and aggregated: top-10 token overlap (Jaccard) and EVR at depths 1/3/10. Writes `logit_lens_comparison.json`.

The headline: a single top-k token ranking explains little of an expert's variance, while
SOMP's first few atoms explain much more — i.e. a single top-k logit-lens token under-reads
a polysemantic expert (the motivation for the multi-atom SOMP basis), consistent with the
interpretability literature that MoE experts are polysemantic (in superposition).
