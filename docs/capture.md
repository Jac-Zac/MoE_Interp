# Capture

```bash
python main.py extract [--model MODEL] [--n_docs N] [--dataset NAME] [--max_length L]
```

Saves per-layer HDF5 activations and `metadata.json` to
`data/<model>/extractions/<dataset>/` (see [storage](storage.md)).

## What gets captured

Capture stores each expert's output contribution at the **last real token** of every
prompt — the generation-ready position, "what does this expert contribute to predicting
the next token?". One forward pass per batch traces the fused-experts boundary, and the
per-expert contribution is reconstructed outside the trace (see
`capture/model_adapter.py`). Over a large document set each expert accumulates plenty of
rows (one per prompt it was routed for), which is what SOMP needs.

The model uses RoPE, so token position must stay correct: prompts are sorted by length and
batched with **right-padding** so each token keeps its true position.

```python
ds = ds.sort("length", reverse=True)
model.tokenizer.padding_side = "right"
```

For a batch with `max_len`, the last token for prompt `i` sits at `i * max_len +
actual_len_i - 1`; `capture/capture.py` builds a boolean mask over the flattened token
axis to keep exactly those rows.

Each layer gets one HDF5 file and each expert a group inside it:

```text
layer_00.h5 -> expert_000/activations, expert_000/tokens, expert_000/routing_weights
```

Each expert group stores:

- `activations`: routed expert contribution vectors (gate + component-RMSNorm folded in).
- `tokens`: source token id for each stored row.
- `routing_weights`: the router gate weight for the selected expert/token pair.

Notes:

- `capture/model_adapter.py` owns the model-specific expert reconstruction math.
- `capture/cache.py` handles HDF5 append/load logic.
- The tokenizer padding side is restored after capture.

> **Future work — per-token tracing.** Capture currently keeps only the last token of each
> prompt. Storing *all* content tokens (averaging an expert over the positions it fires at,
> rather than a single last token) would give a richer, context-diverse view of each
> expert; it was dropped here to keep storage and the pipeline simple, and because
> last-token over many documents is already enough for SOMP.
