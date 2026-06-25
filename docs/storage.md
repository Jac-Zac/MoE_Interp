# Storage

The repository uses a simple on-disk layout so capture and pursuit can run separately.

Model names are sanitized by `moe_interp/config.py` (`/` and `-` become `_`) before they become directory names.

## Metadata

`metadata.json` stores the model, dataset, and layer/expert counts.

## Directory layout

```text
data/<model>/
  extractions/<dataset>/
    metadata.json
    layer_00.h5
    layer_01.h5
    ...
  unembedding/
    dictionary.h5
  pursuit/<dataset>/
    results.jsonl
    evr_matrix.npy
    count_matrix.npy
    evr_heatmap.html
    count_heatmap.html
  analysis/<dataset>/
    logit_lens_comparison.json
```

## Activations

Each layer is stored in one HDF5 file:

```text
layer_00.h5
layer_01.h5
...
```

Each expert inside the file gets a group like `expert_012/`:

- `activations` — the selected (last-token) gated expert outputs
- `tokens` — token ids for those activations
- `routing_weights` — the router gate weight for each stored expert/token row

## Unembedding cache

The normalized unembedding matrix is cached at:

```text
data/<model>/unembedding/dictionary.h5
```

This avoids rebuilding the dictionary every time pursuit runs.

## Why HDF5

HDF5 lets us append expert rows incrementally during capture without keeping everything in memory.
