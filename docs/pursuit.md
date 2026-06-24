# Pursuit

```bash
python main.py pursuit [--k N] [--min_activations N] [--concept {offensive,countries,numbers}]
```

Pursuit loads the stored activations and projects each expert onto the unembedding
dictionary. Outputs go to `data/<model>/pursuit/<dataset>/`, or `.../<concept>/` when
`--concept` is set. Omit `--concept` to project onto the entire unembedding matrix.

The default dictionary is the model's `lm_head.weight`, row-normalized:

```python
dictionary = F.normalize(get_model_unembedding(model), dim=1)
```

It is cached on disk so pursuit can reuse it without rebuilding the matrix.

Then `pursuit/pursuit.py` runs greedy sparse decomposition (`SOMP`) to find the tokens that best explain the expert activations:

```python
decomposition = SOMP(k=k, compute_evr=True, return_full=False)
result = decomposition(X=X, dictionary=dictionary, descriptors=list(range(len(dictionary))), device=device)
```

Two useful modes exist:

- **Full dictionary**: search over all unembedding rows.
- **Concept restriction**: search over a hand-built concept dictionary such as `offensive`, `countries`, or `numbers`.

Output is written as:

- `results.jsonl` — one record per expert (top-k tokens with EVR scores)
- `evr_matrix.npy` — final EVR per expert
- `count_matrix.npy` — number of activations per expert
- `evr_heatmap.html` / `count_heatmap.html` — EVR and activation-count heatmaps across all layers and experts
