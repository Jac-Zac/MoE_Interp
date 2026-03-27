# Pursuit

Pursuit loads the stored activations and projects each expert onto the unembedding dictionary.

The default dictionary is the model's `lm_head.weight`, row-normalized:

```python
dictionary = F.normalize(get_model_unembedding(model), dim=1)
```

It is cached on disk so pursuit can reuse it without rebuilding the matrix.

Then `src/pursuit.py` runs greedy sparse decomposition (`SOMP`) to find the tokens that best explain the expert activations:

```python
decomposition = SOMP(k=k, compute_evr=True, return_full=False)
result = decomposition(X=X, dictionary=dictionary, descriptors=list(range(len(dictionary))), device=device)
```

Two useful modes exist:

- **Full dictionary**: search over all unembedding rows.
- **Concept restriction**: search over a hand-built concept dictionary such as `offensive`, `countries`, or `numbers`.

There is also an optional word dictionary mode that appends common words on top of the base vocabulary.

Output is written as:

- `results.jsonl` — one record per expert
- `evr_matrix.npy` — final EVR per expert
- `count_matrix.npy` — number of activations per expert
