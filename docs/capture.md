# Capture

Capture stores expert outputs for selected real tokens in each prompt. The default
is the **last real token** of each prompt. Pass `--token_selection all` to store
all real non-padding tokens routed to each expert. This matters because the model
uses RoPE, so token position must stay correct.

Prompts are sorted by length and batched with **right-padding**:

```python
ds = ds.sort("length", reverse=True)
model.tokenizer.padding_side = "right"
```

For a batch with `max_len`, the last token for prompt `i` is at:

```python
prompt_offset = i * max_len
last_token_idx = prompt_offset + actual_len_i - 1
```

In practice, `src/capture.py` computes all last-token positions at once and filters the traced expert outputs with a mask:

```python
batch_offsets = torch.arange(b_size) * max_len
last_positions = batch_offsets + actual_lens_tensor - 1
is_last = torch.isin(token_idx, last_positions)
```

Only the selected expert outputs are written. Each layer gets one HDF5 file and each expert gets a group inside it:

```text
layer_00.h5 -> expert_000/activations, expert_000/tokens, ...
```

Each expert group stores:

- `activations`: routed expert contribution vectors.
- `tokens`: source token ids for the stored positions.
- `routing_weights`: router weight for the selected expert/token pair.
- `positions`: token position within the prompt.

Notes:

- `src/model_adapter.py` owns the model-specific trace node paths.
- `src/cache.py` handles HDF5 append/load logic.
- The tokenizer padding side is restored to `"left"` after capture.
