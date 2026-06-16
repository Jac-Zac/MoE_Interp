# Architecture

Expert Pursuit has three `main.py` stages:

1. `extract` captures per-expert activations into HDF5 files.
2. `pursuit` projects each expert's activations onto the unembedding dictionary and ranks the best tokens.
3. `analysis` (no model) compares a bulk logit-lens baseline against SOMP from the stored activations. See [analysis](analysis.md).

Flow: `main.py` -> `io/data.py` -> `capture/capture.py` -> `pursuit/pursuit.py`; then
`analysis/` consumes the stored artifacts.

The CLI entry point is `main.py` (argument parsing lives in `moe_interp/parser.py`).
It wires together:

- `moe_interp/io/data.py` for prompt loading and tokenization
- `moe_interp/capture/capture.py` for tracing and storage
- `moe_interp/pursuit/pursuit.py` for projection pursuit
- `moe_interp/capture/model_adapter.py` for model-specific trace access

Core files on disk look like this:

```text
data/<model>/extractions/<dataset>/metadata.json
data/<model>/extractions/<dataset>/layer_00.h5
data/<model>/unembedding/dictionary.h5
data/<model>/pursuit/<dataset>/results.jsonl
```

The code is intentionally split so model-specific trace details stay in the adapter layer,
while the capture and pursuit logic stays shared.
