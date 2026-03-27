# Architecture

Expert Pursuit has two stages:

1. `extract` captures per-expert activations into HDF5 files.
2. `pursuit` projects each expert's activations onto the unembedding dictionary and ranks the best tokens.

Flow: `main.py` -> `src/data.py` -> `src/capture.py` -> `src/pursuit.py`.

The CLI entry point is `main.py`. It wires together:

- `src/data.py` for prompt loading and tokenization
- `src/capture.py` for tracing and storage
- `src/pursuit.py` for projection pursuit
- `src/model_adapter.py` for model-specific trace access

Core files on disk look like this:

```text
data/<model>/extractions/<dataset>/metadata.json
data/<model>/extractions/<dataset>/layer_00.h5
data/<model>/unembedding/dictionary.h5
data/<model>/pursuit/<dataset>/results.jsonl
```

The code is intentionally split so model-specific trace details stay in the adapter layer,
while the capture and pursuit logic stays shared.
