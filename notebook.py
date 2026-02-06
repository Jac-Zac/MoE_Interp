#!/usr/bin/env python

# %% Imports
import nnsight
import torch
from nnsight import LanguageModel

from src.cache import DocumentTrace
from src.data import load_pile_docs
from src.environment import set_seed

# %% Configuration
seed = 1337
set_seed(seed)

# NOTE: Ollmo (allenai/OLMoE-1B-7B-0924-Instruct) Model Spec:
# - Layers                        : 16
# - Experts / layer               : 64
# - Active experts / token        : 8
# - Hidden size                   : 2048

# Use float16 for mps compatibility (bfloat better for CUDA)
model = LanguageModel(
    "allenai/OLMoE-1B-7B-0924-Instruct",
    device_map="auto",
    dtype=torch.float16,
    dispatch=True,
)

# %% Load single document from The Pile
docs, doc_source_ids = load_pile_docs(
    tokenizer=model.tokenizer,
    n_docs=1,
    max_tokens=2048,  # HACK: For testing
    dataset_name="NeelNanda/pile-10k",
)

# Get the document 0 which is the only one in this case
doc = docs[0]
# Get the actual document which corresponds to it
doc_id = doc_source_ids[0]

print(f"Loaded document {doc_id}")
print(f"Doc length: {len(doc)} tokens")

# %% Process document with nnsight
with torch.no_grad(), model.trace([doc]) as tracer:
    layer_indices, layer_weights = [], []

    for layer in model.model.layers:
        # Get routing info from the gate
        # top_k_weights: weight for each expert
        # top_k_indices: expert id active for each token
        # self_gate_0 outputs: (_, top_k_weights, top_k_indices)
        _, weights, indices = layer.mlp.source.self_gate_0.output
        layer_indices.append(indices)
        layer_weights.append(weights)

    # Stack: [n_layers, seq, k]
    indices = torch.stack(layer_indices, dim=0)
    weights = torch.stack(layer_weights, dim=0)

    nnsight.save(indices)
    nnsight.save(weights)

# Create trace object
trace = DocumentTrace(
    expert_indices=indices,
    expert_weights=weights,
    doc_id=doc_id,
)

# %% Results from processing
print(f"\nDocument {doc_id} trace structure:")
print(f"  Layers: {trace.n_layers}")
print(f"  Sequence length: {trace.seq_len}")
print(f"  K (experts per token): {trace.k}")
print(f"  Shape: [layers={trace.n_layers}, seq={trace.seq_len}, k={trace.k}]")

# Get experts for first token
expert_ids, token_weights = trace.get_token(0, 0)
print("\nLayer 0, Token 0:")
print(f"  Expert ids: {expert_ids}")
print(f"  Weights: {token_weights}")
