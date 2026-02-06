#!/usr/bin/env python

# %% Imports
import nnsight
import torch
from nnsight import LanguageModel

from src.cache import MoETrace
from src.data import load_pile_docs
from src.environment import set_seed

# %% Configuration
seed = 1337
n_docs = 2
# Process documents in batches to manage memory
BATCH_SIZE = n_docs

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

# %% Load data from The Pile
docs, doc_source_ids = load_pile_docs(
    tokenizer=model.tokenizer,
    n_docs=n_docs,
    max_tokens=2048,  # HACK: For testing
    dataset_name="NeelNanda/pile-10k",
)

print(f"Loaded {len(docs)} documents")
print(f"Source doc indices: {doc_source_ids}")

# %% Process documents in batches (in this simple case I work with only 1 batch)
# Pre-compute document boundaries for token-to-doc mapping
doc_lengths = torch.tensor([len(doc) for doc in docs])
batch_boundaries = torch.cat([torch.tensor([0]), doc_lengths.cumsum(0)])

# HACK: I think this set up will let nnsight automatically batch queries
# In the way it believes to be best though I need to check
with torch.no_grad(), model.trace(docs) as tracer:
    layer_indices, layer_weights = [], []

    for layer in model.model.layers:
        # Get routing info from the gate
        # top_k_weights: weight for each expert
        # top_k_indices: expert id active for each token

        # self_gate_0 outputs: (_, top_k_weights, top_k_indices)
        _, weights, indices = layer.mlp.source.self_gate_0.output
        layer_indices.append(indices)
        layer_weights.append(weights)

    # Stack layers inside trace context: [n_layers, batch, seq, k]
    indices_stack = torch.stack(layer_indices, dim=0)
    weights_stack = torch.stack(layer_weights, dim=0)

    nnsight.save(indices_stack)
    nnsight.save(weights_stack)

trace = MoETrace.build(
    docs=docs,
    indices_stack=indices_stack,
    weights_stack=weights_stack,
    doc_boundaries=batch_boundaries,
    doc_source_ids=doc_source_ids,
)

# %% Results from first batch
print(f"Example from batch 0:")
print(f"Total tokens: {len(trace.token_ids)}")
print(
    f"Shape: [tokens={trace.token_ids.shape[0]}, layers={trace.n_layers}, k={trace.k}]"
)

expert_ids, weights = trace.experts_for_token(0, 0)
print("\nLayer 0, Token 0:")
print(f"- Expert ids: {expert_ids}")
print(f"- Weights: {weights}")

# %%
# print(trace)
# NOTE: Try to understand why expert_idices is not the same as token_idx which is the expected shape
print(trace.expert_indices.shape)
print(trace.token_ids.shape)
print(f"{len(docs[0]) = }, {len(docs[1]) = }")
