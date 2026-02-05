#!/usr/bin/env python

# %% Imports
import nnsight
import torch
from nnsight import LanguageModel

from src.cache import MoETrace
from src.data import load_pile_docs
from src.environment import set_seed

# %% Model Definition
seed = 1337
n_docs = 10

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
    # dispatch=False,
    dispatch=True,
)

# %% Load data from The Pile
docs = load_pile_docs(
    tokenizer=model.tokenizer,
    n_docs=n_docs,
    # max token based on what the model can actually do
    # max_tokens=model.config.max_position_embeddings,
    max_tokens=100,  # HACK: For testing
)
print(f"Loaded {len(docs)} documents")

# %% Capture expert activations

# Pre-compute document boundaries for token-to-doc mapping
doc_lens = [len(doc) for doc in docs]
doc_boundaries = torch.tensor([0] + [sum(doc_lens[: i + 1]) for i in range(len(docs))])

# HACK: I think this set up will let nnsight automatically batch queries
# In the way it believes to be best though I need to check
with torch.no_grad(), model.trace(docs) as tracer:
    # Collect per-layer routing info
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
    # More efficient: only 2 tensors to save/extract instead of 2*n_layers
    indices_stack = torch.stack(layer_indices, dim=0)
    weights_stack = torch.stack(layer_weights, dim=0)

    nnsight.save(indices_stack)
    nnsight.save(weights_stack)
    nnsight.save(doc_boundaries)

# %% Build MoETrace from captured tensors
trace = MoETrace.build(
    docs=docs,
    indices_stack=indices_stack,
    weights_stack=weights_stack,
    doc_boundaries=doc_boundaries,
)

# %% Results
print(f"Total tokens: {len(trace.token_ids)}")
print(
    f"Shape: [total_tokens={trace.token_ids.shape[0]}, layer={trace.n_layers}, k={trace.k}]"
)

expert_id, weights = trace.experts_for_token(0, 0)
print("\nLayer 0, Token 0:")
print(f"- Expert id: {expert_id}\n")
print(f"- Token id: {expert_id}")
