#!/usr/bin/env python

# %% Imports
import torch
from nnsight import LanguageModel

from src.cache import MoETrace
from src.data import load_pile_docs
from src.environment import set_seed

# %% Model Definition
seed = 1337
batch_size = 8
n_chunks = 10

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
    dispatch=False,
)

# %% Load data from The Pile
docs = load_pile_docs(
    tokenizer=model.tokenizer,
    n_docs=n_chunks,
    # max token based on what the model can actually do
    max_tokens=model.config.max_position_embeddings,
    seed=seed,
)

print(f"Loaded {docs.shape[0]} documents of size {docs.shape[1]}")

# %% Capture expert activations

# Process in batches
all_indices, all_weights = [], []

for i in range(0, len(docs), batch_size):
    batch = docs[i : i + batch_size]

    with torch.no_grad(), model.trace(batch) as tracer:
        for layer in model.model.layers:
            # Get routing info from the gate
            # top_k_weights: weight for each expert
            # top_k_indices: expert id active for each token

            # self_gate_0 outputs: (_, top_k_weights, top_k_indices)
            _, weights, indices = layer.mlp.source.self_gate_0.output
            all_indices.append(indices.save())
            all_weights.append(weights.save())

# %% Build MoETrace from captured tensors
trace = MoETrace.from_tensors(
    token_ids=docs,
    indices_list=all_indices,
    weights_list=all_weights,
    num_experts=model.model.config.num_experts,
)

# %% Results
print(f"\nexpert_indices shape: {trace.expert_indices.shape}")  # [batch, seq, layer, k]
print(f"expert_weights shape: {trace.expert_weights.shape}")  # [batch, seq, layer, k]
print(f"Active (layer, expert) pairs: {len(trace.expert_token_idx)}")
print(f"\nDoc 0, Token 0, Layer 0 experts: {trace.expert_indices[0, 0, 0, :]}")
