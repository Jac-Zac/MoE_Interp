#!/usr/bin/env python

# %% Imports
import nnsight
import torch
from nnsight import LanguageModel
from tqdm import tqdm

from src.cache import MoETrace
from src.data import load_pile_docs
from src.environment import set_seed

# %% Configuration
seed = 1337
n_docs = 10
# Process documents in batches to manage memory
BATCH_SIZE = 8

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
docs = load_pile_docs(
    tokenizer=model.tokenizer,
    n_docs=n_docs,
    max_tokens=100,  # HACK: For testing
)

print(f"Loaded {len(docs)} documents")

# %% Process documents in batches


def process_batch(model, batch_docs):
    """Process a single batch and return expert trace."""

    # Pre-compute document boundaries for token-to-doc mapping
    batch_doc_lens = [len(doc) for doc in batch_docs]
    batch_boundaries = torch.tensor(
        [0] + [sum(batch_doc_lens[: i + 1]) for i in range(len(batch_docs))]
    )
    # HACK: I think this set up will let nnsight automatically batch queries
    # In the way it believes to be best though I need to check
    with torch.no_grad(), model.trace(batch_docs) as tracer:
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
        nnsight.save(batch_boundaries)

    return MoETrace.build(
        docs=batch_docs,
        indices_stack=indices_stack,
        weights_stack=weights_stack,
        doc_boundaries=batch_boundaries,
    )


# Process in batches
n_batches = (len(docs) + BATCH_SIZE - 1) // BATCH_SIZE
all_traces = []

for batch_idx in tqdm(range(n_batches)):
    start = batch_idx * BATCH_SIZE
    end = min(start + BATCH_SIZE, len(docs))
    batch_docs = docs[start:end]

    print(f"Processing batch {batch_idx + 1}/{n_batches} (docs {start}-{end - 1})...")
    trace = process_batch(model, batch_docs)
    all_traces.append(trace)
    print(f"  Tokens: {len(trace.token_ids)}")

# %% Results from first batch
first_trace = all_traces[0]
print(f"\nTotal batches: {len(all_traces)}")
print(f"Example from batch 0:")
print(f"Total tokens: {len(first_trace.token_ids)}")
print(
    f"Shape: [tokens={first_trace.token_ids.shape[0]}, "
    f"layers={first_trace.n_layers}, k={first_trace.k}]"
)

expert_ids, weights = first_trace.experts_for_token(0, 0)
print("\nLayer 0, Token 0:")
print(f"- Expert ids: {expert_ids}")
print(f"- Weights: {weights}")
