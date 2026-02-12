#!/usr/bin/env python

# %% Imports
from dataclasses import dataclass

import nnsight
import torch
from nnsight import LanguageModel

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


# %% Simple dataclass to hold per-expert data (used only in this notebook)
@dataclass
class ExpertTrace:
    """Per-expert activation data for a specific layer."""

    token_indices: torch.Tensor  # [n_tokens] positions routed to this expert
    raw_outputs: torch.Tensor  # [n_tokens, hidden_dim] down-proj outputs
    top_k_positions: torch.Tensor  # [n_tokens] which of the k slots (0 to k-1)


# %% Process document with nnsight
with torch.no_grad(), model.trace([doc]) as tracer:
    layer_indices, layer_weights = [], []
    expert_traces: list[dict[int, ExpertTrace]] = []

    for layer in model.model.layers:
        # Get routing info from the gate
        # top_k_weights: weight for each expert
        # top_k_indices: expert id active for each token
        # self_gate_0 outputs: (_, top_k_weights, top_k_indices)
        _, weights, indices = layer.mlp.source.self_gate_0.output
        layer_indices.append(indices)
        layer_weights.append(weights)

        # Lists to store per-expert activations for this layer
        # Will be zipped together with active_experts after trace
        token_indices_list: list[torch.Tensor] = []
        down_projs_list: list[torch.Tensor] = []
        top_k_pos_list: list[torch.Tensor] = []

        # NOTE: One must be very carefull of what to get
        # I need to get expert_hit after the nonzero_0
        # expert_mask_sum_0  ->  9 expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        # torch_greater_0    ->  + ...
        # nonzero_0          ->  + ...
        expert_hit = layer.mlp.experts.source.nonzero_0.output
        num_experts_total = model.config.num_experts
        # Filter out padding expert (num_experts) and get actual expert IDs
        active_experts = expert_hit[expert_hit != num_experts_total].squeeze(-1)
        num_iters = active_experts.numel()

        # Iterate over active experts and capture their outputs with token mapping
        with tracer.iter[:num_iters]:
            # Capture token indices and top-k positions for this expert
            # torch_where_0 returns: (top_k_pos, token_idx)
            top_k_pos, token_idx = layer.mlp.experts.source.torch_where_0.output

            # Capture raw down-projection output (before weighting)
            down_proj = layer.mlp.experts.source.nn_functional_linear_1.output

            # Store in lists (iteration order matches active_experts order)
            token_indices_list.append(token_idx)
            top_k_pos_list.append(top_k_pos)
            down_projs_list.append(down_proj)

        # Build dict mapping expert_id -> ExpertTrace
        # active_experts gives us the expert IDs in iteration order
        layer_traces: dict[int, ExpertTrace] = {}

        for i, expert_id_tensor in enumerate(active_experts):
            expert_id = int(expert_id_tensor.item())
            layer_traces[expert_id] = ExpertTrace(
                token_indices=token_indices_list[i],
                raw_outputs=down_projs_list[i],
                top_k_positions=top_k_pos_list[i],
            )

        expert_traces.append(layer_traces)

    # Stack: [n_layers, seq, k]
    all_indices = torch.stack(layer_indices, dim=0)
    all_weights = torch.stack(layer_weights, dim=0)

    nnsight.save(all_indices)
    nnsight.save(all_weights)
    nnsight.save(expert_traces)


# %% Results from tracing
n_layers = len(expert_traces)
print(f"Document {doc_id}: {len(doc)} tokens, {n_layers} layers")
for layer_idx in range(n_layers):
    print(f"  Layer {layer_idx:2d}: {len(expert_traces[layer_idx]):2d} active experts")


# %% Compute gated outputs from captured trace
# Demonstrate: raw_outputs + expert_weights -> gated outputs -> per-doc mean
# This is the aggregation that the encode pipeline does automatically

layer_idx = 0
expert_id = sorted(expert_traces[layer_idx].keys())[0]
et = expert_traces[layer_idx][expert_id]

# Gated output = gate_weight * raw_output (what actually enters the residual stream)
gate_weights = all_weights[layer_idx, et.token_indices, et.top_k_positions]
gated_outputs = gate_weights.unsqueeze(-1) * et.raw_outputs  # [n_tokens, d_model]

# Per-document mean gated output (this is what SOMP operates on)
mean_gated = gated_outputs.mean(dim=0)  # [d_model]

print(f"\nLayer {layer_idx}, Expert {expert_id}:")
print(f"  Routed tokens: {et.token_indices.numel()}")
print(f"  Gated output shape: {tuple(gated_outputs.shape)}")
print(f"  Mean gated output shape: {tuple(mean_gated.shape)}")
print(f"  Mean gated norm: {mean_gated.norm():.4f}")


# %% Simple SOMP demo on this single document
import torch.nn.functional as F

from src.dictionary import extract_unembedding
from src.somp import somp

# Extract unembedding matrix (the SOMP dictionary)
unembed = extract_unembedding(model)  # [vocab_size, d_model]
dictionary = F.normalize(unembed, dim=-1)

# Build aggregated expert activations for all experts in one layer
n_experts = model.config.num_experts
d_model = model.config.hidden_size

# For a single document, the "activation matrix" is just one row per expert
# SOMP needs multiple samples to be meaningful, but we can still demo the mechanics
layer_acts = torch.zeros(n_experts, d_model)
for eid, et in expert_traces[layer_idx].items():
    gw = all_weights[layer_idx, et.token_indices, et.top_k_positions]
    gated = (gw.unsqueeze(-1) * et.raw_outputs).mean(dim=0)
    layer_acts[eid] = gated

# Pick one expert and run SOMP (single sample = not statistically meaningful,
# but demonstrates the API; use main.py encode + pursuit for real analysis)
X = layer_acts[expert_id].unsqueeze(0)  # [1, d_model]
result = somp(X, dictionary, k=10, center=False)  # no centering with 1 sample

# Decode the top atoms
print(f"\nSOMP decomposition for Layer {layer_idx}, Expert {expert_id}:")
print(f"  (NOTE: single-document demo, use main.py pursuit for real analysis)")
for i, atom_idx in enumerate(result["chosen"][:10].tolist()):
    token = model.tokenizer.decode([atom_idx])
    print(f"  {i + 1:2d}. token={repr(token):20s}  EVR={result['evr'][i]:.4f}")
