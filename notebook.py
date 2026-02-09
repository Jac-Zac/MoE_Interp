#!/usr/bin/env python

# %% Imports
import nnsight
import torch
from nnsight import LanguageModel

from src.cache import DocumentTrace, ExpertTrace
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
    indices = torch.stack(layer_indices, dim=0)
    weights = torch.stack(layer_weights, dim=0)

    # Create DocumentTrace (in memory only, not saved)
    trace = DocumentTrace(
        doc_id=doc_id,
        n_layers=len(layer_indices),
        expert_indices=indices,
        expert_weights=weights,
        expert_traces=expert_traces,
    )

    nnsight.save(indices)
    nnsight.save(weights)
    nnsight.save(expert_traces)


# %% Results from processing
print(f"DocumentTrace: {trace}")
print(f"Number of layers: {len(trace.expert_traces)}")
print(f"Active experts in layer 0: {len(trace.expert_traces[0])}")
