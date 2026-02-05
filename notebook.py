#!/usr/bin/env python

# %% Imports
import torch
from nnsight import LanguageModel

from src.cache import MoETrace
from src.environment import print_model_config, set_seed

# %% Model Definition
set_seed(1337)

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

# %% Print model config
config = model.model.config
print_model_config(config)

# %% Prompts
prompts = ["The capital of France is", "The capital of Italy is"]
print(f"\nProcessing {len(prompts)} prompts...")

# %% Capture expert activations
batch_size = len(prompts)
n_layers = config.num_hidden_layers
num_experts = config.num_experts
top_k = config.num_experts_per_tok

# TODO: Pehrpas this can be also done with torch in a more efficent way
# Storage for captured tensors
all_topk_indices = []  # Will be [n_layers, batch, seq_len, top_k]
all_topk_weights = []  # Will be [n_layers, batch, seq_len, top_k]

# Storage for expert outputs: key = (layer_idx, expert_id)
# Value = tensor of shape [n_tokens, hidden_dim]
expert_outputs: dict = {}

with torch.no_grad():
    with model.trace(prompts) as tracer:
        # nnsight handles batching automatically - we get [batch * seq_len, ...]
        for layer_idx, layer in enumerate(model.model.layers):
            # Get routing info from the gate
            # top_k_weights: weight for each expert
            # top_k_indices: expert id active for each token

            # self_gate_0 outputs: (_, top_k_weights, top_k_indices)
            _, top_k_weights, top_k_indices = layer.mlp.source.self_gate_0.output

            # Save routing for this layer
            # top_k_indices: [batch, seq_len, top_k]
            # top_k_weights: [batch, seq_len, top_k]
            all_topk_indices.append(top_k_indices.save())
            all_topk_weights.append(top_k_weights.save())

            # Access the experts computation
            expert_src = layer.mlp.experts.source

            # expert_mask_sum_0 tells us which experts activated in this forward pass
            # [num_experts] boolean tensor
            expert_mask = expert_src.expert_mask_sum_0.output

            # For each expert that activated, capture its output
            # We use the pre-saved indices to determine which experts to track

            # Get the computed expert outputs after the mlp loop
            # nn_functional_linear_1 is the final linear projection per expert

            # This is a bit tricky with nnsight
            # The expert_mask tells us which experts ran
            # expert_hit_indices = expert_mask.nonzero().save()

# %% Post-process: Build MoETrace from saved tensors

# Stack layer-wise tensors: [n_layers, batch, seq_len, top_k]

# [batch * seq_len, n_layers, top_k]
indices_tensor = torch.stack(all_topk_indices, dim=-1)
# [batch * seq_len, n_layers, top_k]
weights_tensor = torch.stack(all_topk_weights, dim=-1)

# Create the trace object
trace = MoETrace(
    prompts=prompts,
    expert_indices=indices_tensor,
    expert_weights=weights_tensor,
    # Currently empty - needs expert output capture
    expert_outputs=expert_outputs,
)


# %% Printing saved things
print(trace.expert_indices.shape)
print(trace.expert_weights.shape)
# print(trace.expert_indices[0, 0, 0])
# print(trace.expert_outputs)
