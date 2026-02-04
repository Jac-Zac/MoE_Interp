#!/usr/bin/env python

# %% Imports
from collections import defaultdict

import nnsight
import torch
from nnsight import LanguageModel

from src.environment import set_seed

# %% Model Definition
set_seed(1337)

# NOTE:
# Ollmo (allenai/OLMoE-1B-7B-0924-Instruct) Model Spec:
# You can get it with somsething like this: config.num_experts
# - 64 total experts
# - 8 active experts
# .. n_pars ...
# Other specs

model = LanguageModel(
    "allenai/OLMoE-1B-7B-0924-Instruct",
    device_map="auto",
    # NOTE: bfloat would be better for CUDA though
    # Use float16 for mps compatibility
    dtype=torch.float16,
    # dispatch=False,
    dispatch=True,
)

# Get the underlying Transformers config
config = model.model.config
print(f"Total experts per layer: {config.num_experts}")  # Outputs: 64
print(f"Layers: {config.num_hidden_layers}")  # Outputs: 16
print(f"Active experts per token: {config.num_experts_per_tok}")  # Outputs: 8

# %% Prompts
prompts = ["The capital of France is", "The capital of Italy is"]
print(model)

# %% Running the inference
print(f"\nProcessing {len(prompts)} prompts...")

with model.trace() as tracer:
    cache = defaultdict(list)

    # TODO: multiple layers
    # Note then I'm going to do something like this
    # cache[layer_idx]

    # for prompt in prompts:
    with tracer.invoke(prompts[0]):
        # For each layer
        layer = model.model.layers[0]

        # NOTE: This is the last elment of this code
        # self_gate_0 -> _, top_k_weights, top_k_index = self.gate(hidden_state...)
        # So I can take top_k_index with it the following

        # It is split as follow for each dim:
        # - 0: a row with everything for a specific token
        # - 1: it is always 8 the number of active experts
        # -> Each number corresopndes to one specifc expert

        _, top_k_weights, top_k_index = layer.mlp.source.self_gate_0.output[-1]
        cache["top_k_weights"].append(top_k_weights)  # [seq_len, active_expert]
        cache["top_k_index"].append(top_k_index)  # [seq_len, active_expert]

        # For each token
        # NOTE: here I have to store the activations etc from this

        #                                      for expert_idx in expert_hit:
        #                                          expert_idx = expert_idx[0]
        #                                          if expert_idx == self.num_experts:
        #                                              continue
        #  torch_where_0                    ->     top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        #                                          current_state = hidden_states[token_idx]
        #  nn_functional_linear_0           ->     gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx
        # ]).chunk(2, dim=-1)
        #  chunk_0                          ->     ...
        #  self_act_fn_0                    ->     current_hidden_states = self.act_fn(gate) * up
        #  nn_functional_linear_1           ->     current_hidden_states = nn.functional.linear(current_hidden_states, self.do
        # wn_proj[expert_idx])
        #                                          current_hidden_states = current_hidden_states * top_k_weights[token_idx, to
        # p_k_pos, None]
        #  current_hidden_states_to_0       ->     final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final
        # _hidden_states.dtype))
        #  final_hidden_states_index_add__0 ->     ...

        # Storing actual activation for each expert

    nnsight.save(cache)

# print(output)

# %% Analyses with those activations
# Let's do something with those now
