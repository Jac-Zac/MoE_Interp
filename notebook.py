#!/usr/bin/env python

# %% Imports
import torch
from nnsight import LanguageModel

from src.cache import ExpertCache, LayerCache, TraceCache
from src.environment import print_model_config, set_seed

# %% Model Definition
set_seed(1337)

# NOTE: Ollmo (allenai/OLMoE-1B-7B-0924-Instruct) Model Spec:
# You can get it with somsething like this: config.num_experts
# - 64 total experts
# - 8 active experts
# .. n_pars ...
# Other specs

# NOTE: bfloat would be better for CUDA though
# Use float16 for mps compatibility
model = LanguageModel(
    "allenai/OLMoE-1B-7B-0924-Instruct",
    device_map="auto",
    dtype=torch.float16,
    # dispatch=True,
    dispatch=False,
)

# %% Print model config in table format
config = model.model.config
print_model_config(config)

# %% Prompts
prompts = ["The capital of France is", "The capital of Italy is"]
print(model)

# %% Running the inference
print(f"\nProcessing {len(prompts)} prompts...")

# Initialize trace cache
cache = TraceCache(prompts=prompts)

with model.trace() as tracer:
    # Iterate over each prompt
    for p_idx, prompt in enumerate(prompts):
        # HACK: To check if this is actually batched with nnisght
        # Start tracing for each prompt in a batched way I belived
        with tracer.invoke(prompt):
            for l_idx, layer in enumerate(model.model.layers):
                # Get or create layer cache
                if l_idx not in cache.layers:
                    cache.layers[l_idx] = LayerCache()

                layer_cache = cache.layers[l_idx]

                # NOTE: This is the last elment of this code
                # self_gate_0 -> _, top_k_weights, top_k_index = self.gate(hidden_state...)
                # So I can take the following things out in a similar way
                _, top_k_weights, top_k_index = layer.mlp.source.self_gate_0.output

                # It is split as follow for each dim:
                # - 0: a row with everything for a specific token
                # - 1: it is always 8 the number of active experts
                # -> Each number corresopndes to one specifc expert

                # [seq_len, active_expert] for both of them
                # TODO: Look more into this
                layer_cache.topk_weights.append(top_k_weights.save())
                layer_cache.topk_indices.append(top_k_index.save())

                # For each token
                # NOTE: here I have to store the activations etc from this

                #                                      for expert_idx in expert_hit:
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
                # TODO: Add expert activation capture using torch_where_0 and nn_functional_linear_1


# %% Print results
for l_idx, layer_cache in cache.layers.items():
    print(f"Layer {l_idx}:")

    for p_idx in range(len(prompts)):
        weights = layer_cache.topk_weights[p_idx]
        indices = layer_cache.topk_indices[p_idx]
        print(f"  Prompt {p_idx}: weights {weights.shape}, indices {indices.shape}")

    break

# %% Analyses with those activations
# Let's do something with those now
