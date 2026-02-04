#!/usr/bin/env python

# %% Imports
import torch
from nnsight import LanguageModel

from src.cache import LayerCache, TraceCache
from src.environment import print_model_config, set_seed

# %% Model Definition
set_seed(1337)

# NOTE: Ollmo (allenai/OLMoE-1B-7B-0924-Instruct) Model Spec:
# You can get it with somsething like this: config.num_experts
# - Layers                        : 16
# - Experts / layer               : 64
# - Active experts / token        : 8
# - Hidden size                   : 2048

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

                breakpoint()
                # For each token for each expert
                # NOTE: here I have to store the activations etc from this
                # outputs: Dict[int, torch.Tensor] = field(default_factory=dict)
                # FIX: I need to fix all of this
                expert_source = layer.mlp.experts.source
                # HACK: Something like this ?
                for token_idx in top_k_index:
                    # top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
                    top_k_pos, _ = expert_source.torch_where_0.output
                    # current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
                    expert_activation = expert_source.nn_functional_linear_1.output
                    layer_cache.expert_outputs[token_idx][top_k_pos] = expert_activation

                #                                       * def forward(
                #                                       0     self,
                #                                       1     hidden_states: torch.Tensor,
                #                                       2     top_k_index: torch.Tensor,
                #                                       3     top_k_weights: torch.Tensor,
                #                                       4 ) -> torch.Tensor:
                #  torch_zeros_like_0               ->  5     final_hidden_states = t
                # orch.zeros_like(hidden_states)
                #  torch_no_grad_0                  ->  6     with torch.no_grad():
                #  torch_nn_functional_one_hot_0    ->  7         expert_mask = torch
                # .nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
                #  expert_mask_permute_0            ->  8         expert_mask = expert_mask.permute(2, 1, 0)
                #  expert_mask_sum_0                ->  9         expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
                #  torch_greater_0                  ->  +         ...
                #  nonzero_0                        ->  +         ...
                #                                      10
                #                                      11     for expert_idx in expert_hit:
                #                                      12         expert_idx = expert_idx[0]
                #                                      13         if expert_idx == self.num_experts:
                #                                      14             continue
                #  torch_where_0                    -> 15         top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
                #                                      16         current_state = hidden_states[token_idx]
                #  nn_functional_linear_0           -> 17         gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
                #  chunk_0                          ->  +         ...
                #  self_act_fn_0                    -> 18         current_hidden_states = self.act_fn(gate) * up
                #  nn_functional_linear_1           -> 19         current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
                #                                      20         current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
                #  current_hidden_states_to_0       -> 21         final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
                #  final_hidden_states_index_add__0 ->  +         ...
                #                                      22
                #                                      23     return final_hidden_states


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
