#!/usr/bin/env python

# %% Imports
import torch
import torch.nn.functional as F
from nnsight import LanguageModel
from tqdm import tqdm

from src.cache import append_expert_h5, save_metadata, save_unembedding
from src.data import load_triviaqa
from src.environment import (
    get_data_dir,
    get_extractions_dir,
    get_unembedding_dir,
    load_env,
    set_seed,
)
from src.model_adapter import get_model_adapter

# %% Configuration
seed = 1337
n_docs = 16
# MODEL_NAME = "openai/gpt-oss-20b"  # Change this to run different models
MODEL_NAME = "allenai/OLMoE-1B-7B-0924-Instruct"  # Change this to run different models

load_env()
set_seed(seed)
data_dir = get_data_dir()
model = LanguageModel(
    MODEL_NAME,
    device_map="auto",
    dtype="auto",  # automatically dispatch bfloat16 usually
    dispatch=True,
)

print(model.dtype)  # Show dtype
tokenizer = model.tokenizer

adapter = get_model_adapter(model=model)
print(repr(adapter))
n_layers = adapter.n_layers
n_experts = adapter.n_experts
d_model = adapter.d_model

# %% Load TriviaQA prompts
prompts = load_triviaqa(tokenizer, n_docs=n_docs)
print(f"Loaded {len(prompts)} TriviaQA prompts")

# %% Setup: per-expert storage (variable length, stored on disk)
output_dir = get_extractions_dir(MODEL_NAME)
output_dir.mkdir(parents=True, exist_ok=True)

# %% Capture: process one prompt at a time, collect per-expert activations
# NOTE: Batching is intentionally avoided. nnsight left-pads shorter sequences
# to match the longest in the batch, which shifts positional embeddings for all
# padded documents. Because OLMoE uses RoPE, positional encoding directly affects
# attention and expert routing, so a token that sits at position 5 in a standalone
# forward pass would appear at a different position inside a padded batch, producing
# different activations. Processing one prompt at a time guarantees no padding is ever introduced.
for prompt in tqdm(prompts, desc="Capturing prompts"):
    prompt_data = {}

    with torch.no_grad(), model.trace(prompt) as tracer:
        # Single prompt: no padding, last real token is always at seq_len - 1
        input_ids = model.inputs[1]["input_ids"].save()

        for layer_idx, layer in enumerate(model.model.layers):
            # Get routing info from the gate
            # top_k_weights: weight for each expert
            # top_k_indices: expert id active for each token
            # self_gate_0 outputs: (_, top_k_weights, top_k_indices)
            _, weights, indices = adapter.get_router_output(layer)

            top_k_weights = weights.save()
            # Lists to store per-expert activations for this layer
            token_indices_list: list[torch.Tensor] = []
            down_projs_list: list[torch.Tensor] = []
            top_k_pos_list: list[torch.Tensor] = []

            expert_hit = adapter.get_expert_hit(layer)
            active_experts = (
                expert_hit[expert_hit != adapter.n_experts].squeeze(-1).save()
            )
            num_iters = active_experts.numel()

            with tracer.iter[:num_iters]:
                top_k_pos, token_idx = adapter.get_top_k_pos_token_idx(layer)
                down_proj = adapter.get_expert_output(layer)

                # NOTE: Similarly to logit lens we apply the last normalization to the expert activations here
                down_proj = model.model.norm(down_proj)

                token_indices_list.append(token_idx.save())
                down_projs_list.append(down_proj.save())
                top_k_pos_list.append(top_k_pos.save())

            # Store for post-processing after the trace
            prompt_data[layer_idx] = {
                "active_experts": active_experts,
                "token_indices": token_indices_list,
                "down_projs": down_projs_list,
                "top_k_pos": top_k_pos_list,
                "weights": top_k_weights,
            }

    # Post-process: filter to last token only, write per-expert to disk
    seq_len = input_ids.shape[1]
    last_token_id = input_ids[0, -1]

    for layer_idx in range(n_layers):
        d = prompt_data[layer_idx]
        active_experts = d["active_experts"]
        token_indices = d["token_indices"]
        down_projs = d["down_projs"]
        top_k_positions = d["top_k_pos"]
        weights = d["weights"]

        for i, expert_id in enumerate(active_experts.tolist()):
            token_idx = token_indices[i]
            down_proj = down_projs[i]
            top_k_pos = top_k_positions[i]

            # Single prompt: last token is always at seq_len - 1
            last_token_mask = token_idx == (seq_len - 1)
            if not last_token_mask.any():
                continue

            last_down_proj = down_proj[last_token_mask]
            last_top_k_pos = top_k_pos[last_token_mask]

            # Get gate weights and compute weighted output
            # weights is [seq_len, top_k], indexed by [token_position, top_k_position]
            gate_weights = weights[seq_len - 1, last_top_k_pos]
            gated_output = gate_weights.unsqueeze(-1) * last_down_proj

            if gated_output.shape[0] == 0:
                continue
            layer_path = output_dir / f"layer_{layer_idx:02d}.h5"
            append_expert_h5(
                layer_path,
                expert_id,
                gated_output.half(),
                last_token_id.unsqueeze(0).expand(gated_output.shape[0]),
                overwrite=True,
            )

save_metadata(
    output_dir,
    model_name=MODEL_NAME,
    n_docs=len(prompts),
    n_layers=n_layers,
    n_experts=n_experts,
    d_model=d_model,
)
print(f"Saved activations to {output_dir}")

# %% Save unembedding dictionary
unembedding_dir = get_unembedding_dir(MODEL_NAME)
dictionary = F.normalize(model.lm_head.weight.detach().float(), dim=1).cpu()
save_unembedding(unembedding_dir / "dictionary.h5", dictionary)
print(f"Saved unembedding to {unembedding_dir}")
