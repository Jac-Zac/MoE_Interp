#!/usr/bin/env python

# %% Imports
import h5py
import torch
import torch.nn.functional as F
from nnsight import LanguageModel
from tqdm import trange

from src.cache import _append_to_file, save_metadata, save_unembedding
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
batch_size = 1
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

# %% Capture: batched with right-padding to preserve RoPE positional encodings
# Sort by length so similar-length prompts land in the same batches
sorted_prompts = sorted(prompts, key=len, reverse=True)

# Keep HDF5 files open for the full run (much faster than open/close per write)
layer_files = {
    i: h5py.File(output_dir / f"layer_{i:02d}.h5", "a") for i in range(n_layers)
}

# Right-pad so token positions are preserved (RoPE stays correct)
model.tokenizer.padding_side = "right"

for batch_start in trange(0, len(sorted_prompts), batch_size):
    batch = sorted_prompts[batch_start : batch_start + batch_size]
    prompt_lengths = [len(p) for p in batch]
    bs = len(batch)

    with torch.no_grad(), model.trace(batch) as tracer:
        input_ids = model.inputs[1]["input_ids"].save()
        norm_layer = model.model.norm

        for layer_idx, layer in enumerate(model.model.layers):
            _, weights, indices = adapter.get_router_output(layer)
            top_k_weights = weights.save()

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
                token_indices_list.append(token_idx.save())
                down_projs_list.append(norm_layer(down_proj).save())
                top_k_pos_list.append(top_k_pos.save())

            layer_data = {
                "active_experts": active_experts,
                "token_indices": token_indices_list,
                "down_projs": down_projs_list,
                "top_k_pos": top_k_pos_list,
                "weights": top_k_weights,
            }

            max_len = input_ids.shape[1]

            # NOTE: Here we take the last token but averaging over content
            # tokens can also be performed instead
            for i, expert_id in enumerate(active_experts.tolist()):
                token_idx = layer_data["token_indices"][i]
                down_proj = layer_data["down_projs"][i]
                top_k_pos = layer_data["top_k_pos"][i]

                for b in range(bs):
                    actual_len = prompt_lengths[b]
                    # With right-padding, prompt b's last real token is at
                    # position b * max_len + actual_len - 1 in the flattened
                    # expert output tensor
                    prompt_offset = b * max_len
                    last_token_pos = prompt_offset + actual_len - 1

                    last_token_mask = token_idx == last_token_pos
                    if not last_token_mask.any():
                        continue

                    # Multi-GPU: ensure mask is on same device as target tensors
                    last_down_proj = down_proj[last_token_mask]
                    last_top_k_pos = top_k_pos[last_token_mask]

                    # Get gate weights and compute weighted output
                    # weights is [seq_len, top_k], indexed by [token_position, top_k_position]
                    gate_weights = top_k_weights[
                        last_token_pos, last_top_k_pos.to(top_k_weights.device)
                    ]
                    gated_output = (
                        gate_weights.unsqueeze(-1).to(last_down_proj.device)
                        * last_down_proj
                    )

                    if gated_output.shape[0] == 0:
                        continue

                    last_token_id = input_ids[b, actual_len - 1]

                    _append_to_file(
                        layer_files[layer_idx],
                        expert_id,
                        gated_output.half(),
                        last_token_id.unsqueeze(0).expand(gated_output.shape[0]),
                    )

# Set back tokenizer to padd to the left
model.tokenizer.padding_side = "left"

for f in layer_files.values():
    f.close()

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
