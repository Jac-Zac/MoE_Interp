#!/usr/bin/env python

# %% Imports
import h5py
import torch
import torch.nn.functional as F
from nnsight import LanguageModel
from tqdm import trange

from src.cache import (
    _append_to_file,
    get_model_unembedding,
    save_metadata,
    save_unembedding,
)
from src.data import load_triviaqa
from src.environment import (
    get_data_dir,
    get_extractions_dir,
    get_unembedding_dir,
    load_env,
    set_seed,
)
from src.model_adapter import get_model_adapter


def apply_component_rmsnorm_like_hf(
    hidden_states: torch.Tensor,
    second_moment: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Apply RMSNorm to a component using residual-stream second moments.

    Math: component contribution at output is
    weight * (component / sqrt(E[residual^2] + eps)).

    Approximation: RMSNorm(sum_i c_i) ≠ sum_i c_i / sqrt(E[(sum_j c_j)^2]).
    We assume the denominator from the full residual stream applies linearly to
    each expert's gated contribution. This ignores cross-terms E[c_i · c_j] in
    the variance, which is reasonable when each expert adds a small delta to the
    residual (top-8 of 64). The alternative — recomputing variance per-component
    — isn't possible since we only capture isolated expert outputs, not the full
    residual at each intermediate point.
    """
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    hidden_states = hidden_states * torch.rsqrt(second_moment.unsqueeze(-1) + eps)
    return weight * hidden_states.to(input_dtype)


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
norm_weight = model.model.norm.weight
norm_eps = model.model.norm.variance_epsilon

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
    b_size = len(batch)

    with torch.no_grad(), model.trace(batch) as tracer:
        input_ids = model.inputs[1]["input_ids"].save().detach().cpu()

        layer_datas: list = []
        for layer_idx, layer in enumerate(model.model.layers):
            _, weights, indices = adapter.get_router_output(layer)
            top_k_weights = weights.save().detach().cpu()

            token_indices_list: list[torch.Tensor] = []
            down_projs_list: list[torch.Tensor] = []
            top_k_pos_list: list[torch.Tensor] = []

            expert_hit = adapter.get_expert_hit(layer)
            active_experts = (
                expert_hit[expert_hit != adapter.n_experts]
                .squeeze(-1)
                .save()
                .detach()
                .cpu()
            )
            num_iters = active_experts.numel()

            with tracer.iter[:num_iters]:
                top_k_pos, token_idx = adapter.get_top_k_pos_token_idx(layer)
                down_proj = adapter.get_expert_output(layer)

                token_indices_list.append(token_idx.save().detach().cpu())
                down_projs_list.append(down_proj.save().detach().cpu())
                top_k_pos_list.append(top_k_pos.save().detach().cpu())

            layer_datas.append(
                {
                    "active_experts": active_experts,
                    "token_indices": token_indices_list,
                    "down_projs": down_projs_list,
                    "top_k_pos": top_k_pos_list,
                    "weights": top_k_weights,
                }
            )

        pre_norm_hidden = model.model.norm.input[0].save().detach().cpu()
        max_len = input_ids.shape[1]

        # NOTE: Here we take the last token but averaging over content tokens can also be performed instead

        # Pre-compute last-token positions for all batches (vectorized)
        batch_offsets = torch.arange(b_size, device="cpu") * max_len
        actual_lens_tensor = torch.tensor(
            prompt_lengths, device="cpu", dtype=torch.long
        )
        last_positions = batch_offsets + actual_lens_tensor - 1
        sample_indices = torch.arange(b_size, device="cpu")
        pre_norm_last = pre_norm_hidden[sample_indices, actual_lens_tensor - 1]
        second_moment_last = pre_norm_last.float().pow(2).mean(dim=-1)

        for layer_idx, layer_data in enumerate(layer_datas):
            active_experts = layer_data["active_experts"]
            for i, expert_id in enumerate(active_experts.tolist()):
                token_idx = layer_data["token_indices"][i]
                down_proj = layer_data["down_projs"][i]
                top_k_pos = layer_data["top_k_pos"][i]

                # Vectorized: single mask
                is_last = torch.isin(token_idx, last_positions)

                if not is_last.any():
                    continue

                # Extract all last-token data at once
                last_down_proj = down_proj[is_last]
                last_top_k_pos = top_k_pos[is_last]
                last_token_idx_flat = token_idx[is_last]

                # Compute gate weights and weighted output
                gate_weights = layer_data["weights"][
                    last_token_idx_flat, last_top_k_pos
                ]
                gated_output = gate_weights.unsqueeze(-1) * last_down_proj
                batch_indices = (last_token_idx_flat // max_len).long()
                gated_output = apply_component_rmsnorm_like_hf(
                    hidden_states=gated_output,
                    second_moment=second_moment_last[batch_indices],
                    weight=norm_weight,
                    eps=norm_eps,
                )

                if gated_output.shape[0] == 0:
                    continue

                # Map flat indices back to get token IDs
                batch_indices = last_token_idx_flat // max_len
                pos_in_batch = last_token_idx_flat % max_len
                last_token_ids = input_ids[batch_indices, pos_in_batch]

                # Single write per expert (was batch_size writes)
                _append_to_file(
                    layer_files[layer_idx],
                    expert_id,
                    gated_output.half(),
                    last_token_ids,
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
dictionary = F.normalize(get_model_unembedding(model), dim=1)
save_unembedding(unembedding_dir / "dictionary.h5", dictionary)
print(f"Saved unembedding to {unembedding_dir}")
