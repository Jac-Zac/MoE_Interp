#!/usr/bin/env python

# %% Imports
import h5py
import torch
import torch.nn.functional as F
from nnsight import LanguageModel
from rich import print as rprint
from tqdm import tqdm

from moe_interp.capture.cache import (
    _append_to_file,
    get_model_unembedding,
    save_metadata,
    save_unembedding,
)
from moe_interp.capture.capture import apply_component_rmsnorm
from moe_interp.capture.model_adapter import get_model_adapter
from moe_interp.config import (
    get_data_dir,
    get_extractions_dir,
    get_unembedding_dir,
    load_env,
    set_seed,
)
from moe_interp.io.data import load_dataset_prompts

# %% Configuration
seed = 1337
n_docs = 16
batch_size = 1

# NOTE: gpt-oss doesn't fit in a V100 but current code support pipeline parallelism by default
# Thus the model will be shared between two gpus.
# MODEL_NAME = "openai/gpt-oss-20b"  # Change this to run different models
MODEL_NAME = "allenai/OLMoE-1B-7B-0924-Instruct"  # Change this to run different models
REMOTE = False
# REMOTE = True

load_env()
set_seed(seed)
data_dir = get_data_dir()
# The main.py CLI automatically detects distributed setup and uses tp_plan="auto"
# TODO: Change this code if on remote device
model = LanguageModel(
    MODEL_NAME,
    # NOTE: Support different things
    device_map="auto",
    # # automatically dispatch bfloat16 usually
    # # cast to bfloat16 gpt-oss on V100 because of unsupported default dtype
    dtype="auto",
    dispatch=True,
)

print(model.dtype)  # Show dtype
tokenizer = model.tokenizer

adapter = get_model_adapter(model=model)
rprint(adapter)
n_layers = adapter.n_layers
n_experts = adapter.n_experts
d_model = adapter.d_model
norm_weight = model.model.norm.weight
norm_eps = model.model.norm.variance_epsilon

# %% Load prompts
DATASET_NAME = "triviaqa"
prompts = load_dataset_prompts(DATASET_NAME, tokenizer, n_docs=n_docs)
print(f"Loaded {len(prompts)} {DATASET_NAME} prompts")

# %% Setup: per-expert storage (variable length, stored on disk)
output_dir = get_extractions_dir(MODEL_NAME, DATASET_NAME)
output_dir.mkdir(parents=True, exist_ok=True)

# %% Capture: batched with right-padding to preserve RoPE positional encodings
# Pre-compute prompt lengths and sort so similar-length prompts land together
# (minimises right-padding waste per batch, preserving RoPE positions).
ds = prompts.map(lambda x: {"length": len(x["input_ids"])})  # type: ignore[index]
ds = ds.sort("length", reverse=True)

# Keep HDF5 files open for the full run (much faster than open/close per write)
layer_files = {
    i: h5py.File(output_dir / f"layer_{i:02d}.h5", "a") for i in range(n_layers)
}

# Right-pad so token positions are preserved (RoPE stays correct)
model.tokenizer.padding_side = "right"

# .iter() yields dicts where batch["input_ids"] is list[list[int]] —
# exactly the format nnsight's model.trace() expects.
n_batches = len(ds) // batch_size
for batch in tqdm(
    ds.iter(batch_size=batch_size), total=n_batches, desc="Encoding batches"
):
    batch_tokens = batch["input_ids"]  # type: ignore[index]
    prompt_lengths = batch["length"]  # type: ignore[index]
    b_size = len(batch_tokens)
    pending_writes: dict[tuple[int, int], list[tuple[torch.Tensor, torch.Tensor]]] = {}

    with torch.no_grad(), model.trace(batch_tokens, remote=REMOTE) as tracer:
        input_ids = model.inputs[1]["input_ids"].save().detach()  # type: ignore[index]

        layer_datas: list = []
        for layer_idx, layer in enumerate(model.model.layers):
            _, weights, indices = adapter.get_router_output(layer)
            top_k_weights = weights.save().detach()

            token_indices_list: list[torch.Tensor] = []
            down_projs_list: list[torch.Tensor] = []
            top_k_pos_list: list[torch.Tensor] = []

            expert_hit = adapter.get_expert_hit(layer)
            active_experts = (
                expert_hit[expert_hit != adapter.n_experts].squeeze(-1).save().detach()
            )
            num_iters = active_experts.numel()

            with tracer.iter[:num_iters]:
                top_k_pos, token_idx = adapter.get_top_k_pos_token_idx(layer)
                down_proj = adapter.get_expert_output(layer)

                token_indices_list.append(token_idx.save().detach())
                down_projs_list.append(down_proj.save().detach())
                top_k_pos_list.append(top_k_pos.save().detach())

            layer_datas.append(
                {
                    "active_experts": active_experts,
                    "token_indices": token_indices_list,
                    "down_projs": down_projs_list,
                    "top_k_pos": top_k_pos_list,
                    "weights": top_k_weights,
                }
            )

        pre_norm_hidden = model.model.norm.input.save().detach()
        max_len = input_ids.shape[1]

        # NOTE: Here we take the last token but averaging over content tokens can also be performed instead

        # Pre-compute last-token positions for all batches (vectorized)
        batch_offsets = torch.arange(b_size) * max_len
        actual_lens_tensor = torch.tensor(prompt_lengths, dtype=torch.long)
        last_positions = batch_offsets + actual_lens_tensor - 1
        sample_indices = torch.arange(b_size)
        pre_norm_last = pre_norm_hidden[sample_indices, actual_lens_tensor - 1]
        second_moment_last = torch.atleast_1d(pre_norm_last.float().pow(2).mean(dim=-1))

        for layer_idx, layer_data in enumerate(layer_datas):
            active_experts = layer_data["active_experts"]
            for i in range(len(layer_data["token_indices"])):
                token_idx = layer_data["token_indices"][i]
                down_proj = layer_data["down_projs"][i]
                top_k_pos = layer_data["top_k_pos"][i]
                expert_id = active_experts[i].item()

                target_device = down_proj.device
                lp = last_positions.to(target_device)
                ids = input_ids.to(target_device)
                tw = layer_data["weights"].to(target_device)
                sm_last = second_moment_last.to(target_device)

                # Vectorized: single mask
                is_last = torch.isin(token_idx, lp)

                if not is_last.any():
                    continue

                # Extract all last-token data at once
                last_down_proj = down_proj[is_last]
                last_top_k_pos = top_k_pos[is_last]
                last_token_idx_flat = token_idx[is_last]

                # Compute gate weights and weighted output
                gate_weights = tw[last_token_idx_flat, last_top_k_pos]
                gated_output = gate_weights.unsqueeze(-1) * last_down_proj
                batch_indices = (last_token_idx_flat // max_len).long()
                gated_output = apply_component_rmsnorm(
                    hidden_states=gated_output,
                    second_moment=sm_last[batch_indices],
                    weight=norm_weight.to(target_device),
                    eps=norm_eps,
                )

                if gated_output.shape[0] == 0:
                    continue

                # Map flat indices back to get token IDs
                batch_indices = last_token_idx_flat // max_len
                pos_in_batch = last_token_idx_flat % max_len
                last_token_ids = ids[batch_indices, pos_in_batch]

                # Single write per expert (was batch_size writes)
                key = (layer_idx, expert_id)
                pending_writes.setdefault(key, []).append(
                    (gated_output.half().cpu(), last_token_ids.cpu())
                )

    for (layer_idx, expert_id), writes in pending_writes.items():
        activations = torch.cat([activations for activations, _ in writes], dim=0)
        tokens = torch.cat([tokens for _, tokens in writes], dim=0)
        _append_to_file(layer_files[layer_idx], expert_id, activations, tokens)

# Set back tokenizer to padd to the left
model.tokenizer.padding_side = "left"

for f in layer_files.values():
    f.close()

save_metadata(
    output_dir,
    model_name=MODEL_NAME,
    n_docs=len(ds),
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
