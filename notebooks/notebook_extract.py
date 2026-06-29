#!/usr/bin/env python

# %% Imports
import math

import h5py
import torch
from dotenv import load_dotenv
from nnsight import LanguageModel
from rich import print as rprint
from tqdm import tqdm

from moe_interp.capture import prepare_prompts_dataset, save_capture_artifacts
from moe_interp.capture.cache import append_to_file
from moe_interp.capture.model_adapter import get_model_adapter
from moe_interp.config import get_extractions_dir, set_seed
from moe_interp.io.data import load_dataset_prompts

# %% Configuration
seed = 1337
n_docs = 16
batch_size = 2

MODEL_NAME = "allenai/OLMoE-1B-7B-0924-Instruct"

load_dotenv()
set_seed(seed)

model = LanguageModel(
    MODEL_NAME,
    device_map="auto",
    dtype="auto",
    dispatch=True,
    # Captures tap the MoE block's boundary tensors (hidden_states, top_k_index, top_k_weights) once per forward
    # and reconstruct each expert's contribution from the expert weights.
)

print(model.dtype)  # Show dtype
tokenizer = model.tokenizer

adapter = get_model_adapter(model)
rprint(adapter)
n_layers = adapter.n_layers
n_experts = adapter.n_experts
d_model = adapter.d_model
norm_weight = model.model.norm.weight
norm_eps = model.model.norm.variance_epsilon

# %% Load prompts
DATASET_NAME = "pile10k"
prompts = load_dataset_prompts(DATASET_NAME, tokenizer, n_docs=n_docs)
print(f"Loaded {len(prompts)} {DATASET_NAME} prompts")

# %% Setup: per-expert storage (variable length, stored on disk)
output_dir = get_extractions_dir(MODEL_NAME, DATASET_NAME)
output_dir.mkdir(parents=True, exist_ok=True)

# %% Capture: batched with right-padding to preserve RoPE positional encodings
ds = prepare_prompts_dataset(prompts)

# Keep HDF5 files open for the full run (much faster than open/close per write)
layer_files = {
    i: h5py.File(output_dir / f"layer_{i:02d}.h5", "a") for i in range(n_layers)
}

# Right-pad so token positions are preserved (RoPE stays correct)
model.tokenizer.padding_side = "right"

# .iter() yields dicts where batch["input_ids"] is list[list[int]] —
# exactly the format nnsight's model.trace() expects.
n_batches = math.ceil(len(ds) / batch_size)
for batch in tqdm(
    ds.iter(batch_size=batch_size), total=n_batches, desc="Encoding batches"
):
    batch_tokens = batch["input_ids"]  # type: ignore[index]
    prompt_lengths = batch["length"]  # type: ignore[index]
    b_size = len(batch_tokens)

    # --- Pass 1: ONE trace, grab the 3 inputs OlmoeExperts.forward receives
    # and recompute every expert's contribution afterwards from the expert weights via
    # the model-specific adapter.reconstruct_expert_contributions. Works under grouped_mm / eager / batched_mm.
    expert_inputs = []
    with torch.no_grad(), model.trace(batch_tokens, remote=REMOTE):
        input_ids = model.inputs[1]["input_ids"].save()  # type: ignore[index]

        for layer in model.model.layers:
            expert_inputs.append(adapter.tap_layer(layer).save())

        pre_norm_hidden = model.model.norm.input.save()

    # --- Pass 2: reconstruct & write each expert's last-token contribution, per layer ---
    max_len = input_ids.shape[1]
    # Keep only each prompt's last real token (right-padding: row r's last token is at
    # r*max_len + length_r - 1).
    lengths = torch.as_tensor(prompt_lengths, dtype=torch.long)
    flat = torch.arange(b_size * max_len)
    keep_mask = flat == ((flat // max_len) * max_len + lengths[flat // max_len] - 1)
    second_moment = pre_norm_hidden.float().pow(2).mean(-1).reshape(-1)  # for RMSNorm

    for layer_idx, layer in enumerate(model.model.layers):
        hidden_states, top_k_index, top_k_weights = adapter.unpack_boundary(
            expert_inputs[layer_idx]
        )
        for expert_id, rows in adapter.reconstruct_expert_contributions(
            layer.mlp.experts,
            hidden_states,
            top_k_index,
            top_k_weights,
            real_mask=keep_mask,
            second_moment=second_moment,
            token_ids=input_ids.reshape(-1),
            norm_weight=norm_weight,
            norm_eps=norm_eps,
        ).items():
            append_to_file(
                layer_files[layer_idx],
                expert_id,
                *rows[:2],
                routing_weights=rows[2],
            )

# Set back tokenizer to pad to the left
model.tokenizer.padding_side = "left"

for f in layer_files.values():
    f.close()

save_capture_artifacts(
    model,
    MODEL_NAME,
    output_dir,
    {
        "model_name": MODEL_NAME,
        "dataset_name": DATASET_NAME,
        "n_docs": len(ds),
        "n_layers": n_layers,
        "n_experts": n_experts,
        "d_model": d_model,
    },
)
