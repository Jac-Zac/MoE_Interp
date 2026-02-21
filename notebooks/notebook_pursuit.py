#!/usr/bin/env python

# %% Imports
from pathlib import Path

import h5py
import numpy as np
import plotly.express as px
import torch
import torch.nn.functional as F
from nnsight import LanguageModel
from tqdm import tqdm

from src.cache import ExpertActivationWriter, save_metadata
from src.data import load_triviaqa
from src.environment import get_device, set_seed

# %% Configuration
seed = 1337
set_seed(seed)
device = get_device()

model = LanguageModel(
    "allenai/OLMoE-1B-7B-0924-Instruct",
    device_map="auto",
    dtype=torch.float16,
    dispatch=True,
)
tokenizer = model.tokenizer

n_layers = model.config.num_hidden_layers
n_experts = model.config.num_experts
d_model = model.config.hidden_size

# %% Load TriviaQA prompts
n_docs = 16
batch_size = 8
prompts = load_triviaqa(tokenizer, n_docs=n_docs)
print(f"Loaded {len(prompts)} TriviaQA prompts")

# %% Setup: create HDF5 writers (memory-mapped, buffered)
output_dir = Path("data/notebook_activations")
output_dir.mkdir(parents=True, exist_ok=True)

writers = [
    ExpertActivationWriter(
        output_dir / f"layer_{layer_idx:02d}.h5", n_experts, d_model, dtype=np.float16
    )
    for layer_idx in range(n_layers)
]

# %% Batched capture: process each batch, write immediately to disk
total_docs = 0
for start in tqdm(range(0, len(prompts), batch_size), desc="Capturing batches"):
    batch = prompts[start : start + batch_size]
    seq_len = len(batch[0])
    batch_n_docs = len(batch)
    total_docs += batch_n_docs
    batch_data = {}

    with torch.no_grad(), model.trace(batch) as tracer:
        for layer_idx, layer in enumerate(model.model.layers):
            # Get routing info from the gate
            # top_k_weights: weight for each expert
            # top_k_indices: expert id active for each token
            # self_gate_0 outputs: (_, top_k_weights, top_k_indices)
            _, weights, indices = layer.mlp.source.self_gate_0.output
            top_k_weights = weights.save()
            top_k_indices = indices.save()

            # Lists to store per-expert activations for this layer
            token_indices_list: list[torch.Tensor] = []
            down_projs_list: list[torch.Tensor] = []
            top_k_pos_list: list[torch.Tensor] = []

            # NOTE: One must be very careful of what to get
            # I need to get expert_hit after the nonzero_0
            # expert_mask_sum_0  ->  9 expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            # torch_greater_0    ->  + ...
            # nonzero_0          ->  + ...
            expert_hit = layer.mlp.experts.source.nonzero_0.output
            active_experts = (
                expert_hit[expert_hit != model.config.num_experts].squeeze(-1).save()
            )
            num_iters = active_experts.numel()

            with tracer.iter[:num_iters]:
                top_k_pos, token_idx = layer.mlp.experts.source.torch_where_0.output
                down_proj = layer.mlp.experts.source.nn_functional_linear_1.output
                token_indices_list.append(token_idx.save())
                down_projs_list.append(down_proj.save())
                top_k_pos_list.append(top_k_pos.save())

            # Store for post-processing in this batch
            batch_data[layer_idx] = {
                "active_experts": active_experts,
                "token_indices": token_indices_list,
                "down_projs": down_projs_list,
                "top_k_pos": top_k_pos_list,
                "weights": top_k_weights,
            }

    # Process this batch: filter to last token, write to HDF5 immediately
    for layer_idx in range(n_layers):
        d = batch_data[layer_idx]
        active_experts = d["active_experts"]
        token_indices = d["token_indices"]
        down_projs = d["down_projs"]
        top_k_positions = d["top_k_pos"]
        weights = d["weights"]

        for i, expert_id in enumerate(active_experts.tolist()):
            token_idx = token_indices[i]
            down_proj = down_projs[i]
            top_k_pos = top_k_positions[i]

            # Filter to last token only
            last_token_mask = (token_idx % seq_len) == (seq_len - 1)
            if not last_token_mask.any():
                continue

            last_token_idx = token_idx[last_token_mask]
            last_down_proj = down_proj[last_token_mask]
            last_top_k_pos = top_k_pos[last_token_mask]

            # Get gate weights and compute weighted output
            # weights is [seq_len, top_k], indexed by [token_position, top_k_position]
            gate_weights = weights[last_token_idx, last_top_k_pos]
            gated_output = gate_weights.unsqueeze(-1) * last_down_proj

            # Write immediately to HDF5 (no memory accumulation)
            for j in range(gated_output.shape[0]):
                writers[layer_idx].add(expert_id, gated_output[j].half())

# %% Close all writers
for w in writers:
    w.close()

save_metadata(output_dir, total_docs, n_layers, n_experts, d_model, dtype="float16")
print(f"Saved activations to {output_dir}")


# %% Simple projection-based expert pursuit
def projection_pursuit(
    X: torch.Tensor,
    dictionary: torch.Tensor,
    tokenizer,
    k: int = 50,
) -> tuple[list[str], list[float]]:
    """Project expert activations onto dictionary, return top-k tokens by EVR.

    EVR per token = var(projection) / total_var, clamped to [0,1].
    Note: Dictionary vectors are non-orthogonal, so sum(EVR) may exceed 1.
    """
    if X.shape[0] <= 1:
        return [], []

    X_centered = X - X.mean(dim=0, keepdim=True)
    projections = X_centered @ dictionary.T

    total_var = X_centered.var(dim=0).sum()
    if total_var < 1e-10:
        return [], []

    var_per_token = projections.var(dim=0)
    evr = (var_per_token / total_var).clamp(0, 1)

    valid_mask = evr > 1e-6
    if not valid_mask.any():
        return [], []

    top_k = evr.topk(min(k, int(valid_mask.sum().item())))

    tokens = [tokenizer.decode([i.item()]).strip() for i in top_k.indices]
    return tokens, top_k.values.tolist()


def load_expert_data(path, expert_id):
    """Load activations for a single expert from HDF5."""
    with h5py.File(path, "r") as f:
        data = f[f"expert_{expert_id:03d}"][:]
    return torch.from_numpy(data).float()


# Load dictionary (full unembedding, L2-normalized)
dictionary = F.normalize(model.lm_head.weight.detach().float(), dim=1).cpu()

# %% Run on all experts
min_activations = 5
results = []
for li in tqdm(range(n_layers), desc="Projection pursuit"):
    layer_path = output_dir / f"layer_{li:02d}.h5"
    for ei in range(n_experts):
        X = load_expert_data(layer_path, ei)
        if X.shape[0] < min_activations:
            continue
        tokens, evr = projection_pursuit(X, dictionary, tokenizer, k=50)
        if not tokens:
            continue
        results.append(
            {
                "layer": li,
                "expert": ei,
                "n_activations": X.shape[0],
                "tokens": tokens,
                "evr": evr,
            }
        )

print(f"Analyzed {len(results)} experts")

# %% Display sample results
for r in results[:5]:
    print(f"\nLayer {r['layer']}, Expert {r['expert']}:")
    for t, e in zip(r["tokens"][:10], r["evr"][:10]):
        print(f"  {t}: {e:.4f}")

# %% Plot EVR heatmap per expert
evr_matrix = np.zeros((n_layers, n_experts))
for r in results:
    evr_matrix[r["layer"], r["expert"]] = r["evr"][0] if r["evr"] else 0.0

fig = px.imshow(
    evr_matrix,
    x=[f"E{i}" for i in range(n_experts)],
    y=[f"L{i}" for i in range(n_layers)],
    color_continuous_scale="Blues",
    labels=dict(x="Expert", y="Layer", color="Top EVR"),
    title="Expert Pursuit: Top Explained Variance Ratio per Expert",
)
fig.update_layout(width=1600, height=600)
fig.show()
