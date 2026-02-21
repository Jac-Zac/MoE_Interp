#!/usr/bin/env python

# %% Imports
from collections import Counter
from dataclasses import dataclass

import nnsight
import plotly.express as px
import torch
from datasets import load_dataset
from nnsight import LanguageModel
from residual.sparse_decomposition import SOMP
from tqdm import tqdm

from src.environment import get_device, set_seed
from src.pursuit import build_filtered_dictionary

# %% Configuration
seed = 1337
set_seed(seed)
device = get_device()

# OLMoE: 16 layers, 64 experts/layer, top-8 routing, d_model=2048
model = LanguageModel(
    "allenai/OLMoE-1B-7B-0924-Instruct",
    device_map="auto",
    dtype=torch.float16,
    dispatch=True,
)


# %% Simple dataclass to hold per-expert data (used in tracing block below)
@dataclass
class ExpertTrace:
    """Per-expert activation data for a specific layer."""

    token_indices: torch.Tensor  # [n_tokens] positions routed to this expert
    raw_outputs: torch.Tensor  # [n_tokens, hidden_dim] down-proj outputs
    top_k_positions: torch.Tensor  # [n_tokens] which of the k slots (0 to k-1)


# %% Load TriviaQA question (Single Document)
dataset = load_dataset("mandarjoshi/trivia_qa", "rc", split="train")

question_text = dataset[0]["question"]
messages = [{"role": "user", "content": question_text}]

# Tokenize and ensure we extract a flat list of token IDs
doc = model.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
if hasattr(doc, "input_ids"):
    doc_ids = getattr(doc, "input_ids")
    doc = doc_ids[0] if isinstance(doc_ids[0], list) else doc_ids
elif not isinstance(doc, list):
    doc = list(doc)

print(f"Loaded TriviaQA question, {len(doc)} tokens")

# %% Process document with nnsight
with torch.no_grad(), model.trace([doc]) as tracer:
    layer_indices, layer_weights = [], []
    expert_traces: list[dict[int, ExpertTrace]] = []

    for layer in model.model.layers:
        # Get routing info from the gate
        # top_k_weights: weight for each expert
        # top_k_indices: expert id active for each token
        # self_gate_0 outputs: (_, top_k_weights, top_k_indices)
        _, weights, indices = layer.mlp.source.self_gate_0.output
        layer_indices.append(indices)
        layer_weights.append(weights)

        # Lists to store per-expert activations for this layer
        # Will be zipped together with active_experts after trace
        token_indices_list: list[torch.Tensor] = []
        down_projs_list: list[torch.Tensor] = []
        top_k_pos_list: list[torch.Tensor] = []

        # NOTE: One must be very carefull of what to get
        # I need to get expert_hit after the nonzero_0
        # expert_mask_sum_0  ->  9 expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        # torch_greater_0    ->  + ...
        # nonzero_0          ->  + ...
        expert_hit = layer.mlp.experts.source.nonzero_0.output
        num_experts_total = model.config.num_experts
        # Filter out padding expert (num_experts) and get actual expert IDs
        active_experts = expert_hit[expert_hit != num_experts_total].squeeze(-1)
        num_iters = active_experts.numel()

        # Iterate over active experts and capture their outputs with token mapping
        with tracer.iter[:num_iters]:
            # Capture token indices and top-k positions for this expert
            # torch_where_0 returns: (top_k_pos, token_idx)
            top_k_pos, token_idx = layer.mlp.experts.source.torch_where_0.output

            # Capture raw down-projection output (before weighting)
            down_proj = layer.mlp.experts.source.nn_functional_linear_1.output

            # Store in lists (iteration order matches active_experts order)
            token_indices_list.append(token_idx)
            top_k_pos_list.append(top_k_pos)
            down_projs_list.append(down_proj)

        # Build dict mapping expert_id -> ExpertTrace
        # active_experts gives us the expert IDs in iteration order
        layer_traces: dict[int, ExpertTrace] = {}

        for i, expert_id_tensor in enumerate(active_experts):
            expert_id = int(expert_id_tensor.item())
            layer_traces[expert_id] = ExpertTrace(
                token_indices=token_indices_list[i],
                raw_outputs=down_projs_list[i],
                top_k_positions=top_k_pos_list[i],
            )

        expert_traces.append(layer_traces)

    # Stack: [n_layers, seq, k]
    all_indices = torch.stack(layer_indices, dim=0)
    all_weights = torch.stack(layer_weights, dim=0)

    nnsight.save(all_indices)
    nnsight.save(all_weights)
    nnsight.save(expert_traces)

# %% Build Filtered Concept Dictionary
# Tokenize a word list and slice the unembedding to those rows.
# Using "countries" by default — change to "colors" or "quantity" as needed.
property_name = "countries"
n_layers = model.config.num_hidden_layers
n_experts = model.config.num_experts
tokenizer = model.tokenizer

unembed = model.lm_head.weight.detach().cpu()
dictionary, tokens_data = build_filtered_dictionary(unembed, tokenizer, property_name)

# %% Run Expert Pursuit (SOMP on all experts)
k = min(50, len(tokens_data))
decomposition = SOMP(k=k)
evr_matrix = torch.zeros(n_layers, n_experts, k)
zscore_matrix = torch.zeros(n_layers, n_experts)
expert_concepts: dict[tuple[int, int], list[str]] = {}

for layer_idx, layer_traces in tqdm(
    enumerate(expert_traces),
    total=n_layers,
    desc="Expert Pursuit",
):
    for expert_id, trace in layer_traces.items():
        token_idxs = trace.token_indices
        raw_output = trace.raw_outputs.cpu()
        k_positions = trace.top_k_positions

        if raw_output.shape[0] == 0:
            continue

        # Gated output: gate_weight * raw_down_proj
        gate_weights = all_weights[layer_idx, 0, token_idxs, k_positions].cpu()
        gated = gate_weights.unsqueeze(-1) * raw_output

        unit = gated.double()
        if unit.norm() < 1e-6:
            continue

        # SOMP decomposition using HeadPursuit's SOMP class
        decomp_out = decomposition(
            X=unit,
            dictionary=dictionary,
            descriptors=list(range(len(dictionary))),
            device=device,
        )
        chosen = decomp_out["chosen"]
        evr_matrix[layer_idx, expert_id] = decomp_out["evr"]

        # Remap filtered indices -> vocab IDs -> decoded tokens
        vocab_ids = [tokens_data[t] for t in chosen.tolist()]
        tokens = [tokenizer.decode([vid]).strip() for vid in vocab_ids]
        expert_concepts[(layer_idx, expert_id)] = tokens

        # Z-score: internal coherence vs. random dictionary similarity
        D_chosen = dictionary[chosen]
        mean_sim = (D_chosen @ dictionary.T).mean()
        std_sim = (D_chosen @ dictionary.T).std()
        if std_sim > 1e-8:
            zscore_matrix[layer_idx, expert_id] = (
                (D_chosen @ D_chosen.T).mean() - mean_sim
            ) / std_sim

        print(f"  L{layer_idx} E{expert_id}: {tokens[:5]}")

# %% EVR Heatmap
fig = px.imshow(
    evr_matrix[:, :, -1].numpy(),
    x=[f"E{i}" for i in range(n_experts)],
    y=[f"L{i}" for i in range(n_layers)],
    color_continuous_scale="Blues",
    labels=dict(x="Experts", y="Layers", color="EVR"),
)
fig.update_layout(title=f"Expert Pursuit EVR ({property_name})", width=1400, height=600)
fig.show()

# %% Z-Score Heatmap
fig = px.imshow(
    zscore_matrix.numpy(),
    x=[f"E{i}" for i in range(n_experts)],
    y=[f"L{i}" for i in range(n_layers)],
    color_continuous_scale=px.colors.diverging.RdYlBu_r,
    color_continuous_midpoint=0.0,
    labels=dict(x="Experts", y="Layers", color="Z-Score"),
)
fig.update_layout(
    title=f"Expert Concept Coherence ({property_name})", width=1400, height=600
)
fig.show()

# %% Concept Frequency Analysis
concept_counter: Counter = Counter()
for tokens in expert_concepts.values():
    concept_counter.update(tokens[:5])

print(f"\nTop 20 concepts across all experts ({property_name}):")
for word, count in concept_counter.most_common(20):
    print(f"  {word}: {count}")
