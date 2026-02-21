#!/usr/bin/env python

# %% Imports
import logging
from collections import Counter
from dataclasses import dataclass

import nnsight
import plotly.express as px
import torch
from nnsight import LanguageModel
from residual.sparse_decomposition import SOMP
from tqdm import tqdm

from src.data import load_triviaqa
from src.environment import get_device, set_seed
from src.pursuit import build_filtered_dictionary

# Suppress noisy HTTP request logs from huggingface/datasets
logging.getLogger("httpx").setLevel(logging.WARNING)

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


# %% Load TriviaQA questions (multiple documents)
n_docs = 20
tokenizer = model.tokenizer
questions = load_triviaqa(tokenizer, n_docs=n_docs)
print(f"Loaded {len(questions)} TriviaQA questions")

# %% Trace all documents and collect last-content-token gated outputs
# For each document, we run the sacred tracing block and extract the
# gated output at the last content token routed to each expert.
# Result: activations[doc_idx, layer, expert, :] = gated vector at last content token
n_layers = model.config.num_hidden_layers
n_experts = model.config.num_experts
d_model = model.config.hidden_size

activations = torch.zeros(len(questions), n_layers, n_experts, d_model)

for doc_idx, question in enumerate(tqdm(questions, desc="Tracing documents")):
    doc = question.token_ids
    content_start = question.content_start
    content_end = question.content_end

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
    # --- End sacred tracing block ---

    # Extract last-content-token gated output per expert per layer
    for layer_idx, layer_traces_dict in enumerate(expert_traces):
        for expert_id, trace in layer_traces_dict.items():
            token_idxs = trace.token_indices
            raw_output = trace.raw_outputs
            k_positions = trace.top_k_positions

            if raw_output.shape[0] == 0:
                continue

            # Filter to content tokens only
            content_mask = (token_idxs >= content_start) & (token_idxs < content_end)
            if not content_mask.any():
                continue

            token_idxs_c = token_idxs[content_mask]
            raw_output_c = raw_output[content_mask]
            k_positions_c = k_positions[content_mask]

            # Take the last content token routed to this expert
            last_idx = token_idxs_c.argmax()  # highest token position = latest
            # all_weights shape: [n_layers, seq_len, k] (no batch dim)
            gate_w = all_weights[
                layer_idx, token_idxs_c[last_idx], k_positions_c[last_idx]
            ]
            gated_last = gate_w.cpu() * raw_output_c[last_idx]

            activations[doc_idx, layer_idx, expert_id] = gated_last

    if (doc_idx + 1) % 5 == 0:
        print(f"  Traced {doc_idx + 1}/{len(questions)} documents")

print(f"Activations shape: {activations.shape}")

# %% Build Filtered Concept Dictionary
# Tokenize a word list and slice the unembedding to those rows.
# Using "countries" by default — change to "colors" or "quantity" as needed.
property_name = "countries"

unembed = model.lm_head.weight.detach().cpu()
dictionary, tokens_data = build_filtered_dictionary(unembed, tokenizer, property_name)

# %% Run Expert Pursuit (SOMP on all experts, multi-document)
# X = [n_docs, d_model] per expert — one row per document's last-content-token
# gated output. This matches HeadPursuit's approach (one sample per document).
k = min(50, len(tokens_data))
decomposition = SOMP(k=k)
evr_matrix = torch.zeros(n_layers, n_experts, k)
zscore_matrix = torch.zeros(n_layers, n_experts)
expert_concepts: dict[tuple[int, int], list[str]] = {}

for li in tqdm(range(n_layers), desc="Expert Pursuit"):
    for ei in range(n_experts):
        X = activations[:, li, ei, :]  # [n_docs, d_model]
        if X.norm() < 1e-6:
            continue

        # SOMP requires float64; MPS doesn't support it, so always run on CPU
        decomp_out = decomposition(
            X=X.double(),
            dictionary=dictionary,
            descriptors=list(range(len(dictionary))),
            device="cpu",
        )
        chosen = decomp_out["chosen"]
        evr_matrix[li, ei] = decomp_out["evr"]

        # Remap filtered indices -> vocab IDs -> decoded tokens
        vocab_ids = [tokens_data[t] for t in chosen.tolist()]
        tokens = [tokenizer.decode([vid]).strip() for vid in vocab_ids]
        expert_concepts[(li, ei)] = tokens

        # Z-score: internal coherence vs. random dictionary similarity
        D_chosen = dictionary[chosen]
        mean_sim = (D_chosen @ dictionary.T).mean()
        std_sim = (D_chosen @ dictionary.T).std()
        if std_sim > 1e-8:
            zscore_matrix[li, ei] = (
                (D_chosen @ D_chosen.T).mean() - mean_sim
            ) / std_sim

print(f"Analyzed {len(expert_concepts)} active experts across {n_layers} layers")

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
