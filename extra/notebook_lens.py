#!/usr/bin/env python
import sys

# %% Imports
from pathlib import Path

import nnsight
import plotly.graph_objects as go
import torch
import torch.nn.functional as F

parent_path = Path.cwd().resolve().parent
if str(parent_path) not in sys.path:
    sys.path.insert(0, str(parent_path))

from src.environment import load_model, set_seed

# %% Configuration
set_seed(1337)

model = load_model()
prompt = "The Eiffel Tower is in the city of"

# Tokenize and decode individual tokens for axis labels
input_ids = model.tokenizer.encode(prompt)
input_tokens = [model.tokenizer.decode([t]) for t in input_ids]
seq_len = len(input_ids)

n_layers = model.config.num_hidden_layers  # 16
n_experts_total = model.config.num_experts  # 64
k = model.config.num_experts_per_tok  # 8 (top-k)
d_model = model.config.hidden_size  # 2048

print(f"Prompt: {repr(prompt)}")
print(f"Tokens ({seq_len}): {input_tokens}")
print(f"Model: {n_layers} layers, {n_experts_total} experts/layer, top-{k} routing")


# %% Trace: capture per-expert outputs for all layers
# We need per-expert raw down_proj outputs + routing info.
# The nnsight trace uses the same proven pattern as notebook.py / capture.py.
with torch.no_grad(), model.trace(prompt) as tracer:
    layer_indices_list, layer_weights_list = [], []

    # Per-layer lists of per-expert captured data
    active_experts_per_layer: list[torch.Tensor] = []
    token_indices_per_layer: list[list[torch.Tensor]] = []
    raw_outputs_per_layer: list[list[torch.Tensor]] = []
    top_k_pos_per_layer: list[list[torch.Tensor]] = []

    for layer in model.model.layers:
        # Gate output: (_, weights [seq, k], indices [seq, k])
        _, weights, indices = layer.mlp.source.self_gate_0.output
        layer_indices_list.append(indices)
        layer_weights_list.append(weights)

        # Which experts are active (filter padding expert ID = num_experts_total)
        expert_hit = layer.mlp.experts.source.nonzero_0.output
        active_experts = expert_hit[expert_hit != n_experts_total].squeeze(-1)
        num_iters = active_experts.numel()

        tok_idx_list: list[torch.Tensor] = []
        down_proj_list: list[torch.Tensor] = []
        tkp_list: list[torch.Tensor] = []

        with tracer.iter[:num_iters]:
            top_k_pos, token_idx = layer.mlp.experts.source.torch_where_0.output
            down_proj = layer.mlp.experts.source.nn_functional_linear_1.output
            tok_idx_list.append(token_idx)
            down_proj_list.append(down_proj)
            tkp_list.append(top_k_pos)

        active_experts_per_layer.append(active_experts)
        token_indices_per_layer.append(tok_idx_list)
        raw_outputs_per_layer.append(down_proj_list)
        top_k_pos_per_layer.append(tkp_list)

    all_indices = torch.stack(layer_indices_list, dim=0)  # [n_layers, seq, k]
    all_weights = torch.stack(layer_weights_list, dim=0)  # [n_layers, seq, k]

    nnsight.save(all_indices)
    nnsight.save(all_weights)
    nnsight.save(active_experts_per_layer)
    nnsight.save(token_indices_per_layer)
    nnsight.save(raw_outputs_per_layer)
    nnsight.save(top_k_pos_per_layer)

print("Trace complete.")
for li in range(n_layers):
    print(f"  Layer {li:2d}: {len(active_experts_per_layer[li])} active experts")


# %% Build per-expert gated output lookup: expert_outputs[layer][expert_id][token_pos] = gated_output
# This reorganizes the trace data so we can look up any (layer, expert, token) triple.

expert_outputs: list[dict[int, dict[int, torch.Tensor]]] = []

for li in range(n_layers):
    layer_dict: dict[int, dict[int, torch.Tensor]] = {}
    active = active_experts_per_layer[li]
    tok_lists = token_indices_per_layer[li]
    raw_lists = raw_outputs_per_layer[li]
    tkp_lists = top_k_pos_per_layer[li]

    for i, eid_tensor in enumerate(active):
        eid = int(eid_tensor.item())
        token_idxs = tok_lists[i]
        raw_output = raw_lists[i]
        k_positions = tkp_lists[i]

        # Compute gated outputs: gate_weight * raw_down_proj
        gate_w = all_weights[li, token_idxs, k_positions]  # [n_tokens]
        gated = gate_w.unsqueeze(-1) * raw_output  # [n_tokens, d_model]

        # Map token position -> gated output vector
        tok_dict: dict[int, torch.Tensor] = {}
        for j in range(token_idxs.numel()):
            tok_pos = int(token_idxs[j].item())
            tok_dict[tok_pos] = gated[j]  # [d_model]
        layer_dict[eid] = tok_dict

    expert_outputs.append(layer_dict)


# %% RMSNorm + lm_head projection

# Extract model parameters for projection (outside trace, just weight access)
norm_weight = model.model.norm.weight.detach().cpu().float()  # [d_model]
lm_head_weight = model.lm_head.weight.detach().cpu().float()  # [vocab_size, d_model]
lm_head_bias = None
if hasattr(model.lm_head, "bias") and model.lm_head.bias is not None:
    lm_head_bias = model.lm_head.bias.detach().cpu().float()


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Apply RMSNorm: weight * x / sqrt(mean(x^2) + eps)."""
    rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return weight * (x / rms)


def expert_to_logits(expert_output: torch.Tensor) -> torch.Tensor:
    """Project a single expert's gated output to vocab logits.

    Args:
        expert_output: [d_model] gated expert output

    Returns:
        logits: [vocab_size]
    """
    x = expert_output.detach().cpu().float()
    x = rms_norm(x.unsqueeze(0), norm_weight, eps=1e-5).squeeze(0)
    logits = F.linear(x, lm_head_weight, lm_head_bias)
    return logits


# %% Compute expert logit lens for all (layer, token, expert) triples

# Results: for each (layer, token_pos), we have k=8 experts with predictions
# top_tokens[layer][token_pos] = list of (expert_id, gate_weight, top1_token_str, top1_prob)
lens_results: list[list[list[tuple[int, float, str, float]]]] = []

for li in range(n_layers):
    layer_results: list[list[tuple[int, float, str, float]]] = []
    for tok_pos in range(seq_len):
        # Get the 8 experts routed to this token, ordered by gate weight (descending)
        expert_ids = all_indices[li, tok_pos].tolist()  # [k]
        gate_weights = all_weights[li, tok_pos].tolist()  # [k]

        # Sort by gate weight descending
        pairs = sorted(zip(expert_ids, gate_weights), key=lambda x: -x[1])

        tok_results: list[tuple[int, float, str, float]] = []
        for eid, gw in pairs:
            # Look up the gated output for this (layer, expert, token)
            if eid in expert_outputs[li] and tok_pos in expert_outputs[li][eid]:
                gated_vec = expert_outputs[li][eid][tok_pos]
                logits = expert_to_logits(gated_vec)
                probs = F.softmax(logits, dim=-1)
                top_prob, top_idx = probs.max(dim=-1)
                top_token = model.tokenizer.decode([top_idx.item()])
                tok_results.append((eid, gw, top_token, top_prob.item()))
            else:
                tok_results.append((eid, gw, "?", 0.0))

        layer_results.append(tok_results)
    lens_results.append(layer_results)


# %% Plot 1: Summary heatmap (layers × tokens)
# Shows the top-1 prediction from the highest-weighted expert at each (layer, token).
# This is the "classic logit lens" shape — a quick overview of the model's processing.

summary_probs = torch.zeros(n_layers, seq_len)
summary_text: list[list[str]] = []
summary_hover: list[list[str]] = []

for li in range(n_layers):
    row_text: list[str] = []
    row_hover: list[str] = []
    for tok_pos in range(seq_len):
        eid, gw, top_tok, top_prob = lens_results[li][tok_pos][0]
        summary_probs[li, tok_pos] = top_prob
        row_text.append(f"E{eid}: {top_tok}")
        row_hover.append(
            f"Layer {li}, Token '{input_tokens[tok_pos]}'<br>"
            f"Top expert: E{eid} (gate={gw:.3f})<br>"
            f"Prediction: {repr(top_tok)} (prob={top_prob:.3f})"
        )
    summary_text.append(row_text)
    summary_hover.append(row_hover)

fig_summary = go.Figure(
    data=go.Heatmap(
        z=summary_probs.numpy(),
        x=[repr(t) for t in input_tokens],
        y=[f"Layer {i}" for i in range(n_layers)],
        text=summary_text,
        hovertext=summary_hover,
        hoverinfo="text",
        texttemplate="%{text}",
        textfont_size=10,
        colorscale="RdYlBu_r",
        zmid=0.3,
        zmin=0.0,
        zmax=1.0,
        colorbar=dict(title="Prob"),
    )
)
fig_summary.update_layout(
    title=f"Expert Logit Lens (top-1 expert per cell) — {repr(prompt)}",
    xaxis_title="Input Token",
    yaxis_title="Layer",
    width=max(700, seq_len * 110),
    height=650,
    yaxis=dict(autorange="reversed"),
)
fig_summary.show()


# %% Plot 2: Per-token detail — one separate figure per token position
# Each figure: rows = layers (0-15), columns = 8 expert slots (by descending gate weight).
# Cell text shows "E{id}: {prediction}", color = gate weight.
# Since different experts are routed at each layer, the expert IDs are embedded in each cell.

for tok_pos in range(seq_len):
    gate_matrix = []  # [n_layers, k] — gate weights (color)
    text_matrix = []  # [n_layers, k] — "E{id}\n{pred}"
    hover_matrix = []  # [n_layers, k] — full details

    for li in range(n_layers):
        gate_row = []
        text_row = []
        hover_row = []
        for eid, gw, top_tok, top_prob in lens_results[li][tok_pos]:
            gate_row.append(gw)
            text_row.append(f"E{eid}\n{top_tok}")
            hover_row.append(
                f"Layer {li}, Expert {eid}<br>"
                f"Gate weight: {gw:.4f}<br>"
                f"Prediction: {repr(top_tok)}<br>"
                f"Pred prob: {top_prob:.4f}"
            )
        gate_matrix.append(gate_row)
        text_matrix.append(text_row)
        hover_matrix.append(hover_row)

    fig = go.Figure(
        data=go.Heatmap(
            z=gate_matrix,
            x=[f"Slot {s + 1}" for s in range(k)],
            y=[f"Layer {li}" for li in range(n_layers)],
            text=text_matrix,
            hovertext=hover_matrix,
            hoverinfo="text",
            texttemplate="%{text}",
            textfont_size=9,
            colorscale="Viridis",
            zmin=0.0,
            colorbar=dict(title="Gate wt"),
        )
    )
    fig.update_layout(
        title=f"Token {tok_pos}: '{input_tokens[tok_pos]}' — Expert predictions by layer",
        xaxis_title="Expert slot (by gate weight, highest first)",
        yaxis_title="Layer",
        width=700,
        height=650,
        yaxis=dict(autorange="reversed"),
    )
    fig.show()


# %% Print summary: top predictions at the last layer
print(f"\nLast layer predictions for each token:")
print(f"{'Token':>15s}   {'#1 Expert':>30s}   {'#2 Expert':>30s}   {'#3 Expert':>30s}")
print("-" * 115)
for tok_pos in range(seq_len):
    cols = []
    for eid, gw, tok, prob in lens_results[n_layers - 1][tok_pos][:3]:
        cols.append(f"E{eid}(g={gw:.2f}) -> {repr(tok)} p={prob:.2f}")
    print(
        f"{repr(input_tokens[tok_pos]):>15s}   {'   '.join(f'{c:>30s}' for c in cols)}"
    )
