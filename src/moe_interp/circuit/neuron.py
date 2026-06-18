"""Method B' — neuron-basis attribution by reconstruction (gradient-free).

nnsight 0.7 does not build an autograd graph through the traced forward here, so
gradient attribution (AtP/RelP) on the gates returns ``None`` (see ``attribution.py``).
Following the user's pointer into ``OlmoeExperts.forward`` and Transluce's "circuits are
sparse in the neuron basis", we instead read the *privileged neuron basis* directly.

For one MoE layer, each selected (token, expert) computes the SwiGLU hidden activation
``a = act_fn(gate) * up`` (``intermediate_dim`` neurons). Neuron ``i`` of expert ``e``
adds ``g · a_i · down[e][:, i]`` to the residual stream, so its push along a target
direction ``v̂`` is ``g · a_i · (v̂ · down[e][:, i])``. Averaging that over toxic vs.
neutral tokens and differencing gives each neuron's contribution to the toxic direction
— a sparse, distributed circuit in the neuron basis, with no gradients required.

Reconstruction mirrors ``capture.py``: tap the fused-experts boundary
(``hidden_states, top_k_index, top_k_weights``) once, then re-derive activations from the
weight params outside the trace (the fused module is not steppable in nnsight 0.7).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from moe_interp.capture.capture import token_real_mask
from moe_interp.capture.model_adapter import get_model_adapter
from moe_interp.circuit.toxicity import right_padded


def _layer_boundary(model, adapter, batch: list[list[int]]):
    """Tap one batch's MoE boundary at every layer; return per-layer (hs, idx, w) + mask."""
    max_len = max(len(t) for t in batch)
    taps: list = []
    with torch.no_grad(), right_padded(model), model.trace(batch):
        for layer in model.model.layers:
            taps.append(adapter.tap_layer(layer).save())
    keep = token_real_mask([len(t) for t in batch], max_len)
    return taps, keep


def neuron_direction_attribution(
    model,
    toxic_prompts: list[list[int]],
    neutral_prompts: list[list[int]],
    layer: int,
    v: torch.Tensor,
    batch_size: int = 6,
) -> torch.Tensor:
    """Per-(expert, neuron) contribution to direction ``v`` at ``layer``: toxic − neutral.

    Returns an ``(n_experts, intermediate_dim)`` tensor; large positive entries are
    neurons that write the toxic direction more on toxic than on neutral context.
    """
    adapter = get_model_adapter(model)
    experts = model.model.layers[layer].mlp.experts
    # Reconstruct on CPU: avoids CPU/MPS device mixing and offloaded-weight placeholders.
    down = experts.down_proj.detach().float().cpu()  # (E, D, I)
    gate_up = experts.gate_up_proj.detach().float().cpu()  # (E, 2I, D)
    act_fn = experts.act_fn
    n_experts, d_model, d_ff = down.shape
    vhat = F.normalize(v.float().cpu(), dim=0)
    proj = torch.einsum("d,edi->ei", vhat, down)  # (E, I): v̂ through each expert's down

    def accumulate(prompts: list[list[int]]) -> torch.Tensor:
        acc = torch.zeros(n_experts, d_ff)
        cnt = torch.zeros(n_experts)
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            taps, keep = _layer_boundary(model, adapter, batch)
            hs, idx, w = adapter.unpack_boundary(taps[layer])
            hs = hs.float().cpu()
            idx = idx.cpu()
            w = w.float().cpu()
            keep = keep.cpu()
            for e in torch.unique(idx).tolist():
                t_idx, k_idx = (idx == e).nonzero(as_tuple=True)
                m = keep[t_idx]
                t_idx, k_idx = t_idx[m], k_idx[m]
                if t_idx.numel() == 0:
                    continue
                gate, up = F.linear(hs[t_idx], gate_up[e]).chunk(2, dim=-1)
                a = act_fn(gate) * up  # (m, I) neuron activations
                contrib = w[t_idx, k_idx, None] * a * proj[e]  # push along v̂ per neuron
                acc[e] += contrib.sum(0)
                cnt[e] += t_idx.numel()
        return acc / cnt.clamp_min(1).unsqueeze(1)

    return accumulate(toxic_prompts) - accumulate(neutral_prompts)


def top_neurons(attr: torch.Tensor, k: int = 20) -> list[tuple[int, int, float]]:
    """Top ``k`` (expert, neuron, value) by |attribution|."""
    flat = attr.flatten()
    order = flat.abs().argsort(descending=True)[:k]
    d_ff = attr.shape[1]
    return [
        (int(i // d_ff), int(i % d_ff), float(flat[i])) for i in order.tolist()
    ]


def sparsity(attr: torch.Tensor) -> dict:
    """How concentrated the toxic contribution is across all (expert, neuron) units.

    ``effective_neurons`` = (Σ|a|)² / Σ a²  — the participation ratio of the absolute
    attribution: ``n_total`` if every neuron contributes equally, ``1`` if a single neuron
    dominates. ``top{20,100}_frac`` is the share of total |attribution| in the largest
    20 / 100 neurons.
    """
    x = attr.flatten().abs()
    total = x.sum().clamp_min(1e-12)
    srt = torch.sort(x, descending=True).values
    return {
        "n_total": int(x.numel()),
        "effective_neurons": float((total**2 / (x**2).sum().clamp_min(1e-12)).item()),
        "top20_frac": float((srt[:20].sum() / total).item()),
        "top100_frac": float((srt[:100].sum() / total).item()),
    }


def name_neurons(
    model,
    layer: int,
    neurons: list[tuple[int, int, float]],
    dictionary: torch.Tensor,
    tokenizer,
    k: int = 6,
) -> list[dict]:
    """Decode each (expert, neuron)'s write-direction to tokens (a per-neuron logit lens).

    Neuron ``i`` of expert ``e`` writes along ``down_proj[e][:, i]``; oriented by the sign
    of its toxic attribution, its top unembedding tokens name what the neuron contributes.
    """
    down = model.model.layers[layer].mlp.experts.down_proj.detach().float().cpu()
    out: list[dict] = []
    for expert_i, neuron_i, val in neurons:
        direction = down[expert_i, :, neuron_i]
        if val < 0:
            direction = -direction
        scores = dictionary @ F.normalize(direction, dim=0)
        idx = torch.topk(scores, k).indices.tolist()
        out.append(
            {
                "expert": expert_i,
                "neuron": neuron_i,
                "attribution": val,
                "tokens": [tokenizer.decode([i]) for i in idx],
            }
        )
    return out
