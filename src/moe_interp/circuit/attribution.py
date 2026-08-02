"""Estimate direct router-gate contributions with gradient × activation.

Zeroing expert ``e``'s post-top-k gate changes the metric, to first order, by
``-g_e · dL/dg_e``. We store its estimated contribution:

    attribution_e  ≈  g_e · dL/dg_e              (summed over token positions)

where ``L`` is the concept-logit metric. A positive value predicts that zeroing the gate will
lower the metric. The stored validation compares this approximation with exact per-slot gate
ablation in ``data/<model>/circuit/compare/faithfulness.json``.
"""

from pathlib import Path

import torch

from moe_interp.capture.model_adapter import model_num_experts
from moe_interp.circuit.concept_probe import relative_logit_score, right_padded


def prompt_regime_suffix(hi: float = 0.5, challenging: bool = False) -> str:
    """Encode non-default RealToxicityPrompts selection settings in artifact names."""
    if hi == 0.5 and not challenging:
        return ""
    return f"_hi{hi:g}" + ("_chal" if challenging else "")


def attribution_grid_path(
    circuit_dir: Path,
    *,
    concept: str,
    n_prompts: int,
    hi: float = 0.5,
    challenging: bool = False,
) -> Path:
    """Return the canonical grid path, reusing the legacy offensive path if present."""
    regime = prompt_regime_suffix(hi, challenging)
    grid_dir = circuit_dir / "attribution"
    canonical = grid_dir / f"atp_grid_{concept}_n{n_prompts}{regime}.npy"
    legacy = grid_dir / f"atp_grid_n{n_prompts}{regime}.npy"
    if concept == "offensive" and legacy.exists() and not canonical.exists():
        return legacy
    return canonical


def gate_attribution(
    model,
    prompts: list[list[int]],
    concept_ids: list[int],
    batch_size: int = 8,
) -> torch.Tensor:
    """Return per-(layer, expert) attribution for the concept-logit metric.

    Returns a ``(n_layers, n_experts)`` tensor; entry ``[l, e]`` is the gradient-times-gate
    attribution summed over all token positions and prompts (sign: positive = the expert
    raises the score, so ablating it is predicted to lower it).
    """
    n_layers = model.config.num_hidden_layers
    n_experts = model_num_experts(model)
    attr = torch.zeros(n_layers, n_experts)

    with right_padded(model):
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            lengths = torch.tensor([len(t) for t in batch])
            # Lists must live OUTSIDE the trace: proxies saved into a list created inside
            # the trace are not bound back after it (nnsight 0.7 scoping; see capture.py).
            idx_saved, gate_saved, grad_saved, gate_proxies = [], [], [], []
            with model.trace(batch):
                for layer in model.model.layers:  # forward order to register
                    _, top_k_index, top_k_weights = layer.mlp.experts.inputs[0]
                    top_k_weights.requires_grad_(True)
                    gate_proxies.append(top_k_weights)
                    idx_saved.append(top_k_index.save())
                    gate_saved.append(top_k_weights.save())
                logits = model.output.logits
                rows = torch.arange(logits.shape[0])
                last = logits[rows, lengths - 1]
                metric = relative_logit_score(last, concept_ids).sum()
                # nnsight: backward() is a context manager; read .grad INSIDE it, in
                # reverse execution order (see nnsight.net/features/3_gradients).
                with metric.backward():
                    for proxy in reversed(gate_proxies):
                        grad_saved.append(proxy.grad.save())
                grad_saved.reverse()  # back to layer order

            for layer_idx in range(n_layers):
                idx = idx_saved[layer_idx].cpu()  # (tokens, top_k)
                gate = gate_saved[layer_idx].detach().float().cpu()
                grad = grad_saved[layer_idx].detach().float().cpu()
                contrib = (gate * grad).flatten()  # gate * dL/dgate, per (token, slot)
                attr[layer_idx].index_add_(0, idx.flatten(), contrib)
    return attr
