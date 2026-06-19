"""Method B — gradient attribution patching over router gates (AtP / RelP-style).

Whole-expert ablation needs one forward *per expert*; attribution patching estimates
every expert's effect from a single backward pass via a first-order Taylor expansion.
For the ablation baseline (gate -> 0), the predicted change in the metric from removing
expert ``e`` is

    attribution_e  ≈  - g_e · dL/dg_e            (summed over token positions)

where ``g_e`` is the router gate weight wherever expert ``e`` was selected and ``L`` is
the toxic-logit metric. This is the linear-attribution core of AtP/RelP: the gate
weights are real differentiable tensors at the fused-experts boundary (the per-expert
hidden neurons are not materialised by the fused kernel, so the gate is the finest node
we can take gradients of here). A large positive attribution means "this expert pushes
the metric up; ablating it would push it down" — summed over the top-ranked experts it
gives the *distributed* circuit.
"""

from __future__ import annotations

import torch

from moe_interp.circuit.toxicity import right_padded, toxic_logit_score


def gate_attribution(
    model,
    prompts: list[list[int]],
    toxic_ids: list[int],
    batch_size: int = 8,
) -> torch.Tensor:
    """Per-(layer, expert) attribution for the toxic-logit metric.

    Returns a ``(n_layers, n_experts)`` tensor; entry ``[l, e]`` is the gradient-times-gate
    attribution summed over all token positions and prompts (sign: positive = the expert
    raises the toxic score, so ablating it would lower it).
    """
    n_layers = model.config.num_hidden_layers
    n_experts = model.config.num_local_experts
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
                metric = toxic_logit_score(last, toxic_ids).sum()
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


def top_experts(attr: torch.Tensor, k: int = 15) -> list[tuple[int, int, float]]:
    """Return the ``k`` (layer, expert, attribution) entries with the largest |attribution|."""
    flat = attr.flatten()
    order = flat.abs().argsort(descending=True)[:k]
    n_experts = attr.shape[1]
    return [
        (int(i // n_experts), int(i % n_experts), float(flat[i])) for i in order.tolist()
    ]
