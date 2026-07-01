"""gate-AtP — gradient attribution patching over the router gates (AtP-style).

The causal localizer for the toxic-expert circuit, and the canonical reference for its
validation (other modules point here). Whole-expert ablation needs one forward *per expert*;
attribution patching estimates every expert's effect from a single backward pass via a
first-order Taylor expansion. Zeroing expert ``e``'s gate (``g_e -> 0``) changes the metric,
to first order, by ``-g_e · dL/dg_e``; we store the expert's *contribution* — the negative of
that, i.e. how much the metric would drop on ablation:

    attribution_e  ≈  g_e · dL/dg_e              (summed over token positions)

where ``g_e`` is the router gate weight wherever expert ``e`` was selected and ``L`` is the
toxic-logit metric. The gate weights are real differentiable tensors at the fused-experts
boundary (the per-expert hidden neurons are not materialised by the fused kernel, so the gate
is the finest node we can take gradients of here). A large positive attribution means "this
expert pushes the metric up; ablating it would push it down" — summed over the top-ranked
experts it gives the *distributed* circuit. (Sign: this is the same convention as the patching
grid's ``base - ablated``, with which the stored grid correlates +0.69; a leading minus would
flip it to anti-correlation.)

gate-AtP is a first-order approximation of exhaustive activation patching (zero each gate in a
separate forward pass). The two were checked once on the toxicity grid and agreed closely
(pooled Pearson r≈0.69, up to ≈0.96 in the late layers), so only the cheap AtP grid is run; the
frozen check lives in ``data/<model>/circuit/compare/faithfulness.json``.
"""

import torch

from moe_interp.capture.model_adapter import model_num_experts
from moe_interp.circuit.concept_probe import relative_logit_score, right_padded


def gate_attribution(
    model,
    prompts: list[list[int]],
    concept_ids: list[int],
    batch_size: int = 8,
) -> torch.Tensor:
    """Per-(layer, expert) attribution for the toxic-logit metric.

    Returns a ``(n_layers, n_experts)`` tensor; entry ``[l, e]`` is the gradient-times-gate
    attribution summed over all token positions and prompts (sign: positive = the expert
    raises the toxic score, so ablating it would lower it).
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
