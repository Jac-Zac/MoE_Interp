"""Method C — RelP vs AtP at the expert-neuron basis (post up-proj + nonlinearity).

Relevance Patching (arXiv:2508.21258) keeps AtP's score = (Δactivation)·(backward signal)
but swaps AtP's raw gradient for an **LRP relevance**, whose key rule is treating RMSNorm
as a constant scale (the LN-rule) and nonlinearities (SiLU) as identity in the backward.
The paper's diagnosis: raw gradients are noisy through RMSNorm, which is exactly why AtP
under-reads MLP nodes (their GPT-2 Large MLP-output faithfulness: AtP r=0.006 vs RelP 0.956).

What that means here. The toxic-logit metric is ``L = toxic_dir · RMSNorm(h_final)``. Under
the LN-rule the norm scale is constant, so the relevance of the final residual is
``∝ toxic_dir`` (the unembedding toxic direction). The residual stream is additive, so under
LRP that relevance reaches lower layers ≈ unchanged. Hence **RelP's backward signal at any
layer's residual is the toxic unembedding direction**, while **AtP uses the raw autograd
``dL/d(residual)``** (which folds in the noisy RMSNorm Jacobian).

The node, per your pointer, is the SwiGLU hidden activation ``a = act_fn(gate) · up`` (after
the up-projection and nonlinearity). Neuron ``i`` of expert ``e`` adds ``g · a_i · down[e][:, i]``
to the residual, so its attribution onto a backward direction ``R`` is
``g · a_i · (R · down[e][:, i])``. Experts fire sparsely (top-k per token), so we reconstruct
only the experts actually selected at the scored (last-token) position — mirroring
``neuron.py``/``capture.py`` since the fused kernel is not differentiable per neuron.

Three backward signals share the same neuron formula, for a clean comparison:
  - **RelP** : R = toxic unembedding direction         (LRP, RMSNorm-as-constant)
  - **AtP**  : R = autograd dL/d(layer-ℓ residual)      (raw gradient, last token)
  - **DLA**  : R = diff-of-means toxic direction        (see neuron.py)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from moe_interp.capture.capture import token_real_mask
from moe_interp.capture.model_adapter import get_model_adapter
from moe_interp.circuit.toxicity import right_padded, toxic_logit_score


def toxic_relevance_direction(
    dictionary: torch.Tensor, toxic_ids: list[int]
) -> torch.Tensor:
    """RelP backward signal: the relative toxic-logit direction in residual space.

    ``mean(U[toxic]) - mean(U)`` — the RMSNorm-as-constant LRP relevance of the residual
    stream for the toxic-logit metric.
    """
    return (dictionary[toxic_ids].mean(0) - dictionary.mean(0)).float().cpu()


def residual_gradient(
    model,
    prompts: list[list[int]],
    toxic_ids: list[int],
    layer: int,
    batch_size: int = 6,
) -> torch.Tensor:
    """AtP backward signal: mean last-token autograd ``dL/d(layer-ℓ residual)`` ``(d_model,)``.

    The raw gradient (with the RMSNorm Jacobian baked in) that RelP deliberately replaces.
    """
    grads: list[torch.Tensor] = []
    with right_padded(model):
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            lengths = [len(t) for t in batch]
            max_len = max(lengths)
            with model.trace(batch):
                hidden = model.model.layers[layer].output[0]
                hidden.requires_grad_(True)
                hidden_saved = hidden.save()
                logits = model.output.logits
                rows = torch.arange(logits.shape[0])
                last = logits[rows, torch.tensor(lengths) - 1]
                metric = toxic_logit_score(last, toxic_ids).sum()
                with metric.backward():
                    grad = hidden_saved.grad.save()
            g = grad.detach().float().cpu()
            # g is (batch*max_len, d_model) or (batch, max_len, d_model); normalise shape.
            g = g.reshape(len(batch), max_len, -1)
            for p, length in enumerate(lengths):
                grads.append(g[p, length - 1])
    return torch.stack(grads).mean(0)


def neuron_attribution(
    model,
    prompts: list[list[int]],
    layer: int,
    direction: torch.Tensor,
    batch_size: int = 6,
) -> torch.Tensor:
    """Per-(expert, neuron) attribution onto ``direction`` at the last token.

    ``attr[e, i] = mean_prompts( g · a_i · (direction · down[e][:, i]) )`` over prompts where
    expert ``e`` fired at the last token. Returns ``(n_experts, intermediate_dim)``.
    """
    adapter = get_model_adapter(model)
    experts = model.model.layers[layer].mlp.experts
    down = experts.down_proj.detach().float().cpu()  # (E, D, I)
    gate_up = experts.gate_up_proj.detach().float().cpu()  # (E, 2I, D)
    act_fn = experts.act_fn
    n_experts, d_model, d_ff = down.shape
    proj = torch.einsum("d,edi->ei", direction.float().cpu(), down)  # (E, I)

    acc = torch.zeros(n_experts, d_ff)
    cnt = torch.zeros(n_experts)
    with right_padded(model):
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            lengths = [len(t) for t in batch]
            max_len = max(lengths)
            with torch.no_grad(), model.trace(batch):
                tap = adapter.tap_layer(model.model.layers[layer]).save()
            hs, idx, w = adapter.unpack_boundary(tap)
            hs = hs.float().cpu().reshape(len(batch), max_len, -1)
            idx = idx.cpu().reshape(len(batch), max_len, -1)
            w = w.float().cpu().reshape(len(batch), max_len, -1)
            for p, length in enumerate(lengths):
                pos = length - 1
                for slot in range(idx.shape[-1]):  # top-k experts fired at this token
                    e = int(idx[p, pos, slot])
                    g = float(w[p, pos, slot])
                    gate, up = F.linear(hs[p, pos], gate_up[e]).chunk(2, dim=-1)
                    a = act_fn(gate) * up  # (I,) neuron activations
                    acc[e] += g * a * proj[e]
                    cnt[e] += 1
    return acc / cnt.clamp_min(1).unsqueeze(1)


def expert_effect(attr: torch.Tensor) -> torch.Tensor:
    """Aggregate the neuron-basis attribution to a per-expert effect (sum over neurons)."""
    return attr.sum(dim=1)


def top_neurons(attr: torch.Tensor, k: int = 20) -> list[tuple[int, int, float]]:
    """Top ``k`` (expert, neuron, value) by |attribution|."""
    flat = attr.flatten()
    d_ff = attr.shape[1]
    order = flat.abs().argsort(descending=True)[:k]
    return [(int(i // d_ff), int(i % d_ff), float(flat[i])) for i in order.tolist()]
