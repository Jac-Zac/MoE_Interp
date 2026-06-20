"""Boundary-B precision edits: reach *inside* a gate-AtP-flagged expert.

gate-AtP and the patching grid act at the router **gate** (``layer.mlp.experts.inputs[0]``)
— the only per-expert node the fused kernel exposes — so they can only scale or zero an
expert's *whole* contribution. This module goes one level finer, into the expert's own MLP
(reconstructed exactly as :class:`~moe_interp.capture.model_adapter.SwiGLUMoEAdapter`),
where the individual intermediate *neurons* and a residual *subspace* live.

Pipeline: identify a causal expert with gate-AtP, then here

  1. score which of *its* neurons write toward the toxic direction ``v``, and
  2. surgically edit only those neurons / that subspace — far more targeted than the gate.

**Why a local linear score, not a neuron-level AtP.** Only the gate is differentiable at
the fused boundary, so a global ``dL/d(neuron)`` is unavailable on nnsight 0.7 — this is
exactly why the earlier *global* neuron-RelP was dropped. We sidestep it by attributing
against the diff-of-means toxic direction ``v`` (:mod:`~moe_interp.circuit.direction`). A
SwiGLU expert's output is ``sum_i neuron_i · down[:, i]``, so neuron ``i`` writes
``down[:, i] · v̂`` toward toxicity per unit activation and its toxic contribution is
``mean_tokens(neuron_i) · (down[:, i] · v̂)``. That is causal-direction DLA at neuron
resolution — no backward pass through the rest of the network.

**The edits are weight surgery** on the flagged expert's ``down_proj`` (restored on exit),
so they need no per-step nnsight hook and compose with an ordinary ``model.generate``:

  - zero a few neurons         → ``down_proj[e][:, ids] = 0``
  - remove the toxic subspace  → ``down_proj[e] = (I − v̂v̂ᵀ) down_proj[e]`` (expert ``e``
                                 can no longer write along ``v̂``; orthogonal features kept)

Only SwiGLU experts (OLMoE/Mixtral) are supported; gpt-oss adds biases/clamp and would
need its own intermediate-neuron path (see the adapter).
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn.functional as F

from moe_interp.capture.model_adapter import SwiGLUMoEAdapter, get_model_adapter
from moe_interp.circuit.toxicity import right_padded


def _require_swiglu(model) -> SwiGLUMoEAdapter:
    adapter = get_model_adapter(model)
    if not isinstance(adapter, SwiGLUMoEAdapter):
        raise NotImplementedError(
            "expert_edit currently supports SwiGLU experts (OLMoE/Mixtral); got "
            f"model_type={adapter.config.model_type!r}. Add an intermediate-neuron path "
            "for it (gpt-oss has biases + a clamp) before using this module."
        )
    return adapter


def _experts(model, layer: int):
    """The real fused-experts module at ``layer`` (weights readable/writable outside a trace)."""
    return model.model.layers[layer].mlp.experts


def _neuron_acts(experts, expert_id: int, h: torch.Tensor) -> torch.Tensor:
    """SwiGLU intermediate activations ``act(gate) * up`` for ``h`` ``(n, D)`` → ``(n, I)``."""
    gate_up = experts.gate_up_proj[expert_id].detach().float()  # (2I, D)
    gate, up = F.linear(h.float(), gate_up).chunk(2, dim=-1)
    return experts.act_fn(gate) * up


# --- attribution: which neurons of one expert write toward the toxic direction -----------


def expert_mean_neuron_acts(
    model, prompts: list[list[int]], layer: int, expert_id: int, *, batch_size: int = 8
) -> tuple[torch.Tensor, int]:
    """Mean intermediate activation of expert ``(layer, expert_id)`` over the *real* tokens
    routed to it across ``prompts``. Returns ``(mean_act (I,), n_routed_tokens)``."""
    experts = _experts(model, layer)
    total: torch.Tensor | None = None
    n = 0
    with right_padded(model), torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            lengths = torch.tensor([len(t) for t in batch])
            with model.trace(batch):
                hs, idx, _ = model.model.layers[layer].mlp.experts.inputs[0]
                hs = hs.save()  # (N, D), N = B * T (fused/flattened token axis)
                idx = idx.save()  # (N, top_k)
            T = idx.shape[0] // len(batch)
            tok = torch.arange(idx.shape[0])
            real = (tok % T) < lengths[tok // T]  # drop right-padding
            routed = ((idx == expert_id).any(dim=-1).cpu() & real).to(hs.device)
            if not bool(routed.any()):
                continue
            acts = _neuron_acts(experts, expert_id, hs[routed])  # (m, I)
            s = acts.sum(0).cpu()
            total = s if total is None else total + s
            n += int(routed.sum())
    if n == 0:
        raise ValueError(
            f"expert L{layer}E{expert_id} was never routed on these prompts"
        )
    return total / n, n


def neuron_toxic_scores(
    model,
    prompts: list[list[int]],
    layer: int,
    expert_id: int,
    v: torch.Tensor,
    *,
    batch_size: int = 8,
) -> tuple[torch.Tensor, int]:
    """Per-neuron toxic-write score for expert ``(layer, expert_id)``::

        score_i = mean_tokens(neuron_i) · (down[:, i] · v̂)

    Signed: positive = the neuron writes toward toxicity (firing raises ``h · v̂``). Returns
    ``(scores (I,), n_routed_tokens)``.
    """
    experts = _experts(model, layer)
    mean_act, n = expert_mean_neuron_acts(
        model, prompts, layer, expert_id, batch_size=batch_size
    )
    down = experts.down_proj[expert_id].detach().float().cpu()  # (D, I)
    vhat = F.normalize(v.float().cpu(), dim=0)  # (D,)
    write = down.transpose(0, 1) @ vhat  # (I,): down[:, i] · v̂
    return mean_act * write, n


def top_toxic_neurons(scores: torch.Tensor, k: int = 20) -> list[tuple[int, float]]:
    """The ``k`` neuron indices with the largest |toxic-write score|, as ``(neuron, score)``."""
    order = scores.abs().argsort(descending=True)[:k]
    return [(int(i), float(scores[i])) for i in order.tolist()]


# --- surgical edits: weight surgery on one expert's down-projection ----------------------


@contextmanager
def _restore_down(model, layer: int, expert_id: int):
    """Snapshot ``down_proj[expert_id]`` and restore it on exit (edits are non-permanent)."""
    experts = _experts(model, layer)
    original = experts.down_proj.data[expert_id].clone()
    try:
        yield experts
    finally:
        experts.down_proj.data[expert_id] = original


@contextmanager
def zero_neurons(model, layer: int, expert_id: int, neuron_ids: list[int]):
    """Zero the output of specific intermediate neurons of one expert (restored on exit).

    ``down_proj`` is ``(E, D, I)``; column ``i`` is neuron ``i``'s write vector, so zeroing
    those columns deletes exactly those neurons' contribution while the expert otherwise
    behaves normally.
    """
    with _restore_down(model, layer, expert_id) as experts:
        ids = torch.as_tensor(list(neuron_ids), device=experts.down_proj.device)
        experts.down_proj.data[expert_id, :, ids] = 0.0
        yield


@contextmanager
def project_expert_off(model, layer: int, expert_id: int, v: torch.Tensor):
    """Remove direction ``v`` from one expert's writable subspace (restored on exit).

    Replaces ``down_proj[e]`` with ``(I − v̂v̂ᵀ) down_proj[e]`` so expert ``e`` can no longer
    add anything along ``v̂`` to the residual stream, while every orthogonal feature it
    writes is untouched. The surgical analogue of the whole-residual project-out in
    :mod:`~moe_interp.circuit.intervene` — scoped to the one causally-implicated expert.
    """
    with _restore_down(model, layer, expert_id) as experts:
        W = experts.down_proj.data[expert_id]  # (D, I)
        vhat = F.normalize(v.to(W.device, W.dtype), dim=0)  # (D,)
        experts.down_proj.data[expert_id] = W - torch.outer(vhat, vhat @ W)
        yield


# --- orchestration -----------------------------------------------------------------------


def run_expert_edit(
    model,
    *,
    concept: str = "offensive",
    layer: int | None = None,
    expert: int | None = None,
    top_neurons: int = 20,
    batch_size: int = 8,
    max_new_tokens: int = 32,
) -> dict:
    """Identify a causal expert (gate-AtP), score its neurons, then generate under each edit.

    With ``layer``/``expert`` unset the top gate-AtP promoter is used. The toxic direction
    ``v`` is the diff-of-means residual *at that expert's own layer*, so neuron scoring and
    the project-out edit share the expert's basis. Returns per-method (baseline /
    zero-top-neurons / project-expert-off) concept propensity + offensive-word rate, plus
    the scored neurons in ``meta``.
    """
    from moe_interp.circuit.attribution import gate_attribution
    from moe_interp.circuit.direction import collect_last_token_residuals
    from moe_interp.circuit.intervene import concept_propensity, concept_regex, generate
    from moe_interp.circuit.prompts import default_prompts
    from moe_interp.pursuit.concepts import CONCEPT_WORDS, build_toxic_token_ids

    _require_swiglu(model)
    eliciting, neutral = default_prompts(model.tokenizer)
    concept_words = CONCEPT_WORDS[concept]
    concept_ids = build_toxic_token_ids(model.tokenizer, concept_words)

    # 1. flagged expert — default to the strongest gate-AtP promoter.
    if layer is None or expert is None:
        attr = gate_attribution(model, eliciting, concept_ids, batch_size=batch_size)
        flat = int(attr.flatten().argmax())
        layer, expert = flat // attr.shape[1], flat % attr.shape[1]

    # 2. toxic direction at the expert's own layer; 3. score its neurons against it.
    v = collect_last_token_residuals(model, eliciting, layer).mean(0) - (
        collect_last_token_residuals(model, neutral, layer).mean(0)
    )
    scores, n_tok = neuron_toxic_scores(
        model, eliciting, layer, expert, v, batch_size=batch_size
    )
    top = top_toxic_neurons(scores, top_neurons)
    neuron_ids = [i for i, _ in top]

    # 4. generate + score under each edit (weight surgery => intervention=None in the trace).
    pattern = concept_regex(concept_words)
    tok = model.tokenizer

    def evaluate() -> dict:
        props, counts, examples = [], [], []
        for j, ids in enumerate(eliciting):
            cont = generate(model, ids, max_new_tokens, None)
            props.append(concept_propensity(model, ids, cont, concept_ids, None))
            counts.append(len(pattern.findall(tok.decode(cont))))
            if j < 4:
                examples.append(tok.decode(cont).strip())
        return {
            "eliciting_propensity": float(sum(props) / len(props)),
            "eliciting_word_frac": float(sum(c > 0 for c in counts) / len(counts)),
            "examples": examples,
        }

    print(
        f"editing L{layer}E{expert} ({n_tok} routed tokens); baseline ...", flush=True
    )
    methods = {"baseline": evaluate()}
    print(f"  zeroing top {len(neuron_ids)} toxic neurons ...", flush=True)
    with zero_neurons(model, layer, expert, neuron_ids):
        methods[f"zero-{len(neuron_ids)}-neurons"] = evaluate()
    print("  projecting expert output off v ...", flush=True)
    with project_expert_off(model, layer, expert, v):
        methods["project-expert-off-v"] = evaluate()

    return {
        "methods": methods,
        "meta": {
            "concept": concept,
            "layer": int(layer),
            "expert": int(expert),
            "n_routed_tokens": n_tok,
            "top_neurons": [[i, s] for i, s in top],
        },
    }
