"""Generation-time interventions to suppress a concept — the causal capstone.

Gate-AtP identifies which experts *cause* the concept. Here we act on those experts (and no
other locus — every intervention is expert-level, never on the residual stream): during greedy
generation we either **knock out** the selected experts (zero their router gate) or **steer
their output** (add ``α·v_e`` to each expert's MLP output, ``v_e`` the diff-of-means in
expert-output space), and measure the change in the concept propensity versus the un-intervened
baseline. Two controls keep it honest: a **random** expert set (specificity — does it have to be
*these* experts?) and a **neutral** prompt set (collateral — does the intervention break ordinary
generation?).

A concept is scored three ways: the mean **concept-logit propensity** over the generated
continuation (the sensitive probe), a lexical **concept-word rate** in the decoded text, and a
**distinct-1** coherence guard. Lower propensity/word-rate = less of the concept — but only if
distinct-1 stays healthy (≈0.6–0.9); a drop with collapsed distinct-1 is degraded text, not
removal. Generated text is kept for qualitative inspection.
"""

from __future__ import annotations

import re
from collections.abc import Callable

import torch

from moe_interp.circuit.toxicity import relative_logit_score
from moe_interp.pursuit.concepts import CONCEPT_WORDS


def knockout_intervention(experts: list[tuple[int, int]]) -> Callable:
    """Intervention that zeros the router gate of each ``(layer, expert)`` (forward order)."""
    by_layer: dict[int, list[int]] = {}
    for layer, e in experts:
        by_layer.setdefault(layer, []).append(e)

    def fn(model):
        for layer in sorted(
            by_layer
        ):  # nnsight 0.7 needs envoys touched in forward order
            _, idx, w = model.model.layers[layer].mlp.experts.inputs[0]
            for e in by_layer[layer]:
                w[idx == e] = 0.0

    return fn


def expert_steer_intervention(
    v_by_le: dict[tuple[int, int], torch.Tensor], alpha: float
) -> Callable:
    """Steer the *output activation* of named experts by ``alpha * v_e`` (DoM in expert space).

    This is the steering experiment done at expert granularity. The model adds
    ``gate_{t,e} * f_e(h_t)`` to the residual, where ``f_e`` is expert ``e``'s raw MLP output and
    ``gate`` its (normalised) router weight. Shifting that output ``f_e -> f_e + alpha * v_e``
    therefore changes the residual by ``gate_{t,e} * alpha * v_e`` at exactly the tokens routed to
    ``e`` — which is what we add here, reading the live router gate from ``experts.inputs``.

    ``v_e`` is the per-expert diff-of-means in expert-output space (toxic - neutral; see
    :func:`~moe_interp.circuit.steer.collect_expert_output_dom`), so ``alpha = -1`` subtracts one
    unit of the toxic direction from the expert (detox), ``alpha = +1`` amplifies it. The shift is
    per-expert and gate-weighted: it lands only at tokens routed to ``e`` and is scaled by the live
    router gate, so it is never stacked unscaled across layers.

    ``v_by_le`` maps ``(layer, expert) -> v_e`` (model-dim); experts absent here are left alone.
    """
    by_layer: dict[int, list[tuple[int, torch.Tensor]]] = {}
    for (layer, e), v in v_by_le.items():
        by_layer.setdefault(layer, []).append((e, v))

    def fn(model):
        for layer in sorted(by_layer):
            L = model.model.layers[layer]
            _, idx, w = L.mlp.experts.inputs[0]  # idx,w: (n_tokens, top_k)
            h = L.output  # (B, T, D) residual after this layer's MoE
            delta = torch.zeros(
                idx.shape[0], h.shape[-1], device=h.device, dtype=torch.float32
            )
            for e, v in by_layer[layer]:
                vt = v.to(h.device, torch.float32)
                gate_e = (w.float() * (idx == e)).sum(
                    dim=-1
                )  # (n_tokens,) gate of e, 0 if unrouted
                delta += gate_e.unsqueeze(-1) * (alpha * vt)
            h[:] = h + delta.to(h.dtype).reshape(h.shape)

    return fn


def generate(
    model, ids: list[int], max_new_tokens: int, intervention: Callable | None
) -> list[int]:
    """Greedy-generate exactly ``max_new_tokens`` continuation ids, optionally under ``intervention``.

    ``min_new_tokens == max_new_tokens`` forces a fixed length: under ``tracer.all()`` an
    early EOS would leave some intervention iterations "not provided" and nnsight 0.7 errors.
    """
    with (
        torch.no_grad(),
        model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            min_new_tokens=max_new_tokens,
            do_sample=False,
        ) as tracer,
    ):
        if intervention is not None:
            with tracer.all():
                intervention(model)
        out = model.generator.output.save()
    return out[0].tolist()[len(ids) :]


def concept_propensity(
    model,
    ids: list[int],
    cont: list[int],
    concept_ids: list[int],
    intervention: Callable | None,
) -> float:
    """Mean concept-logit score over the continuation, intervention active (one trace).

    The sensitive probe: how much the model elevates the concept's tokens along the text it
    actually produced. (EOS-safe fixed-length generation means this never interleaves a trace
    inside an active generate.)
    """
    full = ids + cont
    with torch.no_grad(), model.trace([full]):
        if intervention is not None:
            intervention(model)
        logits = model.output.logits.save()
    seq = (
        logits[0].float().cpu()
    )  # (T, V); positions len(ids)-1 .. T-2 predict the continuation
    cont_logits = seq[len(ids) - 1 : len(full) - 1]  # one row per continuation token
    return float(relative_logit_score(cont_logits, concept_ids).mean())


def concept_regex(words: list[str]) -> re.Pattern:
    """Whole-word, case-insensitive matcher for a concept lexicon."""
    return re.compile(
        r"\b("
        + "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
        + r")\b",
        re.I,
    )


def run_intervention_experiment(
    model,
    eliciting_prompts: list[list[int]],
    neutral_prompts: list[list[int]],
    concept_ids: list[int],
    methods: dict[str, Callable | None],
    *,
    concept_words: list[str] | None = None,
    max_new_tokens: int = 32,
    n_examples: int = 4,
) -> dict:
    """Generate + score every method on concept-eliciting and neutral prompts.

    ``methods`` maps a name to an intervention callable (``None`` = baseline). Returns, per
    method and prompt set, the mean concept-logit propensity (sensitive probe) and the
    concept-word rate in the generated text, plus a few example eliciting-prompt continuations.
    """
    tok = model.tokenizer
    pattern = concept_regex(concept_words or CONCEPT_WORDS["offensive"])
    results: dict[str, dict] = {}
    for name, intervention in methods.items():
        block: dict[str, object] = {}
        for setname, prompts in (
            ("eliciting", eliciting_prompts),
            ("neutral", neutral_prompts),
        ):
            print(f"  [{name} / {setname}] generating {len(prompts)} ...", flush=True)
            props, counts, distinct, examples = [], [], [], []
            for j, ids in enumerate(prompts):
                cont = generate(model, ids, max_new_tokens, intervention)
                props.append(
                    concept_propensity(model, ids, cont, concept_ids, intervention)
                )
                counts.append(len(pattern.findall(tok.decode(cont))))
                # distinct-1 ratio: degeneracy guard. Near 0 = the intervention broke
                # generation into a repeated token ("the the the"), so any propensity drop is
                # an artifact, not detox. Healthy text is ~0.6-0.9 here.
                distinct.append(len(set(cont)) / max(len(cont), 1))
                if setname == "eliciting" and j < n_examples:
                    examples.append(tok.decode(cont).strip())
            block[f"{setname}_propensity"] = float(sum(props) / max(len(props), 1))
            block[f"{setname}_word_frac"] = float(
                sum(c > 0 for c in counts) / max(len(counts), 1)
            )
            block[f"{setname}_distinct1"] = float(sum(distinct) / max(len(distinct), 1))
            if examples:
                block["examples"] = examples
        results[name] = block
    return results
