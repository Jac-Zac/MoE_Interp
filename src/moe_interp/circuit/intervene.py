"""Generation-time gate interventions to suppress a concept — the causal capstone.

Gate-AtP identifies which experts *cause* the concept. Here we act on those experts (and no
other locus — every intervention is expert-level, never on the residual stream): during greedy
generation we scale the selected experts' router gate, either zeroing it (**knockout**) or
multiplying it by a factor (**downweighting**), and measure the change in the concept propensity
versus the un-intervened baseline.

A concept is scored three ways: the mean **concept-logit propensity** over the generated
continuation (the sensitive probe), a lexical **concept-word rate** in the decoded text, and a
**distinct-1** coherence guard. Lower propensity/word-rate = less of the concept — but only if
distinct-1 stays healthy (≈0.6–0.9); a drop with collapsed distinct-1 is degraded text, not
removal.
"""

from __future__ import annotations

import re
from collections.abc import Callable

import torch

from moe_interp.circuit.concept_probe import relative_logit_score


def gate_scale_intervention(experts: list[tuple[int, int]], scale: float) -> Callable:
    """Scale the router gate of each ``(layer, expert)`` by ``scale``.

    ``scale=0.0`` zeros the gate (**knockout** — full removal); ``scale=0.9`` is a 10%
    downweight, ``scale=0.5`` a 50% downweight. The expert's contribution
    to the residual is ``gate_{t,e} * f_e(h_t)``, so scaling the live gate by ``scale`` scales that
    contribution at exactly the tokens routed to ``e`` and leaves every other expert untouched.
    """
    by_layer: dict[int, list[int]] = {}
    for layer, e in experts:
        by_layer.setdefault(layer, []).append(e)

    def fn(model):
        for layer in sorted(by_layer):  # forward order for nnsight 0.7
            _, idx, w = model.model.layers[layer].mlp.experts.inputs[0]
            for e in by_layer[layer]:
                w[idx == e] *= scale

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
