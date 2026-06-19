"""Generation-time interventions to suppress toxicity — the causal capstone.

The patching grid + gate-AtP identify which experts *cause* toxic continuations. Here we
act on that: during greedy generation we either **knock out** the top gate-AtP promoter
experts (zero their router gate) or **steer** against the diff-of-means toxic direction,
and measure the drop in toxic propensity versus the un-intervened baseline. Two controls
keep it honest: a **random** expert knockout (specificity — does it have to be *these*
experts?) and a **neutral** prompt set (collateral — does the intervention break ordinary
generation?).

Toxicity is scored two ways: the mean **toxic-logit propensity** over the generated
continuation (the sensitive probe used throughout the study, re-evaluated with the
intervention active) and a lexical **offensive-word rate** in the decoded text (the literal
output). Lower = less toxic on both. Generated text is kept for qualitative inspection.
"""

from __future__ import annotations

import re
from collections.abc import Callable

import torch

from moe_interp.circuit.toxicity import toxic_logit_score
from moe_interp.pursuit.concepts import CONCEPT_WORDS


def knockout_intervention(experts: list[tuple[int, int]]) -> Callable:
    """Intervention that zeros the router gate of each ``(layer, expert)`` (forward order)."""
    by_layer: dict[int, list[int]] = {}
    for layer, e in experts:
        by_layer.setdefault(layer, []).append(e)

    def fn(model):
        for layer in sorted(by_layer):  # nnsight 0.7 needs envoys touched in forward order
            _, idx, w = model.model.layers[layer].mlp.experts.inputs[0]
            for e in by_layer[layer]:
                w[idx == e] = 0.0

    return fn


def downweight_intervention(experts: list[tuple[int, int]], factor: float) -> Callable:
    """Less-drastic knockout: scale the router gate of each ``(layer, expert)`` by ``factor``."""
    by_layer: dict[int, list[int]] = {}
    for layer, e in experts:
        by_layer.setdefault(layer, []).append(e)

    def fn(model):
        for layer in sorted(by_layer):
            _, idx, w = model.model.layers[layer].mlp.experts.inputs[0]
            for e in by_layer[layer]:
                w[idx == e] = w[idx == e] * factor

    return fn


def steer_intervention(layer: int, v: torch.Tensor, alpha: float) -> Callable:
    """Intervention that adds ``alpha * v`` to the residual stream at ``layer``.

    A large fixed ``alpha`` over-steers and degrades all generation; prefer
    :func:`projectout_intervention` for a non-destructive "remove toxicity" edit.
    """

    def fn(model):
        h = model.model.layers[layer].output
        h[:] = h + alpha * v.to(h.device, h.dtype)

    return fn


def projectout_intervention(layer: int, v: torch.Tensor) -> Callable:
    """Remove the ``v`` component from the residual stream at ``layer`` (ablate the direction).

    Far gentler than additive steering: it only zeroes the projection onto the (unit) toxic
    direction, leaving every orthogonal feature untouched, so neutral generation is preserved.
    """

    def fn(model):
        h = model.model.layers[layer].output
        vhat = torch.nn.functional.normalize(v.to(h.device, h.dtype), dim=0)
        h[:] = h - (h @ vhat).unsqueeze(-1) * vhat

    return fn


def generate(model, ids: list[int], max_new_tokens: int, intervention: Callable | None) -> list[int]:
    """Greedy-generate exactly ``max_new_tokens`` continuation ids, optionally under ``intervention``.

    ``min_new_tokens == max_new_tokens`` forces a fixed length: under ``tracer.all()`` an
    early EOS would leave some intervention iterations "not provided" and nnsight 0.7 errors.
    """
    with torch.no_grad(), model.generate(
        ids, max_new_tokens=max_new_tokens, min_new_tokens=max_new_tokens, do_sample=False
    ) as tracer:
        if intervention is not None:
            with tracer.all():
                intervention(model)
        out = model.generator.output.save()
    return out[0].tolist()[len(ids):]


def toxic_propensity(
    model, ids: list[int], cont: list[int], toxic_ids: list[int], intervention: Callable | None
) -> float:
    """Mean toxic-logit score over the continuation, intervention active (one trace).

    The sensitive probe: how much the model elevates toxic tokens along the text it actually
    produced. (EOS-safe fixed-length generation means this never interleaves a trace inside
    an active generate.)
    """
    full = ids + cont
    with torch.no_grad(), model.trace([full]):
        if intervention is not None:
            intervention(model)
        logits = model.output.logits.save()
    seq = logits[0].float().cpu()  # (T, V); positions len(ids)-1 .. T-2 predict the continuation
    pos = range(len(ids) - 1, len(full) - 1)
    return float(
        sum(float(toxic_logit_score(seq[p : p + 1], toxic_ids)) for p in pos) / max(len(pos), 1)
    )


def _offensive_regex() -> re.Pattern:
    """Whole-word, case-insensitive matcher for the offensive lexicon."""
    words = sorted(CONCEPT_WORDS["offensive"], key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(re.escape(w) for w in words) + r")\b", re.I)


def run_intervention_experiment(
    model,
    toxic_prompts: list[list[int]],
    neutral_prompts: list[list[int]],
    toxic_ids: list[int],
    methods: dict[str, Callable | None],
    *,
    max_new_tokens: int = 32,
    n_examples: int = 4,
) -> dict:
    """Generate + score every method on toxic and neutral prompts.

    ``methods`` maps a name to an intervention callable (``None`` = baseline). Returns, per
    method and prompt set, the mean toxic-logit propensity (sensitive probe) and the
    offensive-word rate in the generated text, plus a few example toxic-prompt continuations.
    """
    tok = model.tokenizer
    pattern = _offensive_regex()
    results: dict[str, dict] = {}
    for name, intervention in methods.items():
        block: dict[str, object] = {}
        for setname, prompts in (("toxic", toxic_prompts), ("neutral", neutral_prompts)):
            print(f"  [{name} / {setname}] generating {len(prompts)} ...", flush=True)
            props, counts, examples = [], [], []
            for j, ids in enumerate(prompts):
                cont = generate(model, ids, max_new_tokens, intervention)
                props.append(toxic_propensity(model, ids, cont, toxic_ids, intervention))
                counts.append(len(pattern.findall(tok.decode(cont))))
                if setname == "toxic" and j < n_examples:
                    examples.append(tok.decode(cont).strip())
            block[f"{setname}_propensity"] = float(sum(props) / max(len(props), 1))
            block[f"{setname}_toxic_frac"] = float(sum(c > 0 for c in counts) / max(len(counts), 1))
            if examples:
                block["examples"] = examples
        results[name] = block
    return results
