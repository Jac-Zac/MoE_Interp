"""Generation-time interventions to suppress toxicity — the causal capstone.

The patching grid + gate-AtP identify which experts *cause* toxic continuations. Here we
act on that: during greedy generation we either **knock out** the top gate-AtP promoter
experts (zero their router gate) or **project out** the diff-of-means toxic direction from
the residual stream, and measure the drop in toxic propensity versus the un-intervened
baseline. Two controls
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


def localized_projectout_intervention(
    layer: int, v: torch.Tensor, experts: list[int]
) -> Callable:
    """Project ``v`` out of the residual at ``layer`` *only at positions routed to ``experts``*.

    Per-expert localized variant of :func:`projectout_intervention`: instead of scrubbing the
    toxic direction from every position, it scrubs it only where the named (toxic) experts
    actually fired, using the router's ``top_k_index`` to build the position mask. If the
    localized edit recovers the global project-out effect, toxicity is carried by a few experts;
    if it does not, the direction is genuinely distributed. ``experts`` are the expert ids *at
    this layer* (empty -> no-op).
    """

    def fn(model):
        if not experts:
            return
        L = model.model.layers[layer]
        _, idx, _ = L.mlp.experts.inputs[0]  # idx: (n_tokens, top_k) router assignments
        h = L.output  # (B, T, D) residual stream at this layer
        vhat = torch.nn.functional.normalize(v.to(h.device, h.dtype), dim=0)
        fired = torch.zeros(idx.shape[0], dtype=torch.bool, device=idx.device)
        for e in experts:
            fired |= (idx == e).any(dim=-1)
        # align the (n_tokens,) routing mask to the (B, T) residual positions
        mask = fired.to(h.dtype).reshape(h.shape[:-1]).unsqueeze(-1)
        h[:] = h - mask * (h @ vhat).unsqueeze(-1) * vhat

    return fn


def compose_interventions(fns: list[Callable]) -> Callable:
    """Chain several interventions into one callable (applied in forward-layer order)."""

    def fn(model):
        for f in fns:
            f(model)

    return fn


def steer_intervention(layer: int, v: torch.Tensor, alpha: float = -1.0) -> Callable:
    """Add ``alpha * unit(v)`` to every token position in the residual stream at ``layer``.

    Additive steering (CAA-style): alpha < 0 steers away from the direction, alpha > 0
    amplifies it. Unlike project-out this shifts the whole residual rather than only
    removing its projection, so the effect is stronger but less surgical.
    """

    def fn(model):
        h = model.model.layers[layer].output
        vhat = torch.nn.functional.normalize(v.to(h.device, h.dtype), dim=0)
        h[:] = h + alpha * vhat

    return fn


def localized_steer_intervention(
    layer: int, v: torch.Tensor, experts: list[int], alpha: float = -1.0
) -> Callable:
    """Add ``alpha * unit(v)`` to the residual at ``layer`` *only at positions routed to ``experts``*.

    The additive (CAA / Head-Pursuit-style) analogue of
    :func:`localized_projectout_intervention`, and the expert-group version of
    :func:`steer_intervention`: it shifts the residual along the diff-of-means direction only
    where the named experts fired, rather than scrubbing the existing projection. This is the
    direct test of whether *steering the identified experts* (not the whole residual) reproduces
    the Head-Pursuit effect — where zeroing a component is near-inert but ``alpha = -1`` steering
    suppresses the target behaviour. ``experts`` are the expert ids at this layer (empty -> no-op).
    """

    def fn(model):
        if not experts:
            return
        L = model.model.layers[layer]
        _, idx, _ = L.mlp.experts.inputs[0]
        h = L.output
        vhat = torch.nn.functional.normalize(v.to(h.device, h.dtype), dim=0)
        fired = torch.zeros(idx.shape[0], dtype=torch.bool, device=idx.device)
        for e in experts:
            fired |= (idx == e).any(dim=-1)
        mask = fired.to(h.dtype).reshape(h.shape[:-1]).unsqueeze(-1)
        h[:] = h + mask * alpha * vhat

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
            props, counts, examples = [], [], []
            for j, ids in enumerate(prompts):
                cont = generate(model, ids, max_new_tokens, intervention)
                props.append(
                    concept_propensity(model, ids, cont, concept_ids, intervention)
                )
                counts.append(len(pattern.findall(tok.decode(cont))))
                if setname == "eliciting" and j < n_examples:
                    examples.append(tok.decode(cont).strip())
            block[f"{setname}_propensity"] = float(sum(props) / max(len(props), 1))
            block[f"{setname}_word_frac"] = float(
                sum(c > 0 for c in counts) / max(len(counts), 1)
            )
            if examples:
                block["examples"] = examples
        results[name] = block
    return results
