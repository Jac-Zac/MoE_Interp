"""Orchestrates the generation-time intervention experiment (the ``circuit-steer`` CLI).

Every intervention here is *expert-level* — it acts on the router gate or the experts' output
activation, never on the residual stream. For each concept it builds the causal (gate-AtP) and
correlational (SOMP / Expert Pursuit) expert sets plus a matched random control, then knocks them
out or steers their output through :func:`run_intervention_experiment`. Prompts default to a real
RealToxicityPrompts split (high- vs low-toxicity).
"""

from __future__ import annotations

import random

import numpy as np
import torch

from moe_interp.analysis.common import load_somp_results
from moe_interp.capture.model_adapter import get_model_adapter, model_num_experts
from moe_interp.circuit.intervene import (
    concept_propensity,
    expert_steer_intervention,
    generate,
    knockout_intervention,
    run_intervention_experiment,
)
from moe_interp.circuit.toxicity import right_padded
from moe_interp.config import get_model_dir, get_pursuit_dir
from moe_interp.grids import top_experts
from moe_interp.pursuit.concepts import CONCEPT_WORDS, build_toxic_token_ids


def _matched_random_set(
    reference: list[tuple[int, int]], n_experts: int, seed: int = 0
) -> list[tuple[int, int]]:
    """Random ``(layer, expert)`` set on the *same layers* as ``reference``, distinct experts.

    The specificity control: same routing depth/footprint as the causal set, so any effect that
    survives here is generic to perturbing *some* experts at those layers, not to *these* experts.
    """
    rng = random.Random(seed)
    used = set(reference)
    rand: list[tuple[int, int]] = []
    for layer, _ in reference:
        while (layer, e := rng.randrange(n_experts)) in used:
            pass
        used.add((layer, e))
        rand.append((layer, e))
    return rand


def somp_concept_experts_evr(
    model_name: str,
    dataset: str,
    concept: str,
    k: int,
    *,
    atom_k: int = 10,
) -> list[tuple[int, int]]:
    """Top-``k`` ``(layer, expert)`` by concept **EVR@atom_k** from the concept-restricted pursuit.

    The proper SOMP identification of the concept experts (this is what we actually analyse and
    report): the pursuit at ``pursuit/<dataset>/<concept>/`` runs the dictionary restricted to the
    concept lexicon, so every atom is already a concept word and EVR@k (the variance of the
    expert's activations explained by the first ``atom_k`` concept atoms) ranks how concept-
    specialised the expert is — cleaner than counting offensive tokens in a full-vocab pursuit.
    Returns ``[]`` if the concept pursuit is missing.
    """
    pursuit_dir = get_pursuit_dir(model_name, dataset, concept=concept)
    if not (pursuit_dir / "results.jsonl").exists():
        return []
    somp = load_somp_results(pursuit_dir)
    i = atom_k - 1
    scored = sorted(
        (
            (r["evr"][min(i, len(r["evr"]) - 1)], le)
            for le, r in somp.items()
            if r.get("evr")
        ),
        reverse=True,
    )
    return [le for _, le in scored[:k]]


def collect_expert_output_dom(
    model,
    adapter,
    toxic: list[list[int]],
    neutral: list[list[int]],
    experts_by_layer: dict[int, list[int]],
) -> dict[tuple[int, int], torch.Tensor]:
    """Per-expert diff-of-means in **expert-output space**: ``v_e = mean(out_e|tox) - mean(out_e|neu)``.

    ``out_e = adapter.expert_forward(experts, e, h_t)`` is the *raw* expert-MLP output that the model
    adds to the residual (scaled by the router gate) — the expert output itself, not the residual
    stream. We tap each layer's ``mlp.experts`` input (the MoE-block
    hidden states + routing) once per prompt, then recompute each named expert's output over the
    tokens routed to it and average over both populations. One prompt per trace (no padding mask
    needed). Returns ``{(layer, expert): v_e}`` for experts seen in both populations.
    """
    layers = sorted(experts_by_layer)

    def population_means(prompts):
        sums: dict[tuple[int, int], torch.Tensor] = {}
        counts: dict[tuple[int, int], int] = {}
        for ids in prompts:
            saved: dict[int, object] = {}
            with torch.no_grad(), model.trace([ids]):
                for layer in layers:
                    saved[layer] = model.model.layers[layer].mlp.experts.inputs.save()
            for layer in layers:
                h, idx, _ = saved[layer][0]  # h:(n_tok,D) idx:(n_tok,top_k)
                experts_mod = model.model.layers[layer].mlp.experts
                for e in experts_by_layer[layer]:
                    routed = (idx == e).any(dim=-1)
                    n = int(routed.sum())
                    if n == 0:
                        continue
                    out_e = adapter.expert_forward(experts_mod, e, h[routed].float())
                    s = out_e.sum(dim=0).cpu()
                    le = (layer, e)
                    sums[le] = sums.get(le, torch.zeros_like(s)) + s
                    counts[le] = counts.get(le, 0) + n
        return {le: sums[le] / counts[le] for le in sums}

    mt, mn = population_means(toxic), population_means(neutral)
    return {le: mt[le] - mn[le] for le in mt if le in mn}


def _causal_grid_set(path, k: int) -> list[tuple[int, int]] | None:
    """Top-``k`` promoter experts from a cached effect grid (signed: most toxicity-promoting)."""
    if not path.exists():
        return None
    grid = np.nan_to_num(np.load(path))
    return [(layer, e) for layer, e, _ in top_experts(grid, k, by="signed")]


def expert_intervention_sets(
    model,
    model_name: str,
    eliciting,
    *,
    concept: str,
    dataset: str,
    k: int,
    atp_grid_path=None,
) -> dict[str, list[tuple[int, int]]]:
    """The expert sets compared by :func:`run_expert_steer`: SOMP, AtP, + random.

    SOMP is the concept-restricted EVR@k pursuit (token-association). AtP is the *causal* top-``k``
    promoters from the gate-AtP grid (the experts whose ablation most lowers the concept logit) —
    the experts this experiment most wants to steer. ``random`` is matched to the AtP causal set's
    layers (falling back to SOMP) so the specificity control shares the causal set's routing depth.

    ``atp_grid_path`` overrides the AtP grid location (e.g. a concept-specific ``atp_<concept>``
    grid); it defaults to the toxicity grid keyed by train size. Missing grids are skipped.
    """
    ne = model_num_experts(model)
    md = get_model_dir(model_name)
    somp = somp_concept_experts_evr(model_name, dataset, concept, k)
    if not somp:
        raise RuntimeError(
            f"No concept pursuit at {get_pursuit_dir(model_name, dataset, concept=concept)}; "
            "run the concept-restricted pursuit first."
        )
    sets: dict[str, list[tuple[int, int]]] = {"SOMP": somp}
    if atp_grid_path is None:
        atp_grid_path = (
            md / "circuit" / "attribution" / f"atp_grid_n{len(eliciting)}.npy"
        )
    atp = _causal_grid_set(atp_grid_path, k)
    if atp:
        sets["AtP"] = atp
    sets["random"] = _matched_random_set(sets.get("AtP", somp), ne)
    return sets


def _sets_and_dom(
    model,
    model_name: str,
    *,
    concept: str,
    dataset: str,
    k: int,
    train: tuple[list[list[int]], list[list[int]]],
    atp_grid_path=None,
) -> tuple[dict[str, list[tuple[int, int]]], dict[tuple[int, int], torch.Tensor]]:
    """Shared preamble of the two intervention runners: selector sets + per-expert DoM.

    Picks each selector's experts (SOMP / AtP / random), then collects every named expert's
    toxic−neutral diff-of-means in expert-output space on the *train* prompts. Returns
    ``(sets, dom)``.
    """
    eliciting, neutral = train
    adapter = get_model_adapter(model)
    sets = expert_intervention_sets(
        model,
        model_name,
        eliciting,
        concept=concept,
        dataset=dataset,
        k=k,
        atp_grid_path=atp_grid_path,
    )
    by_layer_all: dict[int, list[int]] = {}
    for experts in sets.values():
        for layer, e in experts:
            by_layer_all.setdefault(layer, []).append(e)
    with right_padded(model):
        dom = collect_expert_output_dom(
            model, adapter, eliciting, neutral, by_layer_all
        )
    return sets, dom


def run_expert_steer(
    model,
    model_name: str,
    *,
    concept: str,
    dataset: str,
    k: int,
    batch_size: int,
    max_new_tokens: int,
    train: tuple[list[list[int]], list[list[int]]],
    test: tuple[list[list[int]], list[list[int]]],
    alphas: tuple[float, ...] = (5.0, -5.0, -10.0),
    atp_grid_path=None,
) -> dict:
    """Expert-level causal interventions on each identifier's experts (SOMP / AtP) vs random.

    Runs the two interventions at *expert granularity*, on **every** expert selector — the concept
    SOMP set (token-association), the AtP *causal* set (top-``k`` promoters from the gate-AtP grid),
    and a matched random control — so we can see whether steering the *causally-identified* experts
    works where steering the SOMP ones did not:

    * **knockout** — zero the router gate (α=0; near-inert under top-k redundancy)
    * **esteer(α)** — add ``α·v_e`` to each expert's *output* (``v_e`` = toxic−neutral diff-of-means
      in expert-output space). ``α<0`` subtracts the toxic direction (detox), ``α>0`` amplifies it
      (toxicity should rise — the causal sanity check). The expert-output DoM is tiny relative to the
      residual and gate-weighted, so α is swept to large magnitudes (±5, −10) to give the steer real
      authority while the distinct-1 guard catches any degeneration.

    Directions ``v_e`` are estimated on the *train* prompts; every method is scored on the held-out
    *test* split, including the **neutral collateral** and a **distinct-1 degeneracy** guard (a
    propensity drop only counts if the text stays coherent, i.e. distinct-1 not collapsed).
    """
    eliciting_eval, neutral_eval = test
    concept_words = CONCEPT_WORDS[concept]
    concept_ids = build_toxic_token_ids(model.tokenizer, concept_words)

    sets, dom = _sets_and_dom(
        model,
        model_name,
        concept=concept,
        dataset=dataset,
        k=k,
        train=train,
        atp_grid_path=atp_grid_path,
    )

    methods: dict = {"baseline": None}
    for name, experts in sets.items():
        methods[f"knockout-{name}"] = knockout_intervention(experts)
        v_set = {le: dom[le] for le in experts if le in dom}
        for a in alphas:
            methods[f"esteer({a:+g})-{name}"] = expert_steer_intervention(v_set, a)

    results = run_intervention_experiment(
        model,
        eliciting_eval,
        neutral_eval,
        concept_ids,
        methods,
        concept_words=concept_words,
        max_new_tokens=max_new_tokens,
    )
    return {
        "methods": results,
        "meta": {
            "concept": concept,
            "dataset": dataset,
            "k": k,
            "alphas": list(alphas),
            "n_train": len(train[0]),
            "n_test": len(eliciting_eval),
            "max_new_tokens": max_new_tokens,
            "sets": sets,
            "dom_norms": {f"{l}.{e}": float(v.norm()) for (l, e), v in dom.items()},
        },
    }


def run_dose_response(
    model,
    model_name: str,
    *,
    concept: str,
    dataset: str,
    k: int,
    max_new_tokens: int,
    train: tuple[list[list[int]], list[list[int]]],
    test: tuple[list[list[int]], list[list[int]]],
    alpha: float = -5.0,
    ks: tuple[int, ...] = (1, 5, 10, 15),
    atp_grid_path=None,
) -> dict:
    """Cumulative dose-response: toxic propensity vs #experts intervened, per selector vs random.

    For each budget ``j`` in ``ks`` we α-steer the top-``j`` experts of each
    selector (SOMP / AtP / random), scoring *eliciting* propensity only (the curve we
    care about). Shows where — if anywhere — a causal signal emerges and whether the causal set
    separates from random as the budget grows. Cheap: reuses the same ``v_e`` for every budget.
    """
    eliciting_eval = test[0]
    concept_ids = build_toxic_token_ids(model.tokenizer, CONCEPT_WORDS[concept])

    sets, dom = _sets_and_dom(
        model,
        model_name,
        concept=concept,
        dataset=dataset,
        k=k,
        train=train,
        atp_grid_path=atp_grid_path,
    )

    def elic_prop(intervention) -> float:
        props = [
            concept_propensity(
                model,
                ids,
                generate(model, ids, max_new_tokens, intervention),
                concept_ids,
                intervention,
            )
            for ids in eliciting_eval
        ]
        return float(sum(props) / max(len(props), 1))

    base = elic_prop(None)
    # α-steer is the detox arm — the only intervention that cleanly separated the causal (AtP)
    # set from random while staying coherent (plain gate-zero knockout was inert: top-k redundancy).
    curves: dict[str, dict[str, list]] = {
        f"esteer({alpha:+g})": {name: [] for name in sets},
    }
    for name, experts in sets.items():
        for j in ks:
            sub = experts[:j]
            v_sub = {le: dom[le] for le in sub if le in dom}
            curves[f"esteer({alpha:+g})"][name].append(
                {"k": j, "prop": elic_prop(expert_steer_intervention(v_sub, alpha))}
            )
    return {
        "baseline_prop": base,
        "curves": curves,
        "meta": {
            "concept": concept,
            "dataset": dataset,
            "alpha": alpha,
            "ks": list(ks),
            "n_test": len(eliciting_eval),
        },
    }
