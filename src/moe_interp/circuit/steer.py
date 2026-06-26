"""Orchestrates the generation-time intervention experiment (the ``circuit-steer`` CLI).

Builds the set of methods to compare — baseline, the causal/correlational expert
knockouts, a matched random control, and the project-out direction edit — then runs them
all through :func:`run_intervention_experiment`. The expert sets come from the artifacts
produced by the other ``circuit`` commands (gate-AtP, the patching grid) and the SOMP
results. Prompts default to a real RealToxicityPrompts split (high- vs low-toxicity).
"""

from __future__ import annotations

import random

import numpy as np
import torch

from moe_interp.analysis.common import load_somp_results
from moe_interp.capture.model_adapter import model_num_experts
from moe_interp.circuit.attribution import gate_attribution
from moe_interp.circuit.intervene import (
    compose_interventions,
    knockout_intervention,
    localized_projectout_intervention,
    localized_steer_intervention,
    projectout_intervention,
    run_intervention_experiment,
    steer_intervention,
)
from moe_interp.circuit.prompts import rtp_split
from moe_interp.circuit.toxicity import right_padded
from moe_interp.config import get_model_dir, get_pursuit_dir
from moe_interp.grids import top_experts
from moe_interp.pursuit.concepts import CONCEPT_WORDS, build_toxic_token_ids


def _group_by_layer(experts: list[tuple[int, int]]) -> dict[int, list[int]]:
    """Regroup a ``[(layer, expert), ...]`` list into ``{layer: [experts]}``."""
    by_layer: dict[int, list[int]] = {}
    for layer, e in experts:
        by_layer.setdefault(layer, []).append(e)
    return by_layer


def somp_concept_experts(
    model_name: str,
    concept_words: list[str],
    k: int,
    *,
    source: str = "pile10k",
) -> list[tuple[int, int]]:
    """Top-``k`` ``(layer, expert)`` whose SOMP atoms most overlap the concept lexicon.

    The correlational (no-model) selector: ranks experts by how many of their pursuit atoms are
    concept words. Concept-general — pass any ``concept_words`` list (offensive, numbers,
    countries …). Returns ``[]`` if the pursuit results are missing. Shared by the toxicity
    intervention pipeline (:func:`_offensive_expert_sets`) and counterfactual editing.
    """
    pursuit_dir = get_pursuit_dir(model_name, source)
    if not (pursuit_dir / "results.jsonl").exists():
        return []
    lex = {w.lower() for w in concept_words}
    somp = load_somp_results(pursuit_dir)
    scored = sorted(
        (
            (sum(t.strip().lower() in lex for t in r.get("tokens", [])), le)
            for le, r in somp.items()
        ),
        reverse=True,
    )
    return [le for s, le in scored[:k] if s > 0]


def collect_last_token_residuals(
    model, prompts: list[list[int]], layer: int, batch_size: int = 8
) -> torch.Tensor:
    """Residual stream at ``layer`` output, gathered at each prompt's last real token.

    Used for the diff-of-means toxic *direction*: ``v = mean(h | toxic) - mean(h |
    neutral)`` (see :func:`run_steer`). Toxicity is thus isolated as a direction in the
    residual stream rather than a single expert.
    """
    chunks: list[torch.Tensor] = []
    with right_padded(model):
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            lengths = torch.tensor([len(t) for t in batch])
            with torch.no_grad(), model.trace(batch):
                # decoder layer .output is the bare hidden-states tensor (B, T, D)
                hs = model.model.layers[layer].output.save()
            rows = torch.arange(hs.shape[0])
            chunks.append(hs[rows, lengths - 1].float().cpu())
    return torch.cat(chunks)


def _offensive_expert_sets(
    model,
    model_name: str,
    eliciting,
    concept_ids,
    concept_words,
    *,
    k: int,
    batch_size: int,
) -> dict[str, list[tuple[int, int]]]:
    """Top-``k`` (layer, expert) sets from each identifier, plus a matched random control."""
    ne = model_num_experts(model)
    md = get_model_dir(model_name)

    def topk_grid(grid: np.ndarray) -> list[tuple[int, int]]:
        return [(layer, e) for layer, e, _ in top_experts(grid, k, by="signed")]

    # gate-AtP over the seed prompts: cache the grid so reruns (e.g. a different
    # --knockout_k) skip the backward pass. Keyed by train-set size — a different
    # N_PROMPTS identifies the grid on a different prompt prefix, so it must never reuse
    # a stale cache (the patching grid it is compared against is on the same prefix).
    atp_path = (
        md / "circuit" / "attribution" / f"atp_grid_n{len(eliciting)}.npy"
    )
    if atp_path.exists():
        atp = np.load(atp_path)
    else:
        atp = gate_attribution(
            model, eliciting, concept_ids, batch_size=batch_size
        ).numpy()
        atp_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(atp_path, atp)
    sets: dict[str, list[tuple[int, int]]] = {"AtP": topk_grid(atp)}
    patch_path = md / "circuit" / "patching" / "patching_grid.npy"
    if patch_path.exists():
        sets["patching"] = topk_grid(np.nan_to_num(np.load(patch_path)))

    somp_set = somp_concept_experts(model_name, concept_words, k)
    if somp_set:
        sets["SOMP"] = somp_set

    # Specificity control: same layers as AtP, but random (distinct) experts.
    rng = random.Random(0)
    used = set(sets["AtP"])
    rand: list[tuple[int, int]] = []
    for ly, _ in sets["AtP"]:
        while (ly, e := rng.randrange(ne)) in used:
            pass
        used.add((ly, e))
        rand.append((ly, e))
    sets["random"] = rand
    return sets


def run_steer(
    model,
    model_name: str,
    *,
    concept: str,
    knockout_k: int,
    steer_layer: int,
    batch_size: int,
    max_new_tokens: int,
    train: tuple[list[list[int]], list[list[int]]] | None = None,
    test: tuple[list[list[int]], list[list[int]]] | None = None,
) -> dict:
    """Build the intervention methods and run the generation experiment.

    Experts and the diff-of-means direction are identified on the *train* eliciting/neutral
    prompts; every method is then scored on the held-out *test* prompts, so the comparison is
    out-of-sample. ``train`` / ``test`` are ``(eliciting, neutral)`` id-list pairs; if omitted
    they default to a disjoint RealToxicityPrompts split (high- vs low-toxicity). Returns
    ``{"methods": <per-method scores>, "meta": {...}}``; ``meta.sets`` records the
    knocked-out expert sets.
    """
    if train is None or test is None:
        elic_tr, elic_te, neut_tr, neut_te = rtp_split(model.tokenizer)
        train, test = (elic_tr, neut_tr), (elic_te, neut_te)
    eliciting, neutral = train
    eliciting_eval, neutral_eval = test
    concept_words = CONCEPT_WORDS[concept]
    concept_ids = build_toxic_token_ids(model.tokenizer, concept_words)

    methods: dict = {"baseline": None}
    sets = _offensive_expert_sets(
        model,
        model_name,
        eliciting,
        concept_ids,
        concept_words,
        k=knockout_k,
        batch_size=batch_size,
    )
    for name, experts in sets.items():
        methods[f"{name}-knockout"] = knockout_intervention(experts)
    # diff-of-means direction (validated for toxicity)
    steer_dir = collect_last_token_residuals(model, eliciting, steer_layer).mean(0) - (
        collect_last_token_residuals(model, neutral, steer_layer).mean(0)
    )

    methods[f"projectout@L{steer_layer}"] = projectout_intervention(
        steer_layer, steer_dir
    )
    methods[f"steer@L{steer_layer}(α=-1)"] = steer_intervention(
        steer_layer, steer_dir, alpha=-1.0
    )

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
            "k": knockout_k,
            "steer_layer": steer_layer,
            "n_train": len(eliciting),
            "n_test": len(eliciting_eval),
            "sets": sets,
        },
    }


def run_localized_steer(
    model,
    *,
    concept: str,
    sets: dict[str, list[tuple[int, int]]],
    train: tuple[list[list[int]], list[list[int]]],
    test: tuple[list[list[int]], list[list[int]]],
    steer_layer: int,
    batch_size: int,
    max_new_tokens: int,
    selectors: tuple[str, ...] = ("AtP", "patching", "SOMP", "random"),
    steer_alpha: float | None = None,
) -> dict:
    """Localized project-out (and optional localized steering) at the selected experts.

    The question this answers: is the distributed toxic direction *carried at* the positions
    routed to the causally-identified experts? For each selector we project the diff-of-means
    direction out of the residual only at positions routed to that selector's experts (see
    :func:`~moe_interp.circuit.intervene.localized_projectout_intervention`), versus a global
    single-layer project-out control. The decisive comparison is *between routed variants*
    (``routed-patching`` / ``routed-AtP`` vs ``routed-SOMP`` / ``routed-random``): same
    machinery, different selector, so it isolates the value of causal vs correlational
    localization. The single-layer global edit is only a rough reference (it touches one layer
    while routed variants span every layer in their set).

    ``sets`` is the per-selector ``{name: [(layer, expert), ...]}`` mapping from
    :func:`run_steer`'s ``meta.sets``; a routed variant is built for each selector present.
    Directions are identified on the *train* residuals; every method (and the **neutral
    collateral check**) is scored on the held-out *test* prompts. A real drop must suppress the
    eliciting prompts *more* than the neutral ones — compare ``eliciting`` vs ``neutral``
    propensity, not the eliciting drop alone.

    If ``steer_alpha`` is set (e.g. ``-1.0``), also add additive **steering** arms — a global
    ``global-steer`` control plus one ``routed-steer-<selector>`` per selector — that *add*
    ``alpha·unit(v)`` at the routed positions instead of projecting it out. This is the direct
    Head-Pursuit comparison: zeroing/knockout is near-inert (redundancy), so the question is
    whether ``alpha=-1`` steering *localized to the causal experts* suppresses toxicity while
    staying specific (neutral preserved), unlike the blunt whole-residual steer.
    Returns ``{"methods": ..., "meta": {...}}``.
    """
    eliciting, neutral = train
    eliciting_eval, neutral_eval = test
    concept_words = CONCEPT_WORDS[concept]
    concept_ids = build_toxic_token_ids(model.tokenizer, concept_words)

    routed = {k: sets[k] for k in selectors if k in sets}
    # Per-layer diff-of-means toxic direction for the control layer + every routed layer.
    needed = {steer_layer} | {layer for s in routed.values() for layer, _ in s}
    dirs = {
        layer: collect_last_token_residuals(model, eliciting, layer, batch_size).mean(0)
        - collect_last_token_residuals(model, neutral, layer, batch_size).mean(0)
        for layer in sorted(needed)
    }

    methods: dict = {
        "baseline": None,
        f"global-projectout@L{steer_layer}": projectout_intervention(
            steer_layer, dirs[steer_layer]
        ),
    }
    routed_by_layer = {
        name: _group_by_layer(experts) for name, experts in routed.items()
    }
    for name, by_layer in routed_by_layer.items():
        methods[f"routed-{name}"] = compose_interventions(
            [
                localized_projectout_intervention(layer, dirs[layer], elist)
                for layer, elist in sorted(by_layer.items())
            ]
        )

    if steer_alpha is not None:
        methods[f"global-steer@L{steer_layer}(α={steer_alpha:g})"] = steer_intervention(
            steer_layer, dirs[steer_layer], alpha=steer_alpha
        )
        for name, by_layer in routed_by_layer.items():
            methods[f"routed-steer-{name}(α={steer_alpha:g})"] = compose_interventions(
                [
                    localized_steer_intervention(
                        layer, dirs[layer], elist, alpha=steer_alpha
                    )
                    for layer, elist in sorted(by_layer.items())
                ]
            )

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
            "steer_layer": steer_layer,
            "n_train": len(eliciting),
            "n_test": len(eliciting_eval),
            "max_new_tokens": max_new_tokens,
            "steer_alpha": steer_alpha,
            "routed_sets": routed,
        },
    }
