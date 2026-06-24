"""Orchestrates the generation-time intervention experiment (the ``circuit-steer`` CLI).

Builds the set of methods to compare — baseline, the causal/correlational expert
knockouts, a matched random control, and the project-out direction edit — then runs them
all through :func:`run_intervention_experiment`. For the ``offensive`` concept the expert
sets come from the artifacts produced by the other ``circuit`` commands (gate-AtP, the
patching grid, DLA) and the SOMP results; other concepts only get the generic project-out
of the unembedding concept direction, because the seed prompts only reliably elicit
toxicity.
"""

from __future__ import annotations

import random

import numpy as np

from moe_interp.analysis.common import load_somp_results
from moe_interp.capture.cache import load_unembedding
from moe_interp.circuit.attribution import gate_attribution
from moe_interp.circuit.direction import collect_last_token_residuals
from moe_interp.circuit.intervene import (
    knockout_intervention,
    projectout_intervention,
    run_intervention_experiment,
)
from moe_interp.circuit.prompts import default_prompts
from moe_interp.config import get_model_dir, get_pursuit_dir, get_unembedding_dir
from moe_interp.grids import top_experts
from moe_interp.pursuit.concepts import CONCEPT_WORDS, build_toxic_token_ids


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
    ne = model.config.num_local_experts
    md = get_model_dir(model_name)

    def topk_grid(grid: np.ndarray) -> list[tuple[int, int]]:
        return [(layer, e) for layer, e, _ in top_experts(grid, k, by="signed")]

    # gate-AtP over the seed prompts: cache the grid so reruns (e.g. a different
    # --knockout_k) skip the backward pass. Keyed under the model dir, so it's the
    # offensive/seeds grid this command always builds.
    atp_path = md / "circuit" / "attribution" / "atp_grid.npy"
    if atp_path.exists():
        atp = np.load(atp_path)
    else:
        atp = gate_attribution(
            model, eliciting, concept_ids, batch_size=batch_size
        ).numpy()
        atp_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(atp_path, atp)
    sets: dict[str, list[tuple[int, int]]] = {"AtP": topk_grid(atp)}
    for name, rel in [
        ("patching", "patching/patching_grid.npy"),
        ("DLA", "dla/pile10k/dla_grid.npy"),
    ]:
        p = md / "circuit" / rel
        if p.exists():
            sets[name] = topk_grid(np.nan_to_num(np.load(p)))

    pursuit_dir = get_pursuit_dir(model_name, "pile10k")
    if (pursuit_dir / "results.jsonl").exists():
        lex = {w.lower() for w in concept_words}
        somp = load_somp_results(pursuit_dir)
        scored = sorted(
            (
                (sum(t.strip().lower() in lex for t in r.get("tokens", [])), le)
                for le, r in somp.items()
            ),
            reverse=True,
        )
        sets["SOMP"] = [le for s, le in scored[:k] if s > 0]

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
) -> dict:
    """Build the intervention methods and run the generation experiment.

    Returns ``{"methods": <per-method scores>, "meta": {...}}``; ``meta.sets`` records the
    knocked-out expert sets (empty for non-``offensive`` concepts).
    """
    eliciting, neutral = default_prompts(model.tokenizer)
    concept_words = CONCEPT_WORDS[concept]
    concept_ids = build_toxic_token_ids(model.tokenizer, concept_words)

    methods: dict = {"baseline": None}
    meta_sets: dict = {}
    if concept == "offensive":
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
        meta_sets = sets
        # diff-of-means direction (validated for toxicity)
        steer_dir = collect_last_token_residuals(model, eliciting, steer_layer).mean(
            0
        ) - (collect_last_token_residuals(model, neutral, steer_layer).mean(0))
    else:
        # Generic concepts: project out the unembedding concept direction.
        U = load_unembedding(get_unembedding_dir(model_name) / "dictionary.h5").float()
        steer_dir = U[concept_ids].mean(0) - U.mean(0)

    methods[f"projectout@L{steer_layer}"] = projectout_intervention(
        steer_layer, steer_dir
    )

    results = run_intervention_experiment(
        model,
        eliciting,
        neutral,
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
            "sets": meta_sets,
        },
    }
