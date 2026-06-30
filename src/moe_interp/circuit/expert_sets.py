"""Build the expert sets the intervention experiments act on.

For each concept this assembles the causal (gate-AtP) and correlational (SOMP / Expert Pursuit)
expert sets plus a matched random control. The interventions themselves (gate knockout /
downweighting) live in :mod:`moe_interp.circuit.intervene`; the knockout/downweight sweep that
consumes these sets is :mod:`moe_interp.circuit.downweight`.
"""

from __future__ import annotations

import random

import numpy as np

from moe_interp.analysis.common import load_somp_results
from moe_interp.capture.model_adapter import model_num_experts
from moe_interp.config import get_model_dir, get_pursuit_dir
from moe_interp.grids import top_experts


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
    """The expert sets compared by the intervention experiments: SOMP, AtP, + random.

    SOMP is the concept-restricted EVR@k pursuit (token-association). AtP is the *causal* top-``k``
    promoters from the gate-AtP grid (the experts whose ablation most lowers the concept logit) —
    the experts this experiment most wants to act on. ``random`` is matched to the AtP causal set's
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
