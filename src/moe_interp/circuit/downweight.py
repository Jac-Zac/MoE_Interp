"""Knockout / downweighting sweep with per-prompt error bars (no steering).

This is the trimmed-down causal-necessity experiment: the steering arm (``esteer``/``ablate``/
dose-response) is dropped entirely. We keep only the two *gate* interventions — full **knockout**
and partial **downweighting** — applied to the SOMP (correlational), gate-AtP (causal) and
matched-random expert sets, at two budgets expressed as a fraction of the model's
``(layer, expert)`` slots (e.g. 1% and 5% of all experts; top-ranked per selector).

Downweighting and knockout are the same mechanism at different strengths: scale the selected
experts' router gate by ``s`` during multi-token greedy generation. ``s=0`` is knockout, ``s=0.9``
a 10% downweight, ``s=0.5`` a 50% downweight. We sweep ``s`` so the propensity-vs-strength curve
has resolution, and we keep the **per-prompt** propensity / concept-word-hit / distinct-1 arrays so
a bootstrap can put 95% CIs (error bars) on every point and on every paired delta-vs-baseline.

Each (budget, selector, scale, prompt-set) cell is checkpointed to the output JSON as it finishes,
so a run cut off by the 2h GPU cap resumes where it stopped.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from moe_interp.capture.model_adapter import model_num_experts
from moe_interp.circuit.expert_sets import expert_intervention_sets
from moe_interp.circuit.intervene import (
    concept_propensity,
    concept_regex,
    gate_scale_intervention,
    generate,
)
from moe_interp.pursuit.concepts import CONCEPT_WORDS, build_concept_token_ids


def _score_set(
    model,
    prompts: list[list[int]],
    concept_ids: list[int],
    pattern,
    intervention,
    max_new_tokens: int,
) -> dict:
    """Per-prompt scores for one (intervention, prompt-set) cell.

    Returns the per-prompt arrays the bootstrap consumes: ``propensity`` (mean concept-logit
    score over the continuation), ``word_hit`` (1 if any concept word appears in the decoded
    text, else 0) and ``distinct1`` (the coherence guard). No aggregation here — the means and
    CIs are derived from these in :func:`_bootstrap_cell`.
    """
    tok = model.tokenizer
    props, hits, distinct = [], [], []
    for ids in prompts:
        cont = generate(model, ids, max_new_tokens, intervention)
        props.append(concept_propensity(model, ids, cont, concept_ids, intervention))
        hits.append(1 if pattern.findall(tok.decode(cont)) else 0)
        distinct.append(len(set(cont)) / max(len(cont), 1))
    return {"propensity": props, "word_hit": hits, "distinct1": distinct}


def _bootstrap_cell(cell: dict, base: dict | None, n_boot: int, rng) -> dict:
    """Attach mean + 95% CI to a cell, plus the paired delta-vs-baseline CI when ``base`` given.

    Error bars come from resampling the per-prompt arrays with replacement. For the delta we
    resample the *paired* (same-prompt) baseline and method arrays so the CI reflects the
    within-prompt change (positive delta = concept removed relative to baseline).
    """
    out = {**cell}
    for field in ("propensity", "word_hit", "distinct1"):
        x = np.asarray(cell[field], dtype=float)
        bs = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(1)
        out[f"{field}_mean"] = float(x.mean())
        out[f"{field}_ci"] = [
            float(np.percentile(bs, 2.5)),
            float(np.percentile(bs, 97.5)),
        ]
        if base is not None:
            b = np.asarray(base[field], dtype=float)
            n = min(len(b), len(x))
            idx = rng.integers(0, n, size=(n_boot, n))
            d = b[:n][idx].mean(1) - x[:n][idx].mean(1)
            out[f"{field}_delta"] = float(b[:n].mean() - x[:n].mean())
            out[f"{field}_delta_ci"] = [
                float(np.percentile(d, 2.5)),
                float(np.percentile(d, 97.5)),
            ]
    return out


def run_downweight_sweep(
    model,
    model_name: str,
    *,
    concept: str,
    dataset: str,
    train: tuple[list[list[int]], list[list[int]]],
    test: tuple[list[list[int]], list[list[int]]],
    out_path: Path,
    budgets_frac: tuple[float, ...] = (0.01, 0.05),
    scales: tuple[float, ...] = (0.9, 0.5, 0.25, 0.0),
    max_new_tokens: int = 24,
    atp_grid_path=None,
    n_boot: int = 10000,
) -> dict:
    """Knockout/downweight sweep over SOMP / AtP / random at each budget, with per-prompt CIs.

    ``budgets_frac`` are fractions of the model's ``n_layers * n_experts`` slots; each becomes a
    top-``k`` budget per selector. ``scales`` are the gate multipliers swept per selector (``0.0``
    = knockout). The baseline (no intervention) is scored once and reused for every delta. Results
    are checkpointed to ``out_path`` cell-by-cell and completed cells are skipped on resume.
    """
    elic_eval, neut_eval = test
    concept_words = CONCEPT_WORDS[concept]
    concept_ids = build_concept_token_ids(model.tokenizer, concept_words)
    pattern = concept_regex(concept_words)
    rng = np.random.default_rng(0)

    n_total = model.config.num_hidden_layers * model_num_experts(model)
    budgets = {f"{f:g}": max(1, round(f * n_total)) for f in budgets_frac}
    prompt_sets = (("eliciting", elic_eval), ("neutral", neut_eval))

    # Resume from any prior partial run.
    if out_path.exists():
        res = json.loads(out_path.read_text())
    else:
        res = {
            "meta": {
                "concept": concept,
                "dataset": dataset,
                "n_total_experts": n_total,
                "budgets": budgets,
                "scales": list(scales),
                "max_new_tokens": max_new_tokens,
                "n_test": len(elic_eval),
                "n_boot": n_boot,
            },
            "baseline": {},
            "budgets": {},
        }

    def save():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(res, indent=2))

    # Baseline (no intervention), scored once on each prompt set.
    for setname, prompts in prompt_sets:
        if setname not in res["baseline"]:
            print(f"[baseline / {setname}] generating {len(prompts)} ...", flush=True)
            res["baseline"][setname] = _score_set(
                model, prompts, concept_ids, pattern, None, max_new_tokens
            )
            save()

    for frac_key, k in budgets.items():
        bnode = res["budgets"].setdefault(frac_key, {"k": k, "sets": {}, "methods": {}})
        if not bnode["sets"]:
            sets = expert_intervention_sets(
                model,
                model_name,
                elic_eval,
                concept=concept,
                dataset=dataset,
                k=k,
                atp_grid_path=atp_grid_path,
            )
            bnode["sets"] = {name: [list(le) for le in s] for name, s in sets.items()}
            save()
        sets = {name: [tuple(le) for le in s] for name, s in bnode["sets"].items()}

        for name, experts in sets.items():
            mnode = bnode["methods"].setdefault(name, {})
            for s in scales:
                skey = f"{s:g}"
                if skey in mnode and "eliciting" in mnode[skey]:
                    continue
                print(
                    f"[budget={frac_key} (k={k}) / {name} / scale={skey}] ...",
                    flush=True,
                )
                intervention = gate_scale_intervention(experts, s)
                cell = {}
                for setname, prompts in prompt_sets:
                    raw = _score_set(
                        model,
                        prompts,
                        concept_ids,
                        pattern,
                        intervention,
                        max_new_tokens,
                    )
                    cell[setname] = _bootstrap_cell(
                        raw, res["baseline"][setname], n_boot, rng
                    )
                mnode[skey] = cell
                save()

    print(f"Downweight sweep complete -> {out_path}", flush=True)
    return res
