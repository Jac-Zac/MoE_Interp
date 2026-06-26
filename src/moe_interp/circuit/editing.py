"""Counterfactual expert editing — interchange interventions on factual/counterfactual pairs.

The causal test the knockout / steer arms cannot give. Take a minimal pair that differs only in
its answer ("The sum of 2 and 3 is" -> 5 vs "... 2 and 4 is" -> 6), capture the residual the
*counterfactual* run writes, and splice it into the *factual* run only at the positions where
the target experts fire. If the factual prediction moves toward the counterfactual answer more
for the identified experts than for a random group, the group causally carries the concept.
Scored by single-token log-prob (the proof-of-concept granularity).

The fused MoE kernel exposes no per-expert *output* node, so the interchange is at the residual
stream **restricted to the target experts' routed positions** — a localized interchange, not a
per-expert-output swap (the same fused-kernel limit that blocks neuron-level AtP). A
random-expert group at the same layers is the specificity control.
"""

from __future__ import annotations

import torch


def _group_by_layer(experts: list[tuple[int, int]]) -> dict[int, list[int]]:
    by_layer: dict[int, list[int]] = {}
    for layer, e in experts:
        by_layer.setdefault(layer, []).append(e)
    return by_layer


def interchange_edit(
    model, pair: dict, experts: list[tuple[int, int]]
) -> tuple[dict, dict]:
    """Answer log-probs for one pair, with no edit and under the cf->fact interchange.

    ``pair`` is a dict from :func:`~moe_interp.circuit.prompts.numbers_counterfactual_pairs`.
    The counterfactual residual is captured at every layer in ``experts`` and spliced into the
    factual run at the positions where those experts fired. Returns ``(base, edited)`` where
    each maps the factual and counterfactual answer ids to their last-position log-prob.
    """
    by_layer = _group_by_layer(experts)
    layers = sorted(by_layer)
    ans = (pair["fact_ans"], pair["cf_ans"])

    # Donor: the residuals the counterfactual run produces at each target layer.
    with torch.no_grad(), model.trace(pair["cf"]):
        donors = {layer: model.model.layers[layer].output.save() for layer in layers}

    def run(edit: bool) -> dict:
        with torch.no_grad(), model.trace(pair["fact"]):
            if edit:
                for layer in layers:
                    L = model.model.layers[layer]
                    _, idx, _ = L.mlp.experts.inputs[0]
                    h = L.output
                    hcf = donors[layer].to(h.device, h.dtype)
                    fired = torch.zeros(
                        idx.shape[0], dtype=torch.bool, device=idx.device
                    )
                    for e in by_layer[layer]:
                        fired |= (idx == e).any(dim=-1)
                    mask = fired.reshape(h.shape[:-1]).unsqueeze(-1).bool()
                    h[:] = torch.where(mask, hcf, h)
            logits = model.output.logits.save()
        lp = torch.log_softmax(logits[0, -1].float().cpu(), dim=-1)
        return {a: float(lp[a]) for a in ans}

    return run(False), run(True)


def run_counterfactual_edit(model, pairs: list[dict], groups: dict[str, list]) -> dict:
    """Mean interchange effect per expert group over all minimal pairs.

    ``groups`` maps a name (e.g. ``"numbers-SOMP"``, ``"random"``) to a ``[(layer, expert), ...]``
    list. For each group we report, averaged over pairs:

    - ``toward_cf_swing`` — change in the (cf − factual) answer log-prob *gap* caused by the
      edit; positive means the splice pushed the prediction toward the counterfactual answer.
    - ``flip_rate`` — fraction of pairs where the model preferred the factual answer at baseline
      but the counterfactual answer after the edit (a hard causal flip).

    A real causal group beats the random control on both. ``groups`` with an empty expert list
    score ~0 by construction (no positions edited).
    """
    results: dict[str, dict] = {}
    for name, experts in groups.items():
        swings, flips = [], []
        for p in pairs:
            base, edit = interchange_edit(model, p, experts)
            f, c = p["fact_ans"], p["cf_ans"]
            swings.append((edit[c] - edit[f]) - (base[c] - base[f]))
            flips.append(base[f] >= base[c] and edit[c] > edit[f])
        n = max(len(pairs), 1)
        results[name] = {
            "toward_cf_swing": sum(swings) / n,
            "flip_rate": sum(flips) / n,
            "n_pairs": len(pairs),
            "n_experts": len(experts),
        }
    return results
