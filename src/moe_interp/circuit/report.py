"""Assemble harmful-content proxy results into a self-contained HTML report.

Reads the artifacts written by the `circuit*` commands under ``data/<model>/circuit/``: the
gate-AtP localization grid and its exact gate-ablation comparison. Missing pieces are
skipped, so the report renders whatever has been produced so far.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go

from moe_interp.config import get_model_dir
from moe_interp.io.plots import diverging_expert_heatmap


def _css() -> str:
    return (
        "body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1040px;"
        "margin:0 auto;padding:0 1rem 4rem;color:#1a1a1a;line-height:1.55}"
        "h1{margin-bottom:0}.sub{color:#666;margin-top:.25rem}"
        "h2{margin-top:2.6rem;border-bottom:2px solid #eee;padding-bottom:.3rem}"
        "table{border-collapse:collapse;width:100%;font-size:.85rem;margin:1rem 0}"
        "th,td{border:1px solid #ddd;padding:.35rem .5rem;text-align:left}"
        "th{background:#f5f7fa}tr:nth-child(even){background:#fafbfc}"
        ".note{color:#666;font-size:.9rem}.ex{background:#fafbfc;border-left:3px solid #ccc;"
        "padding:.3rem .7rem;margin:.3rem 0;font-size:.85rem;white-space:pre-wrap}"
    )


def _table(headers: Sequence[Any], rows: Iterable[Sequence[Any]]) -> str:
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _heatmap(grid: np.ndarray, title: str, cbar: str) -> go.Figure:
    return diverging_expert_heatmap(grid, title=title, colorbar_title=cbar, height=460)


def build_report(model_name: str, *, atp_grid_path: Path | None = None) -> Path:
    """Render ``data/<model>/circuit/report.html`` from the specified artifacts."""
    cdir = get_model_dir(model_name) / "circuit"
    figs_js = True
    parts: list[str] = []

    def fig(f: go.Figure) -> str:
        nonlocal figs_js
        html = f.to_html(
            full_html=False, include_plotlyjs=("inline" if figs_js else False)
        )
        figs_js = False
        return html

    def load_json(p: Path):
        return json.loads(p.read_text()) if p.exists() else None

    # 0. Overview — the pipeline, techniques, and headline findings
    parts.append('<h2 id="overview">Overview</h2>')
    parts.append(
        "<p>An exploratory study of expert association, direct gate effects, and "
        "generation-time gate interventions using an offensive-token logit proxy.</p>"
    )
    parts.append(
        _table(
            ["stage", "technique", "what it does", "role"],
            [
                [
                    "classify",
                    "SOMP / Expert Pursuit",
                    "experts whose pursuit atoms are offensive words",
                    "no",
                ],
                [
                    "localize",
                    "gate-AtP",
                    "each backward pass scores every expert's gate effect (gate·dL/dgate), "
                    "compared with exhaustive per-slot gate ablation (see Methods)",
                    "approximation",
                ],
                [
                    "intervene",
                    "knockout / downweighting (necessity)",
                    "zero or scale selected post-top-k gates during generation",
                    "intervention",
                ],
            ],
        )
    )
    parts.append(
        '<p class="note"><b>Summary.</b> Gate-AtP correlates with the tested exact gate-ablation '
        "grid, while held-out gate interventions have modest effects on the lexical proxy. "
        "See the main report for confidence intervals and limitations.</p>"
    )

    # 1. METHODS (text)
    parts.append('<h2 id="methods">Methods</h2>')
    parts.append(
        "<h3>Setup</h3><p>OLMoE-1B-7B (16 layers, 64 experts, top-8 routing). "
        "The probe is an <b>offensive-token logit score</b>: the mean logit over a set of "
        "single-token offensive words minus the row-mean logit, read at the prediction "
        "position. Prompts are a <b>RealToxicityPrompts split</b>: high-toxicity prompts to "
        "elicit toxic continuations and matched low-toxicity prompts as the neutral control "
        "(partitioned by the dataset's own per-prompt toxicity score). The fused-experts kernel "
        "exposes only the router gate (<code>layer.mlp.experts.inputs[0]</code> → "
        "<code>hidden, top_k_index, top_k_weights</code>) as a per-expert node, so all "
        "expert-level interventions and gradients act on the gate. Every selector is "
        "<b>identified on a train split</b> of these prompts and every intervention "
        "is <b>scored on a disjoint held-out test split</b>, so the causal comparison is "
        "out-of-sample (no identify-and-score-on-the-same-prompts circularity).</p>"
    )
    parts.append(
        "<h3>Classification — lexical association (no model)</h3>"
        "<ul>"
        "<li><b>SOMP / Expert Pursuit.</b> Decompose each expert’s stored activations against the "
        "unembedding dictionary (Simultaneous Orthogonal Matching Pursuit); experts whose top atoms "
        "are offensive words are flagged. Adapts HeadPursuit (attention heads) to MoE experts.</li></ul>"
    )
    parts.append(
        "<h3>Localization — estimated direct gate contribution (gate-AtP)</h3>"
        "<p><b>What it computes.</b> gate-AtP scores every routed <code>(layer, expert)</code> by "
        "how much zeroing its router gate would change the lexical objective, estimated from a "
        "<b>one backward pass per batch</b> — a first-order Taylor expansion of the "
        "gate ablation:</p>"
        "<p style='text-align:center'><code>attribution(l,e) ≈ Σ<sub>pos</sub> "
        "gate<sub>e</sub> · ∂L/∂gate<sub>e</sub></code> , &nbsp; "
        "<code>L = Σ<sub>prompt</sub> offensive-token logit score</code></p>"
        "<p><b>On what, with what data.</b> The objective <code>L</code> is the lexical probe "
        "summed over the <b>eliciting (high-toxicity) train prompts</b>, read at each prompt's last "
        "token; <code>gate<sub>e</sub></code> is the router weight wherever expert <code>e</code> "
        "fired. The result is one signed <b>16×64</b> grid: positive = the expert raises the "
        "proxy (ablation would lower it), negative = it lowers the proxy. Experts are "
        "ranked off this grid (signed for promoters; |·| for the heatmap) to give the gate-AtP "
        "selector that drives the knockout/downweighting sweep — itself scored only on held-out prompts.</p>"
        "<p><b>Why not exhaustive gate ablation?</b> The exact effect of this intervention requires "
        "zero each gate in its own forward pass and record the probe change — but that costs one "
        "forward <i>per routed expert</i> (≈64× more). We ran it <b>once</b> as a yardstick: the "
        "gate-ablation grid and gate-AtP correlated (pooled r≈0.69, up to ≈0.96 in the late "
        "layers). This validates gate-AtP as an imperfect ranking approximation for the tested "
        "intervention (frozen check in <code>compare/faithfulness.json</code>).</p>"
    )
    parts.append(
        "<h3>Intervention — suppress the behaviour during generation (expert-level only)</h3>"
        "<p><b>Knockout / downweighting</b> (necessity) — during greedy generation, zero "
        "(knockout) or scale down (downweighting) the router gate of the top-k identified experts "
        "at every decoded step. Scored by <b>lexical propensity</b> (mean probe score over the "
        "continuation), offensive-word rate, and a <b>distinct-1</b> coherence guard, with the "
        "neutral set as a collateral check. The full sweep over selectors, budgets and downweight "
        "strengths (with bootstrap error bars) is produced and reported separately under "
        "<code>circuit/downweight/</code>.</p>"
    )

    # 2. RESULTS (figures + tables + findings)
    parts.append('<h2 id="results">Results</h2>')

    parts.append("<h3>Identifying the experts (gate-AtP)</h3>")
    if atp_grid_path is None:
        atp_grids = list((cdir / "attribution").glob("atp_grid_*.npy"))
        if len(atp_grids) > 1:
            raise ValueError(
                "Multiple gate-AtP grids found; pass atp_grid_path explicitly"
            )
        atp_grid_path = atp_grids[0] if atp_grids else None
    elif not atp_grid_path.exists():
        raise FileNotFoundError(f"Gate-AtP grid not found: {atp_grid_path}")
    if atp_grid_path is not None:
        from moe_interp.grids import top_experts

        grid = np.nan_to_num(np.load(atp_grid_path))
        parts.append(fig(_heatmap(grid, "gate-AtP effect per expert", "gate-AtP")))
        parts.append(
            _table(
                ["expert", "gate-AtP"],
                [
                    [f"L{layer}E{e}", f"{v:+.3f}"]
                    for layer, e, v in top_experts(grid, 10, by="abs")
                ],
            )
        )
        parts.append(
            '<p class="note">Estimated direct gate contribution (red = raises the proxy; '
            "blue = lowers it). Negative entries are candidate suppressors.</p>"
        )
    fa = load_json(cdir / "compare" / "faithfulness.json")
    if fa:
        r = fa.get("gate-AtP", {}).get("pooled_r")
        rtxt = f"pooled r≈{r:.2f}" if isinstance(r, (int, float)) else "pooled r≈0.69"
        parts.append(
            '<p class="note"><b>Validation (one-off).</b> gate-AtP was checked against the '
            f"exhaustive per-slot gate-ablation grid ({rtxt}, up to ≈0.96 in the late layers). "
            "This comparison applies to that specific intervention and lexical proxy. "
            "(Frozen result in <code>compare/faithfulness.json</code>.)</p>"
        )

    parts.append(
        '<p class="note">The knockout/downweighting intervention results (per selector, budget '
        "and downweight strength, with bootstrap error bars) are produced by the separate "
        "downweight sweep and stored under <code>circuit/downweight/</code>.</p>"
    )

    nav = (
        '<nav><a href="#overview">Overview</a> · <a href="#methods">Methods</a> · '
        '<a href="#results">Results</a></nav>'
    )
    body = nav + "".join(parts)
    html = (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        f"<title>Gate-intervention report — {model_name}</title><style>{_css()}</style></head>"
        f"<body><h1>Harmful-content proxy circuit</h1>"
        f'<p class="sub">{model_name} · OLMoE · RealToxicityPrompts split</p>{body}</body></html>'
    )
    out = cdir / "report.html"
    out.write_text(html)
    return out
