"""Assemble the toxic-circuit results into one self-contained HTML report.

Reads the artifacts written by the `circuit*` commands under ``data/<model>/circuit/``:
patching grid, DLA grid, the faithfulness comparison, and the intervention experiment.
Missing pieces are skipped, so the report renders whatever has been produced so far.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go

from moe_interp.config import get_model_dir


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
    vmax = float(np.nanmax(np.abs(grid))) or 1.0
    fig = go.Figure(go.Heatmap(z=grid, zmid=0, zmin=-vmax, zmax=vmax,
                               colorscale="RdBu_r", colorbar={"title": cbar}))
    fig.update_layout(title=title, xaxis_title="expert", yaxis_title="layer",
                      height=460, yaxis={"autorange": "reversed"})
    return fig


def build_report(model_name: str) -> Path:
    """Render ``data/<model>/circuit/report.html`` from whatever artifacts exist."""
    cdir = get_model_dir(model_name) / "circuit"
    figs_js = True
    parts: list[str] = []

    def fig(f: go.Figure) -> str:
        nonlocal figs_js
        html = f.to_html(full_html=False, include_plotlyjs=("inline" if figs_js else False))
        figs_js = False
        return html

    def load_json(p: Path):
        return json.loads(p.read_text()) if p.exists() else None

    # 1. Expert identification (causal patching grid + DLA grid + top tables)
    parts.append('<h2 id="id">Identifying toxic experts</h2>')
    pg = cdir / "patching" / "patching_grid.npy"
    if pg.exists():
        parts.append(
            '<p class="note">Causal ground truth: Δ toxic-logit when each expert\'s gate is '
            "ablated (red = the expert promotes toxicity; blue = suppresses it).</p>"
        )
        parts.append(fig(_heatmap(np.load(pg), "Causal patching effect per expert", "Δ toxic")))
        top = load_json(cdir / "patching" / "top_experts.json") or []
        parts.append(_table(["expert", "effect"],
                            [[f"L{r['layer']}E{r['expert']}", f"{r['effect']:+.3f}"] for r in top[:10]]))
    dg = cdir / "dla" / "pile10k" / "dla_grid.npy"
    if dg.exists():
        parts.append(
            '<p class="note">Gradient-free DLA: how much each expert writes toward toxic '
            "vocabulary (from stored activations only, no model forward).</p>"
        )
        parts.append(fig(_heatmap(np.load(dg), "DLA toxic-write score per expert", "DLA")))

    # 2. Faithfulness of cheap attributors vs the causal grid
    fa = load_json(cdir / "compare" / "faithfulness.json")
    if fa:
        parts.append('<h2 id="faith">Which cheap method predicts the causal grid?</h2>')
        names = list(fa)
        bar = go.Figure(go.Bar(x=names, y=[fa[n]["pooled_r"] for n in names]))
        bar.update_layout(title="Attributor faithfulness vs causal patching",
                          yaxis_title="Pearson r", height=380)
        parts.append(fig(bar))
        parts.append('<p class="note">gate-AtP (one backward pass) predicts the expensive '
                     "patching grid best; direction-based methods only track it at the final layer.</p>")

    # 3. Causal intervention: suppress the concept during generation
    iv = load_json(cdir / "steer" / "intervention.json")
    if iv:
        concept = iv.get("_meta", {}).get("concept", "toxic")
        parts.append(f'<h2 id="steer">Suppressing "{concept}" generation</h2>')
        rows, methods = [], [m for m in iv if m != "_meta"]
        base = iv.get("baseline", {}).get("eliciting_propensity", 0.0)
        for m in methods:
            b = iv[m]
            drop = base - b.get("eliciting_propensity", 0.0)
            rows.append([m, f"{b.get('eliciting_propensity', 0):+.3f}",
                         f"{drop:+.3f}", f"{b.get('neutral_propensity', 0):+.3f}",
                         f"{b.get('eliciting_word_frac', 0):.2f}"])
        parts.append(_table(
            ["method", "concept propensity", "Δ vs baseline", "neutral propensity", "word frac"], rows))
        parts.append('<p class="note">Lower propensity = less of the concept; the neutral column '
                     "is the collateral check (should stay near baseline). Δ &gt; 0 means the "
                     "intervention suppressed the concept.</p>")
        # example generations: baseline vs the best intervention
        others = [m for m in methods if m != "baseline"]
        best = min(others, key=lambda m: iv[m].get("eliciting_propensity", 0.0), default=None)
        for label in [m for m in ("baseline", best) if m]:
            ex = iv[label].get("examples", [])
            if ex:
                parts.append(f"<h3>{label} — example continuations</h3>")
                parts.extend(f'<div class="ex">{e}</div>' for e in ex[:4])

    nav = ('<nav><a href="#id">Identify</a> · <a href="#faith">Faithfulness</a> · '
           '<a href="#steer">Intervene</a></nav>')
    body = nav + "".join(parts)
    html = (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        f"<title>Toxic-circuit report — {model_name}</title><style>{_css()}</style></head>"
        f"<body><h1>Causal toxic-expert circuit</h1>"
        f'<p class="sub">{model_name} · OLMoE · pile10k/RTP toxic prompts</p>{body}</body></html>'
    )
    out = cdir / "report.html"
    out.write_text(html)
    return out
