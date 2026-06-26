"""Faithfulness of cheap attributors against the causal patching grid.

The patching grid (single-expert gate ablation, one forward per expert) is the causal
ground truth for "which experts drive the toxic continuation". This scores how well a
cheap method predicts it, by Pearson correlation over the experts the grid actually scored.
Empirically gate-AtP (one backward pass) tracks it closely (pooled r≈0.80, per-layer up to
0.98) — causal attribution, not token association, is what predicts causal effect.
"""

from __future__ import annotations

import torch


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-12))


def faithfulness(
    grids: dict[str, torch.Tensor], patching_grid: torch.Tensor
) -> dict[str, dict]:
    """Pooled + per-layer Pearson r of each method grid vs the causal patching grid."""
    mask = patching_grid != 0  # experts the patching sweep actually scored
    gt = patching_grid[mask]
    out: dict[str, dict] = {}
    for name, g in grids.items():
        per_layer = {}
        for layer in range(patching_grid.shape[0]):
            fm = patching_grid[layer] != 0
            if fm.sum() >= 3:
                per_layer[layer] = _pearson(g[layer][fm], patching_grid[layer][fm])
        out[name] = {"pooled_r": _pearson(g[mask], gt), "per_layer_r": per_layer}
    return out


def faithfulness_bar(scores: dict[str, dict], *, title: str, height: int = 400):
    """Bar chart figure of pooled faithfulness r per method (shared with the report)."""
    import plotly.graph_objects as go

    names = list(scores)
    fig = go.Figure(go.Bar(x=names, y=[scores[n]["pooled_r"] for n in names]))
    fig.update_layout(
        title=title, yaxis_title="Pearson r vs causal patching", height=height
    )
    return fig


def plot_faithfulness(scores: dict[str, dict], output_path, *, title: str) -> None:
    """Write the pooled-faithfulness bar chart to ``output_path`` as standalone HTML."""
    faithfulness_bar(scores, title=title).write_html(str(output_path))


def intervention_bar(
    methods: dict[str, dict], *, title: str, height: int = 420
):
    """Grouped bar chart: concept propensity + Δ vs baseline per intervention method."""
    import plotly.graph_objects as go

    names = list(methods)
    base = methods.get("baseline", {}).get("eliciting_propensity", 0.0)
    elic = [methods[m].get("eliciting_propensity", 0.0) for m in names]
    delta = [base - v for v in elic]
    neutral = [methods[m].get("neutral_propensity", 0.0) for m in names]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="eliciting propensity", x=names, y=elic, marker_color="crimson"))
    fig.add_trace(go.Bar(name="Δ vs baseline", x=names, y=delta, marker_color="steelblue"))
    fig.add_trace(go.Bar(name="neutral propensity", x=names, y=neutral, marker_color="gray"))
    fig.update_layout(
        title=title,
        barmode="group",
        yaxis_title="toxic-logit propensity",
        height=height,
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def plot_intervention(methods: dict[str, dict], output_path, *, title: str) -> None:
    """Write the intervention propensity bar chart to ``output_path`` as standalone HTML."""
    intervention_bar(methods, title=title).write_html(str(output_path))
