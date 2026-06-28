"""Plotting helper for the intervention experiment: the concept-propensity bar chart."""

from __future__ import annotations


def intervention_bar(methods: dict[str, dict], *, title: str, height: int = 420):
    """Grouped bar chart: concept propensity + Δ vs baseline per intervention method."""
    import plotly.graph_objects as go

    names = list(methods)
    base = methods.get("baseline", {}).get("eliciting_propensity", 0.0)
    elic = [methods[m].get("eliciting_propensity", 0.0) for m in names]
    delta = [base - v for v in elic]
    neutral = [methods[m].get("neutral_propensity", 0.0) for m in names]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(name="eliciting propensity", x=names, y=elic, marker_color="crimson")
    )
    fig.add_trace(
        go.Bar(name="Δ vs baseline", x=names, y=delta, marker_color="steelblue")
    )
    fig.add_trace(
        go.Bar(name="neutral propensity", x=names, y=neutral, marker_color="gray")
    )
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
