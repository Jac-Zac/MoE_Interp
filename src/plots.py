"""Plotting utilities for Expert Pursuit results."""

from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def plot_evr_heatmap(
    evr_matrix: np.ndarray,
    output_path: Path | None = None,
) -> go.Figure:
    """EVR heatmap (top EVR per expert, all layers)."""
    n_layers, n_experts = evr_matrix.shape
    fig = px.imshow(
        evr_matrix,
        x=[f"E{i}" for i in range(n_experts)],
        y=[f"L{i}" for i in range(n_layers)],
        color_continuous_scale="Blues",
        labels=dict(x="Experts", y="Layers", color="EVR"),
    )
    fig.update_layout(
        title="Expert Pursuit: Top Explained Variance Ratio",
        width=1400,
        height=600,
    )
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
    return fig


def plot_concept_frequency(
    concepts: list[tuple[str, int]],
    output_path: Path | None = None,
) -> go.Figure:
    """Bar chart of most frequent concepts."""
    if not concepts:
        fig = go.Figure()
        fig.add_annotation(text="No concepts found", x=0.5, y=0.5, showarrow=False)
        return fig

    words = [w for w, _ in concepts]
    counts = [c for _, c in concepts]

    fig = go.Figure(
        go.Bar(
            x=counts,
            y=words,
            orientation="h",
            marker_color="#22577A",
        )
    )
    fig.update_layout(
        title=f"Top {len(concepts)} concepts",
        xaxis_title="Expert count",
        yaxis=dict(autorange="reversed"),
        height=max(400, len(words) * 25),
        width=800,
    )

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
    return fig
