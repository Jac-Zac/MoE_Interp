"""Plotting utilities for Expert Pursuit results."""

from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_evr_heatmap(
    evr_matrix: np.ndarray,
    count_matrix: np.ndarray | None = None,
    output_path: Path | None = None,
) -> go.Figure:
    """EVR heatmap (final EVR per expert, all layers) with optional activation counts."""
    n_layers, n_experts = evr_matrix.shape

    if count_matrix is not None:
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Final EVR (Top-k Tokens)", "Activation Count"),
            specs=[[{"type": "heatmap"}, {"type": "heatmap"}]],
        )
        fig.add_trace(
            go.Heatmap(
                z=evr_matrix,
                x=[f"E{i}" for i in range(n_experts)],
                y=[f"L{i}" for i in range(n_layers)],
                coloraxis="coloraxis1",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Heatmap(
                z=count_matrix,
                x=[f"E{i}" for i in range(n_experts)],
                y=[f"L{i}" for i in range(n_layers)],
                coloraxis="coloraxis2",
            ),
            row=1,
            col=2,
        )
        fig.update_layout(
            title="Expert Pursuit: Final Explained Variance & Activation Counts",
            width=1600,
            height=600,
            coloraxis1=dict(colorbar=dict(title="Final EVR"), colorscale="Blues"),
            coloraxis2=dict(colorbar=dict(title="Count"), colorscale="Blues"),
        )
    else:
        fig = px.imshow(
            evr_matrix,
            x=[f"E{i}" for i in range(n_experts)],
            y=[f"L{i}" for i in range(n_layers)],
            color_continuous_scale="Blues",
            labels=dict(x="Experts", y="Layers", color="Final EVR"),
        )
        fig.update_layout(
            title="Expert Pursuit: Final Explained Variance Ratio (Top-k Tokens)",
            width=1400,
            height=600,
        )
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
    return fig
