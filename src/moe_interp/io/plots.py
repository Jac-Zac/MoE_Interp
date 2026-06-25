"""Plotting utilities for Expert Pursuit results."""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go


def _save_fig(fig: go.Figure, output_path: Path | None) -> None:
    """Save a Plotly figure as an HTML file, creating parent directories as needed."""
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))


def diverging_expert_heatmap(
    grid,
    *,
    title: str,
    colorbar_title: str,
    height: int = 560,
    output_path: Path | None = None,
) -> go.Figure:
    """Layer×expert diverging heatmap: RdBu_r, centred at 0, layer 0 at the top.

    ``grid`` is an ``(n_layers, n_experts)`` array or tensor (NaNs allowed; they render as
    gaps). Shared by the causal patching and report figures.
    """
    z = grid.cpu().numpy() if hasattr(grid, "cpu") else np.asarray(grid)
    vmax = float(np.nanmax(np.abs(z))) or 1.0
    fig = go.Figure(
        go.Heatmap(
            z=z,
            zmid=0,
            zmin=-vmax,
            zmax=vmax,
            colorscale="RdBu_r",
            colorbar={"title": colorbar_title},
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="expert",
        yaxis_title="layer",
        height=height,
        yaxis={"autorange": "reversed"},
    )
    _save_fig(fig, output_path)
    return fig


def plot_scatter_grid(
    values: np.ndarray,
    title: str,
    color_label: str,
    output_path: Path | None = None,
) -> go.Figure:
    """Scatter grid: Layer (x) vs Expert (y) with colored square markers.

    Args:
        values: 2D array of shape (n_layers, n_experts) with values to visualize.
        title: Plot title.
        color_label: Label for the color scale legend.
        output_path: If provided, saves the figure as an HTML file.

    Returns:
        A Plotly Figure with one scatter trace.
    """
    n_layers, n_experts = values.shape

    layers_grid, experts_grid = np.meshgrid(np.arange(n_layers), np.arange(n_experts))
    layers = layers_grid.ravel().tolist()
    experts = experts_grid.ravel().tolist()
    flat_values = values.T.ravel().tolist()
    hovers = [
        f"L{lyr} E{exp}<br>{color_label}: {v:.4f}"
        for lyr, exp, v in zip(layers, experts, flat_values)
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=layers,
            y=experts,
            mode="markers",
            marker=dict(
                symbol="square",
                size=10,
                color=flat_values,
                colorscale="Blues",
                line=dict(width=0),
                showscale=True,
                colorbar=dict(title=color_label),
            ),
            text=hovers,
            hoverinfo="text",
        )
    )

    fig.update_layout(
        title=title,
        xaxis=dict(
            title="layer",
            tickmode="linear",
            tick0=0,
            dtick=1,
            range=[-0.5, n_layers - 0.5],
        ),
        yaxis=dict(
            title="expert",
            tickmode="linear",
            tick0=0,
            dtick=4,
            range=[-0.5, n_experts - 0.5],
        ),
        width=max(800, n_layers * 45 + 250),
        height=max(600, n_experts * 12 + 100),
        plot_bgcolor="white",
    )

    _save_fig(fig, output_path)
    return fig


def plot_evr_heatmap(
    evr_matrix: np.ndarray,
    output_path: Path | None = None,
) -> go.Figure:
    """EVR heatmap (final EVR per expert, all layers)."""
    return plot_scatter_grid(
        values=evr_matrix,
        title="Expert Pursuit: Final Explained Variance Ratio (Top-k Tokens)",
        color_label="Final EVR",
        output_path=output_path,
    )


def plot_count_heatmap(
    count_matrix: np.ndarray,
    output_path: Path | None = None,
) -> go.Figure:
    """Activation count heatmap (per expert, all layers)."""
    return plot_scatter_grid(
        values=count_matrix,
        title="Expert Pursuit: Activation Count",
        color_label="Activation Count",
        output_path=output_path,
    )
