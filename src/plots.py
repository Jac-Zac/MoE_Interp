"""Plotting utilities for Expert Pursuit results."""

from collections import defaultdict
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


def plot_label_grid(
    results: list[dict],
    n_layers: int = 16,
    n_experts: int = 64,
    label_colors: dict[str, str] | None = None,
    output_path: Path | None = None,
) -> go.Figure:
    """Categorical grid of expert labels: Layer (x) vs Expert (y).
    Each cell is a colored square whose color encodes the expert's primary label

    Args:
        results:      List of pursuit result dicts, each with keys
                      ``layer``, ``expert``, and optionally ``labels`` (list[str]).
        n_layers:     Total number of layers in the model (default 16).
        n_experts:    Total number of experts per layer (default 64).
        label_colors: Optional mapping of label name → CSS/hex color string to
                      override auto-assigned colors for specific labels.
        output_path:  If provided, saves the figure as an HTML file.

    Returns:
        A Plotly Figure with one scatter trace per label category.
    """
    palette = px.colors.qualitative.Plotly + px.colors.qualitative.D3
    color_map: dict[str, str] = {"other": "lightgray", **(label_colors or {})}
    palette_iter = (c for c in palette if c not in color_map.values())

    # Build per-label coordinate lists with hover text
    label_coords: dict[str, tuple[list[int], list[int], list[str]]] = defaultdict(
        lambda: ([], [], [])
    )
    by_pos = {(r["layer"], r["expert"]): r for r in results}
    for lyr in range(n_layers):
        for exp in range(n_experts):
            r = by_pos.get((lyr, exp))
            label = (r.get("labels") or ["other"])[0] if r else "other"
            if label not in color_map:
                color_map[label] = next(palette_iter)
            evr_list = (r or {}).get("evr") or []
            hover = (
                (
                    f"L{lyr} E{exp}<br>"
                    f"Labels: {', '.join((r or {}).get('labels') or ['other'])}<br>"
                    f"Top tokens: {', '.join(((r or {}).get('tokens') or [])[:5])}<br>"
                    f"Final EVR: {evr_list[-1]:.4f}<br>"
                    f"n activations: {(r or {}).get('n_activations', '?')}"
                )
                if r
                else f"L{lyr} E{exp}<br>Label: other"
            )
            coords = label_coords[label]
            coords[0].append(lyr)
            coords[1].append(exp)
            coords[2].append(hover)

    # "other" always last in legend so it doesn't dominate
    ordered_labels = [l for l in label_coords if l != "other"] + (
        ["other"] if "other" in label_coords else []
    )

    fig = go.Figure()
    for label in ordered_labels:
        layers, experts, hovers = label_coords[label]
        color = color_map[label]
        fig.add_trace(
            go.Scatter(
                x=layers,
                y=experts,
                mode="markers",
                marker=dict(
                    symbol="square",
                    size=10,
                    color=color,
                    line=dict(width=0),
                ),
                name=label,
                text=hovers,
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title="Model Architecture: Layer vs Expert",
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
        legend=dict(title="category", itemsizing="constant"),
        width=max(800, n_layers * 45 + 250),
        height=max(600, n_experts * 12 + 100),
        plot_bgcolor="white",
    )

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
    return fig
