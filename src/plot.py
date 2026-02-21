"""Plotting utilities for Expert Pursuit results.

Generates EVR heatmaps, z-score heatmaps, and concept frequency charts
from saved PursuitResult data. Uses Plotly for all figures.
"""

from __future__ import annotations

from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go

from src.pursuit import PursuitResult


def plot_evr_heatmap(
    result: PursuitResult,
    output_path: Path | None = None,
) -> go.Figure:
    """EVR heatmap (final EVR per expert, all layers).

    Args:
        result: PursuitResult with evr_matrix
        output_path: If set, save figure as HTML

    Returns:
        Plotly Figure
    """
    data = result.evr_matrix[:, :, -1].cpu().numpy()
    n_layers, n_experts = data.shape

    fig = px.imshow(
        data,
        x=[f"E{i}" for i in range(n_experts)],
        y=[f"L{i}" for i in range(n_layers)],
        color_continuous_scale="Blues",
        labels=dict(x="Experts", y="Layers", color="EVR"),
    )
    fig.update_layout(
        title=f"Expert Pursuit EVR ({result.property_name})",
        width=1400,
        height=600,
    )

    if output_path:
        _save_figure(fig, output_path)
    return fig


def plot_zscore_heatmap(
    result: PursuitResult,
    output_path: Path | None = None,
) -> go.Figure:
    """Z-score heatmap (concept coherence per expert).

    Args:
        result: PursuitResult with zscore_matrix
        output_path: If set, save figure

    Returns:
        Plotly Figure
    """
    data = result.zscore_matrix.cpu().numpy()
    n_layers, n_experts = data.shape

    fig = px.imshow(
        data,
        x=[f"E{i}" for i in range(n_experts)],
        y=[f"L{i}" for i in range(n_layers)],
        color_continuous_scale=px.colors.diverging.RdYlBu_r,
        color_continuous_midpoint=0.0,
        labels=dict(x="Experts", y="Layers", color="Z-Score"),
    )
    fig.update_layout(
        title=f"Expert Concept Coherence ({result.property_name})",
        width=1400,
        height=600,
    )

    if output_path:
        _save_figure(fig, output_path)
    return fig


def plot_concept_frequency(
    result: PursuitResult,
    top_n: int = 20,
    output_path: Path | None = None,
) -> go.Figure:
    """Bar chart of most frequent concepts across all experts.

    Args:
        result: PursuitResult with expert decompositions
        top_n: Number of top concepts to show
        output_path: If set, save figure

    Returns:
        Plotly Figure
    """
    counter = result.concept_frequency(top_n=5)
    if not counter:
        fig = go.Figure()
        fig.add_annotation(text="No concepts found", x=0.5, y=0.5, showarrow=False)
        return fig

    most_common = counter.most_common(top_n)
    words = [w for w, _ in most_common]
    counts = [c for _, c in most_common]

    fig = go.Figure(
        go.Bar(
            x=counts,
            y=words,
            orientation="h",
            marker_color="#22577A",
        )
    )
    fig.update_layout(
        title=f"Top {top_n} concepts ({result.property_name})",
        xaxis_title="Expert count",
        yaxis=dict(autorange="reversed"),
        height=max(400, len(words) * 25),
        width=800,
    )

    if output_path:
        _save_figure(fig, output_path)
    return fig


def _save_figure(fig: go.Figure, path: Path) -> None:
    """Save a Plotly figure. Uses .html for interactive, .png/.pdf for static."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".html":
        fig.write_html(str(path))
    else:
        # Static export (.png, .pdf, .svg) requires kaleido
        fig.write_image(str(path))


def generate_all_plots(
    result: PursuitResult,
    output_dir: Path,
) -> list[Path]:
    """Generate and save all standard plots.

    Args:
        result: PursuitResult to visualize
        output_dir: Directory to save figures

    Returns:
        List of saved file paths
    """
    output_dir = Path(output_dir)
    paths = []

    evr_path = output_dir / "evr_heatmap.html"
    plot_evr_heatmap(result, evr_path)
    paths.append(evr_path)

    zscore_path = output_dir / "zscore_heatmap.html"
    plot_zscore_heatmap(result, zscore_path)
    paths.append(zscore_path)

    freq_path = output_dir / "concept_frequency.html"
    plot_concept_frequency(result, output_path=freq_path)
    paths.append(freq_path)

    return paths
