#!/usr/bin/env python

# %% Imports
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from src.cache import iter_layers, load_metadata
from src.environment import get_data_dir, load_env, set_seed

# %% Configuration
seed = 1337
load_env()
set_seed(seed)

# %% Setup
data_dir = get_data_dir()
encodings_dir = data_dir / "encodings"
clustering_dir = data_dir / "clustering"
clustering_dir.mkdir(parents=True, exist_ok=True)

metadata_path = encodings_dir / "metadata.json"
if not metadata_path.exists():
    raise FileNotFoundError(f"Run encode first — metadata not found at {metadata_path}")

metadata = load_metadata(encodings_dir)
n_layers = metadata["n_layers"]
n_experts = metadata["n_experts"]

# %% Stream activations layer-by-layer and build per-layer arrays
# Peak memory = one layer's activations, not all layers at once.
counts_matrix = np.zeros((n_layers, n_experts), dtype=np.int64)

# layer_idx -> {"X": ndarray, "X_norm": ndarray, "labels": ndarray}
layer_data: dict[int, dict] = {}


def _l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return X / norms


print("Loading activations layer by layer...")
for layer_idx, expert_acts in iter_layers(encodings_dir, n_layers, n_experts):
    Xs, labels = [], []
    for expert_id, acts in expert_acts.items():
        if acts.numel() == 0:
            continue
        if acts.ndim == 1:
            acts = acts.reshape(1, -1)
        n = acts.shape[0]
        counts_matrix[layer_idx, expert_id] = n
        Xs.append(acts.float().numpy())
        labels.extend([expert_id] * n)
    if Xs:
        X = np.concatenate(Xs, axis=0)  # (N_layer, d_model)
        layer_data[layer_idx] = {
            "X": X,
            "X_norm": _l2_normalize_rows(X),
            "labels": np.array(labels),  # (N_layer,) — true expert id
        }

print(
    f"Loaded {len(layer_data)} layers, "
    f"{sum(d['X'].shape[0] for d in layer_data.values())} total activations"
)

# %% Plot 1 — Activation frequency heatmaps (routing uniformity check)
# Shows how many last-token activations each expert collected per layer.
# A uniform distribution across experts is the ideal case.
fig = px.imshow(
    counts_matrix,
    x=[f"E{i}" for i in range(n_experts)],
    y=[f"L{i}" for i in range(n_layers)],
    color_continuous_scale="Viridis",
    labels=dict(x="Expert", y="Layer", color="Activation count"),
    title="Expert activation frequency (counts)",
)
fig.update_layout(width=1600, height=600)
fig.write_html(clustering_dir / "activation_frequency_counts.html")
fig.show()

row_sums = counts_matrix.sum(axis=1, keepdims=True)
row_sums = np.where(row_sums == 0, 1, row_sums)
counts_fraction = counts_matrix / row_sums
fig = px.imshow(
    counts_fraction,
    x=[f"E{i}" for i in range(n_experts)],
    y=[f"L{i}" for i in range(n_layers)],
    color_continuous_scale="Cividis",
    labels=dict(x="Expert", y="Layer", color="Fraction of layer"),
    title="Expert activation frequency (row-normalized)",
)
fig.update_layout(width=1600, height=600)
fig.write_html(clustering_dir / "activation_frequency_fraction.html")
fig.show()

# %% K-means clustering per layer + Hungarian matching accuracy
# For each layer:
#   1. L2-normalise activations so geometry matches cosine distance
#   2. Run KMeans(k=n_experts)
#   3. Build contingency matrix C[cluster, true_expert]
#   4. Optimal assignment via Hungarian algorithm (maximize overlap)
#   5. Accuracy = matched activations / total activations

print("\nRunning K-means per layer...")
accuracies: dict[int, float] = {}
layer_kmeans_labels: dict[int, np.ndarray] = {}  # predicted cluster labels (remapped)

for layer_idx, data in layer_data.items():
    X = data["X_norm"]
    true_labels = data["labels"]
    n_samples = X.shape[0]

    # Only run KMeans if we have enough samples
    k = min(n_experts, n_samples)
    km = KMeans(n_clusters=k, n_init="auto", random_state=seed)
    pred = km.fit_predict(X)

    # Contingency matrix: C[i, j] = # activations predicted cluster i and true expert j
    active_experts = sorted(set(true_labels.tolist()))
    expert_to_col = {e: c for c, e in enumerate(active_experts)}
    n_active = len(active_experts)
    C = np.zeros((k, n_active), dtype=np.int64)
    for p, t in zip(pred, true_labels):
        C[p, expert_to_col[t]] += 1

    # Hungarian: find optimal cluster → expert assignment
    row_ind, col_ind = linear_sum_assignment(-C)
    matched = C[row_ind, col_ind].sum()
    accuracy = matched / n_samples
    accuracies[layer_idx] = accuracy

    # Remap predicted cluster ids to matched expert ids for plotting
    cluster_to_expert = np.full(k, -1, dtype=np.int64)
    for r, c in zip(row_ind, col_ind):
        cluster_to_expert[r] = active_experts[c]
    remapped = cluster_to_expert[pred]
    layer_kmeans_labels[layer_idx] = remapped

    print(
        f"  Layer {layer_idx:2d}: accuracy = {accuracy:.3f}  "
        f"(n={n_samples}, k={k}, active_experts={n_active})"
    )

# %% Plot 2 — Clustering accuracy over layers
random_baseline = 1.0 / n_experts
layers_sorted = sorted(accuracies)
acc_values = [accuracies[li] for li in layers_sorted]

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=layers_sorted,
        y=acc_values,
        mode="lines+markers",
        name="K-means accuracy",
        line=dict(color="royalblue", width=2),
        marker=dict(size=8),
    )
)
fig.add_hline(
    y=random_baseline,
    line_dash="dash",
    line_color="red",
    annotation_text=f"Random baseline (1/{n_experts} = {random_baseline:.3f})",
    annotation_position="bottom right",
)
fig.update_layout(
    title="K-means clustering accuracy per layer (Hungarian matching, k=n_experts)",
    xaxis_title="Layer",
    yaxis_title="Accuracy",
    yaxis=dict(range=[0, 1]),
    width=900,
    height=500,
)
fig.write_html(clustering_dir / "clustering_accuracy.html")
fig.show()

# %% Plot 3 — Normalized activation magnitude per layer
# Compare how active each expert is, normalized by sqrt(d_model).
mag_matrix = np.zeros((n_layers, n_experts), dtype=np.float32)
d_model = None
for layer_idx, expert_acts in iter_layers(encodings_dir, n_layers, n_experts):
    for expert_id, acts in expert_acts.items():
        if acts.numel() == 0:
            continue
        acts_np = acts.float().numpy()
        if d_model is None:
            d_model = acts_np.shape[1]
        mag = np.linalg.norm(acts_np, axis=1)
        mag_matrix[layer_idx, expert_id] = float(np.mean(mag))

if d_model is not None:
    mag_matrix = mag_matrix / np.sqrt(d_model)
fig = px.imshow(
    mag_matrix,
    x=[f"E{i}" for i in range(n_experts)],
    y=[f"L{i}" for i in range(n_layers)],
    color_continuous_scale="Magma",
    labels=dict(x="Expert", y="Layer", color="Mean L2 / sqrt(d_model)"),
    title="Mean activation magnitude (normalized)",
)
fig.update_layout(width=1600, height=600)
fig.write_html(clustering_dir / "activation_magnitude_normalized.html")
fig.show()

# %% PCA visualisation — 4×4 subplot grid, one panel per layer
# We render a single grid for readability.


def _pca_scatter(
    layer_idx: int, color_labels: np.ndarray, title_suffix: str
) -> go.Figure:
    """Return a single-layer 2D PCA scatter figure."""
    X = layer_data[layer_idx]["X_norm"]
    coords = PCA(n_components=2, random_state=seed).fit_transform(X)

    fig = px.scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        color=color_labels.astype(str),
        opacity=0.5,
        title=f"Layer {layer_idx} — {title_suffix}",
        labels={"x": "PC1", "y": "PC2", "color": "Expert"},
        color_discrete_sequence=px.colors.qualitative.Alphabet
        + px.colors.qualitative.Dark24,
    )
    fig.update_traces(marker_size=4)
    fig.update_layout(showlegend=False)
    return fig


def _make_pca_grid(color_key: str, title: str) -> go.Figure:
    """Build a 4×4 PCA subplot grid for all layers using the given color source."""
    rows, cols = 4, 4
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[f"L{li}" for li in range(n_layers)],
        horizontal_spacing=0.04,
        vertical_spacing=0.07,
    )
    # Precompute a discrete palette cycling through 64 colours
    palette = px.colors.qualitative.Alphabet + px.colors.qualitative.Dark24

    for panel_idx in range(n_layers):
        if panel_idx not in layer_data:
            continue
        row = panel_idx // cols + 1
        col = panel_idx % cols + 1

        X = layer_data[panel_idx]["X_norm"]
        coords = PCA(n_components=2, random_state=seed).fit_transform(X)

        if color_key == "true":
            color_labels = layer_data[panel_idx]["labels"]
        else:
            color_labels = layer_kmeans_labels[panel_idx]

        # Map expert ids → colour strings
        unique_experts = sorted(set(color_labels.tolist()))
        expert_color = {
            e: palette[i % len(palette)] for i, e in enumerate(unique_experts)
        }
        colors = [expert_color[e] for e in color_labels]

        fig.add_trace(
            go.Scatter(
                x=coords[:, 0],
                y=coords[:, 1],
                mode="markers",
                marker=dict(size=3, color=colors, opacity=0.5),
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    fig.update_layout(title=title, width=1600, height=1400)
    # Hide axis ticks on all subplots to keep panels clean
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
    return fig


print("\nGenerating PCA plot...")
fig = _make_pca_grid(
    "pred", "PCA of expert activations — coloured by K-means prediction"
)
fig.write_html(clustering_dir / "pca_kmeans_labels.html")
fig.show()

fig = _make_pca_grid("true", "PCA of expert activations — coloured by true expert")
fig.write_html(clustering_dir / "pca_true_labels.html")
fig.show()

print(f"\nAll plots saved to {clustering_dir}")
print("\nAccuracy summary:")
for li in sorted(accuracies):
    print(f"  Layer {li:2d}: {accuracies[li]:.3f}")
print(f"  Mean:    {np.mean(list(accuracies.values())):.3f}")
