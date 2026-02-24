"""Projection pursuit for Expert Pursuit."""

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.cache import load_expert, load_unembedding


def projection_pursuit(
    X: torch.Tensor,
    dictionary: torch.Tensor,
    tokenizer,
    k: int = 50,
) -> tuple[list[str], list[float]]:
    """Project expert activations onto dictionary, return top-k tokens by EVR.

    EVR per token = var(projection) / total_var, clamped to [0,1].
    Note: Dictionary vectors are non-orthogonal, so sum(EVR) may exceed 1.
    """
    if X.shape[0] <= 1:
        return [], []

    X_centered = X - X.mean(dim=0, keepdim=True)
    projections = X_centered @ dictionary.T

    total_var = X_centered.var(dim=0).sum()
    if total_var < 1e-10:
        return [], []

    var_per_token = projections.var(dim=0)
    evr = (var_per_token / total_var).clamp(0, 1)

    valid_mask = evr > 1e-6
    if not valid_mask.any():
        return [], []

    top_k = evr.topk(min(k, int(valid_mask.sum().item())))

    tokens = [tokenizer.decode([i.item()]).strip() for i in top_k.indices]
    return tokens, top_k.values.tolist()


def run_pursuit(
    encodings_dir: Path,
    tokenizer,
    min_activations: int = 5,
    k: int = 50,
    output_dir: Path | None = None,
    data_dir: Path | None = None,
) -> tuple[list[dict], np.ndarray]:
    """Run projection pursuit on all experts.

    Args:
        encodings_dir: Directory containing expert encodings
        tokenizer: Model tokenizer
        min_activations: Minimum activations required to analyze an expert
        k: Number of top tokens to return per expert
        output_dir: Optional output directory for results and plots
        data_dir: Data directory containing unembedding. If None, derived from encodings_dir.

    Returns:
        Tuple of (results list, evr_matrix)
    """
    encodings_dir = Path(encodings_dir)
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    if data_dir is None:
        data_dir = encodings_dir.parent
    dictionary = load_unembedding(data_dir / "unembedding" / "dictionary.safetensors")

    metadata_path = encodings_dir / "metadata.json"
    if metadata_path.exists():
        import json

        metadata = json.loads(metadata_path.read_text())
        n_layers = metadata["n_layers"]
        n_experts = metadata["n_experts"]
    else:
        raise ValueError(f"No metadata found in {encodings_dir}")

    results = []
    for li in tqdm(range(n_layers), desc="Projection pursuit"):
        layer_dir = encodings_dir / f"layer_{li:02d}"
        for ei in range(n_experts):
            expert_path = layer_dir / f"expert_{ei:03d}.safetensors"
            if not expert_path.exists():
                continue
            data = load_expert(expert_path)
            X = data["activations"].float()
            if X.shape[0] < min_activations:
                continue
            tokens, evr = projection_pursuit(X, dictionary, tokenizer, k=k)
            if not tokens:
                continue
            results.append(
                {
                    "layer": li,
                    "expert": ei,
                    "n_activations": X.shape[0],
                    "tokens": tokens,
                    "evr": evr,
                }
            )

    print(f"Analyzed {len(results)} experts")

    evr_matrix = np.zeros((n_layers, n_experts))
    for r in results:
        evr_matrix[r["layer"], r["expert"]] = r["evr"][0] if r["evr"] else 0.0

    if output_dir:
        import json

        (output_dir / "results.json").write_text(json.dumps(results))
        np.save(output_dir / "evr_matrix.npy", evr_matrix)

        import plotly.express as px

        fig = px.imshow(
            evr_matrix,
            x=[f"E{i}" for i in range(n_experts)],
            y=[f"L{i}" for i in range(n_layers)],
            color_continuous_scale="Blues",
            labels=dict(x="Expert", y="Layer", color="Top EVR"),
            title="Expert Pursuit: Top Explained Variance Ratio per Expert",
        )
        fig.update_layout(width=1600, height=600)
        fig.write_html(output_dir / "evr_heatmap.html")

        print(f"Saved results to {output_dir}")

    return results, evr_matrix
