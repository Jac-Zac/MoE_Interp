"""Projection pursuit for Expert Pursuit."""

import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.cache import iter_layer_activations, load_metadata, load_unembedding
from src.environment import get_device
from src.plots import plot_evr_heatmap
from src.sparse_decomposition import SOMP


def projection_pursuit(
    X: torch.Tensor,
    dictionary: torch.Tensor,
    tokenizer,
    k: int = 50,
) -> tuple[list[str], list[float]]:
    """Greedy projection pursuit with SOMP."""
    if k <= 0 or X.shape[0] <= 1:
        return [], []

    X = X.float()
    device = get_device()
    if device.type == "mps":
        device = torch.device("cpu")

    total_var = X.var(dim=0).sum()
    if total_var < 1e-10:
        return [], []

    decomposition = SOMP(k=k, criterion="l1")
    result = decomposition(
        X=X,
        dictionary=dictionary,
        descriptors=list(range(len(dictionary))),
        device=device,
    )

    tokens = [tokenizer.decode([idx]).strip() for idx in result["chosen"].tolist()]
    evr_values = result["evr"].tolist()
    return tokens, evr_values


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
    dictionary = load_unembedding(data_dir / "unembedding" / "dictionary.h5")

    metadata_path = encodings_dir / "metadata.json"
    if not metadata_path.exists():
        raise ValueError(f"No metadata found in {encodings_dir}")
    metadata = load_metadata(metadata_path)
    n_layers = metadata["n_layers"]
    n_experts = metadata["n_experts"]

    total_experts = n_layers * n_experts

    results = []
    for li, ei, acts in tqdm(
        iter_layer_activations(encodings_dir, n_layers, n_experts, min_activations),
        desc="Projection pursuit",
        total=total_experts,
    ):
        X = acts.float()
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
    count_matrix = np.zeros((n_layers, n_experts))
    for r in results:
        evr_matrix[r["layer"], r["expert"]] = r["evr"][-1] if r["evr"] else 0.0
        count_matrix[r["layer"], r["expert"]] = r["n_activations"]

    if output_dir:
        (output_dir / "results.json").write_text(json.dumps(results))
        np.save(output_dir / "evr_matrix.npy", evr_matrix)
        np.save(output_dir / "count_matrix.npy", count_matrix)
        plot_evr_heatmap(
            evr_matrix, count_matrix, output_path=output_dir / "evr_heatmap.html"
        )
        print(f"Saved results to {output_dir}")

    return results, evr_matrix
