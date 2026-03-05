"""Projection pursuit for Expert Pursuit."""

import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.cache import load_layer_h5, load_metadata, load_unembedding
from src.environment import get_device
from src.plots import plot_evr_heatmap
from src.sparse_decomposition import SOMP


def projection_pursuit(
    X: torch.Tensor,
    dictionary: torch.Tensor,
    tokenizer,
    device: torch.device,
    k: int = 50,
) -> tuple[list[str], list[float]]:
    """Greedy projection pursuit with SOMP.

    Args:
        X: Expert activations (n_samples × d_model).
        dictionary: Unembedding matrix. Should already be on the target device
            when called in a loop — avoids repeated host-to-device transfers.
        tokenizer: Tokenizer for decoding chosen atom indices.
        k: Number of atoms to select.
        device: Device to run on. If None, determined automatically (adds overhead
            per call — prefer passing a pre-determined device in loops).
    """
    if k <= 0 or X.shape[0] <= 1:
        return [], []

    X = X.float()

    total_var = X.var(dim=0).sum()
    if total_var < 1e-10:
        return [], []

    decomposition = SOMP(k=k, compute_evr=True)
    result = decomposition(
        X=X,
        dictionary=dictionary,
        descriptors=list(range(len(dictionary))),
        device=device,
    )

    tokens = [tokenizer.decode([idx]).strip() for idx in result["chosen"].tolist()]
    evr_values = result["evr"].tolist()
    return tokens, evr_values


def load_pursuit(pursuit_dir: Path) -> tuple[list[dict], np.ndarray]:
    """Load previously computed pursuit results from disk."""
    pursuit_dir = Path(pursuit_dir)
    results = []
    with open(pursuit_dir / "results.jsonl") as f:
        for line in f:
            results.append(json.loads(line))
    evr_matrix = np.load(pursuit_dir / "evr_matrix.npy")
    return results, evr_matrix


def run_pursuit(
    encodings_dir: Path,
    min_activations: int = 5,
    k: int = 50,
    output_dir: Path | None = None,
    data_dir: Path | None = None,
) -> tuple[list[dict], np.ndarray]:
    """Run projection pursuit on all experts.

    Args:
        encodings_dir: Directory containing expert encodings
        min_activations: Minimum activations required to analyze an expert
        k: Number of top tokens to return per expert
        output_dir: Optional output directory for results and plots
        data_dir: Data directory containing unembedding. If None, derived from encodings_dir.

    Returns:
        Tuple of (results list, evr_matrix)
    """
    from transformers import AutoTokenizer

    encodings_dir = Path(encodings_dir)
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    if data_dir is None:
        data_dir = encodings_dir.parent

    # Determine device and move dictionary to it once — avoids 1024 redundant
    # host-to-device transfers of the 393 MB unembedding matrix.
    # PERF: no longer forcing MPS to CPU here — somp() handles MPS internally
    # by only falling back to CPU for the lstsq solve (which needs float64).
    device = get_device()

    dictionary = (
        load_unembedding(data_dir / "unembedding" / "dictionary.h5").float().to(device)
    )

    metadata_path = encodings_dir / "metadata.json"
    if not metadata_path.exists():
        raise ValueError(f"No metadata found in {encodings_dir}")
    metadata = load_metadata(metadata_path)

    if "model_name" not in metadata:
        raise ValueError(
            "model_name not found in metadata. "
            "Please re-encode with a newer version that saves model_name."
        )

    tokenizer = AutoTokenizer.from_pretrained(metadata["model_name"])
    n_layers = metadata["n_layers"]
    n_experts = metadata["n_experts"]

    # Open the JSONL log file once if output_dir is set. Each expert result is
    # flushed immediately so progress is never lost if the run is interrupted.
    jsonl_file = None
    if output_dir:
        jsonl_path = output_dir / "results.jsonl"
        jsonl_file = open(jsonl_path, "w")

    results = []
    evr_matrix = np.zeros((n_layers, n_experts))
    count_matrix = np.zeros((n_layers, n_experts))
    try:
        for layer_idx in tqdm(range(n_layers), desc="Projection pursuit"):
            expert_acts = load_layer_h5(
                encodings_dir, layer_idx, n_experts, min_activations
            )
            for expert_idx, acts in tqdm(
                expert_acts.items(), desc=f"Layer {layer_idx}", leave=False
            ):
                X = acts.float()
                tokens, evr = projection_pursuit(
                    X, dictionary, tokenizer, device=device, k=k
                )
                if not tokens:
                    continue

                record = {
                    "layer": layer_idx,
                    "expert": expert_idx,
                    "n_activations": X.shape[0],
                    "tokens": tokens,
                    "evr": evr,
                }
                results.append(record)

                # Update matrices in-place
                evr_matrix[layer_idx, expert_idx] = evr[-1]
                count_matrix[layer_idx, expert_idx] = X.shape[0]

                if jsonl_file is not None:
                    # One JSON object per line — safe to append, readable mid-run.
                    jsonl_file.write(json.dumps(record) + "\n")
                    jsonl_file.flush()
    finally:
        if jsonl_file is not None:
            jsonl_file.close()

    print(f"Analyzed {len(results)} experts")

    if output_dir:
        np.save(output_dir / "evr_matrix.npy", evr_matrix)
        np.save(output_dir / "count_matrix.npy", count_matrix)
        plot_evr_heatmap(
            evr_matrix, count_matrix, output_path=output_dir / "evr_heatmap.html"
        )
        print(f"Saved results to {output_dir}")

    return results, evr_matrix
