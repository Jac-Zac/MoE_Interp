"""Projection pursuit for Expert Pursuit."""

import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.cache import load_layer_h5, load_metadata, load_unembedding
from src.concepts import CONCEPT_WORDS
from src.environment import get_device
from src.sparse_decomposition import SOMP


def projection_pursuit(
    X: torch.Tensor,
    dictionary: torch.Tensor,
    tokenizer,
    device: torch.device | str,
    k: int = 50,
    token_ids: list[int] | None = None,
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
    if X.device.type == "cpu" and X.var(dim=0).sum() < 1e-10:
        return [], []

    decomposition = SOMP(k=k, compute_evr=True, return_full=False)
    result = decomposition(
        X=X,
        dictionary=dictionary,
        descriptors=list(range(len(dictionary))),
        device=device,
    )

    tokens = []
    for idx in result["chosen"].tolist():
        token_id = idx if token_ids is None else token_ids[idx]
        tokens.append(tokenizer.decode([token_id]).strip())
    evr_values = result["evr"].tolist()
    return tokens, evr_values


def _build_dictionary(
    dictionary: torch.Tensor,
    tokenizer,
    concept: str | None,
) -> tuple[torch.Tensor, list[int] | None]:
    if concept is None:
        return dictionary, None

    if concept not in CONCEPT_WORDS:
        options = ", ".join(sorted(CONCEPT_WORDS))
        raise ValueError(f"Unknown concept '{concept}'. Available concepts: {options}")

    token_ids: set[int] = set()
    for word in CONCEPT_WORDS[concept]:
        token_ids.update(tokenizer(word, add_special_tokens=False)["input_ids"])

    sorted_token_ids = sorted(token_ids)
    if not sorted_token_ids:
        raise ValueError(f"Concept '{concept}' produced no token ids")

    return dictionary[sorted_token_ids], sorted_token_ids


def load_pursuit(pursuit_dir: Path) -> tuple[list[dict], np.ndarray, np.ndarray | None]:
    """Load previously computed pursuit results from disk."""
    pursuit_dir = Path(pursuit_dir)
    results = []
    with open(pursuit_dir / "results.jsonl") as f:
        for line in f:
            results.append(json.loads(line))
    evr_matrix = np.load(pursuit_dir / "evr_matrix.npy")
    count_path = pursuit_dir / "count_matrix.npy"
    count_matrix = np.load(count_path) if count_path.exists() else None
    return results, evr_matrix, count_matrix


def run_pursuit(
    encodings_dir: Path,
    min_activations: int = 5,
    k: int = 50,
    output_dir: Path | None = None,
    data_dir: Path | None = None,
    concept: str | None = None,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Run projection pursuit on all experts.

    Args:
        encodings_dir: Directory containing expert encodings
        min_activations: Minimum activations required to analyze an expert
        k: Number of top tokens to return per expert
        output_dir: If set, results.jsonl is written incrementally (flush per expert)
            so progress is never lost if the run is interrupted.
        data_dir: Data directory containing unembedding. If None, derived from encodings_dir.
        concept: Optional concept name to restrict the unembedding dictionary.
            Must be a key in CONCEPT_WORDS (e.g. "offensive", "countries", "numbers").

    Returns:
        Tuple of (results list, evr_matrix, count_matrix)
    """
    from transformers import AutoTokenizer

    encodings_dir = Path(encodings_dir)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    if data_dir is None:
        data_dir = encodings_dir.parent

    # Determine device and move dictionary to it once — avoids 1024 redundant
    # host-to-device transfers of the 393 MB unembedding matrix.
    # PERF: no longer forcing MPS to CPU here — somp() handles MPS internally
    # by only falling back to CPU for the lstsq solve (which needs float64).
    device = get_device()

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
    dictionary = (
        load_unembedding(data_dir / "unembedding" / "dictionary.h5").float().to(device)
    )
    # When a concept is given, restrict the dictionary to tokens for that concept's
    # word list and keep sorted_token_ids for remapping SOMP row indices back to
    # full-vocabulary ids at decode time (mirrors HeadPursuit's tokens_data[token]).
    dictionary, token_ids = _build_dictionary(dictionary, tokenizer, concept)
    n_layers = metadata["n_layers"]
    n_experts = metadata["n_experts"]

    # Open the JSONL log file once if output_dir is set. Each expert result is
    # flushed immediately so progress is never lost if the run is interrupted.
    jsonl_file = None
    if output_dir is not None:
        jsonl_file = open(output_dir / "results.jsonl", "w")

    results = []
    evr_matrix = np.zeros((n_layers, n_experts))
    count_matrix = np.zeros((n_layers, n_experts))
    k = min(k, dictionary.shape[0])
    try:
        for layer_idx in tqdm(range(n_layers), desc="Projection pursuit"):
            expert_acts = load_layer_h5(
                encodings_dir, layer_idx, n_experts, min_activations
            )
            for expert_idx, acts in tqdm(
                expert_acts.items(), desc=f"Layer {layer_idx}", leave=False
            ):
                acts = acts.float()
                # PERF: Keep the zero-variance guard on CPU and skip the device sync.
                if acts.var(dim=0).sum() < 1e-10:
                    continue
                X = acts.to(device)
                tokens, evr = projection_pursuit(
                    X,
                    dictionary,
                    tokenizer,
                    device=device,
                    k=k,
                    token_ids=token_ids,
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
    return results, evr_matrix, count_matrix
