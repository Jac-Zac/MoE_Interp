"""Projection pursuit for Expert Pursuit."""

import json
from pathlib import Path

import numpy as np
import torch
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from src.cache import load_layer_h5, load_metadata, load_unembedding
from src.concepts import CONCEPT_WORDS
from src.environment import get_device, get_unembedding_dir
from src.sparse_decomposition import SOMP
from src.word_dictionary import WordDictionary


def projection_pursuit(
    X: torch.Tensor,
    dictionary: torch.Tensor,
    tokenizer,
    device: torch.device | str,
    k: int = 50,
    token_ids: list[int] | None = None,
    labels: list[str] | None = None,
    base_vocab_size: int | None = None,
) -> tuple[list[str], list[float]]:
    """Greedy projection pursuit with SOMP.

    Args:
        X: Expert activations (n_samples × d_model).
        dictionary: Unembedding matrix. Should already be on the target device
            when called in a loop — avoids repeated host-to-device transfers.
        tokenizer: Tokenizer for decoding chosen atom indices.
        k: Number of atoms to select.
        device: Device to run on.
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
        if (
            labels is not None
            and base_vocab_size is not None
            and idx >= base_vocab_size
        ):
            tokens.append(labels[idx - base_vocab_size])
            continue
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
        token_ids.update(tokenizer(word, add_special_tokens=False).input_ids)

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
    extractions_dir: Path,
    min_activations: int = 5,
    k: int = 50,
    output_dir: Path | None = None,
    concept: str | None = None,
    word_dictionary: WordDictionary | None = None,
    tokenizer=None,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Run projection pursuit on all experts.

    Args:
        extractions_dir: Directory containing expert extractions
        min_activations: Minimum activations required to analyze an expert
        k: Number of top tokens to return per expert
        output_dir: If set, results.jsonl is written incrementally (flush per expert)
            so progress is never lost if the run is interrupted.
        concept: Optional concept name to restrict the unembedding dictionary.
            Must be a key in CONCEPT_WORDS (e.g. "offensive", "countries", "numbers").

    Returns:
        Tuple of (results list, evr_matrix, count_matrix)
    """
    extractions_dir = Path(extractions_dir)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Determine device and move dictionary to it once — avoids 1024 redundant
    # host-to-device transfers of the 393 MB unembedding matrix.
    # PERF: no longer forcing MPS to CPU here — somp() handles MPS internally
    # by only falling back to CPU for the lstsq solve (which needs float64).
    device = get_device()

    metadata_path = extractions_dir / "metadata.json"
    if not metadata_path.exists():
        raise ValueError(f"No metadata found in {extractions_dir}")
    metadata = load_metadata(metadata_path)

    if "model_name" not in metadata:
        raise ValueError(
            "model_name not found in metadata. "
            "Please re-extract with a newer version that saves model_name."
        )

    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(metadata["model_name"])
    if word_dictionary is None:
        dictionary = (
            load_unembedding(
                get_unembedding_dir(metadata["model_name"]) / "dictionary.h5",
            )
            .float()
            .to(device)
        )
        # When a concept is given, restrict the dictionary to tokens for that concept's
        # word list and keep sorted_token_ids for remapping SOMP row indices back to
        # full-vocabulary ids at decode time (mirrors HeadPursuit's tokens_data[token]).
        dictionary, token_ids = _build_dictionary(dictionary, tokenizer, concept)
        labels = None
        base_vocab_size = None
    else:
        dictionary = word_dictionary.embeddings.float().to(device)
        labels = word_dictionary.labels
        base_vocab_size = word_dictionary.base_vocab_size
        if base_vocab_size > dictionary.shape[0]:
            raise ValueError("word_dictionary base_vocab_size exceeds embedding rows")
        if len(labels) != dictionary.shape[0] - base_vocab_size:
            raise ValueError(
                "word_dictionary labels must match appended embedding rows"
            )
        token_ids = word_dictionary.kept_token_ids

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
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        ) as progress:
            layer_task = progress.add_task("Projection pursuit", total=n_layers)

            for layer_idx in range(n_layers):
                expert_acts = load_layer_h5(
                    extractions_dir, layer_idx, n_experts, min_activations
                )
                expert_task = progress.add_task(
                    f"Layer {layer_idx}", total=len(expert_acts), parent=layer_task
                )

                for expert_idx, acts in expert_acts.items():
                    X = acts.float().to(device)
                    tokens, evr = projection_pursuit(
                        X,
                        dictionary,
                        tokenizer,
                        device=device,
                        k=k,
                        token_ids=token_ids,
                        labels=labels,
                        base_vocab_size=base_vocab_size,
                    )
                    if not tokens:
                        progress.advance(expert_task)
                        continue

                    record = {
                        "layer": layer_idx,
                        "expert": expert_idx,
                        "n_activations": X.shape[0],
                        "tokens": tokens,
                        "evr": evr,
                    }
                    results.append(record)

                    evr_matrix[layer_idx, expert_idx] = evr[-1]
                    count_matrix[layer_idx, expert_idx] = X.shape[0]

                    if jsonl_file is not None:
                        jsonl_file.write(json.dumps(record) + "\n")
                        jsonl_file.flush()

                    progress.advance(expert_task)

                progress.remove_task(expert_task)
                progress.advance(layer_task)
    finally:
        if jsonl_file is not None:
            jsonl_file.close()

    print(f"Analyzed {len(results)} experts")

    if output_dir is not None:
        np.save(output_dir / "evr_matrix.npy", evr_matrix)
        np.save(output_dir / "count_matrix.npy", count_matrix)

    return results, evr_matrix, count_matrix
