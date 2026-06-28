"""Projection pursuit for Expert Pursuit."""

import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from moe_interp.capture.cache import (
    load_layer_activations,
    load_metadata,
    load_unembedding,
)
from moe_interp.config import get_device, get_unembedding_dir
from moe_interp.pursuit.concepts import CONCEPT_WORDS
from moe_interp.pursuit.decomposition import SOMP


def _empty_device_cache(device: torch.device) -> None:
    """Return freed blocks to the OS so the caching allocator's high-water mark
    doesn't accumulate one `cross` matrix per expert.

    Only done on MPS, whose allocator fragments badly across the per-expert
    size changes. CUDA's caching allocator handles this fine, and a per-expert
    `empty_cache()` there only forces a synchronizing free + re-malloc."""
    if device.type == "mps":
        torch.mps.empty_cache()


def projection_pursuit(
    X: torch.Tensor,
    dictionary: torch.Tensor,
    tokenizer,
    device: torch.device | str,
    k: int = 50,
    pc: int | None = None,
    token_ids: list[int] | None = None,
    labels: list[str] | None = None,
    base_vocab_size: int | None = None,
    dict_t: torch.Tensor | None = None,
) -> tuple[list[str], list[float]]:
    """Greedy projection pursuit with SOMP.

    Args:
        X: Expert activations (n_samples × d_model).
        dictionary: Unembedding matrix. Should already be on the target device
            when called in a loop — avoids repeated host-to-device transfers.
        tokenizer: Tokenizer for decoding base-vocab atom indices.
        k: Number of atoms to select.
        device: Device to run on.
        dict_t: Optional precomputed dictionary transpose, shared across experts.
    """
    if k <= 0 or X.shape[0] <= 1:
        return [], []

    X = X.float()
    if X.var(dim=0).sum().item() < 1e-10:
        return [], []

    decomposition = SOMP(k=k, pc=pc, compute_evr=True, return_full=False)
    result = decomposition(
        X=X,
        dictionary=dictionary,
        descriptors=list(range(len(dictionary))),
        device=device,
        dict_t=dict_t,
    )

    tokens = [
        _decode_atom(
            idx,
            tokenizer=tokenizer,
            token_ids=token_ids,
            labels=labels,
            base_vocab_size=base_vocab_size,
        )
        for idx in result["chosen"].tolist()
    ]
    return tokens, result["evr"].tolist()


def _decode_atom(
    idx: int,
    tokenizer,
    token_ids: list[int] | None,
    labels: list[str] | None,
    base_vocab_size: int | None,
) -> str:
    """Resolve a chosen atom index to its label.

    The dictionary is ``[base-vocab rows | appended label atoms]``; ``base_vocab_size``
    is where the appended labels start (0 in concept mode, where every atom is a label).
    Atoms at or past that boundary decode from ``labels``; earlier atoms are base-vocab
    tokens decoded via ``token_ids`` (row→id remap) or the raw index.
    """
    base = base_vocab_size or 0
    if labels is not None and idx >= base:
        return labels[idx - base]
    token_id = idx if token_ids is None else token_ids[idx]
    return tokenizer.decode([token_id])


def _build_concept_dictionary(
    concept: str,
    dictionary: torch.Tensor,
    tokenizer,
    device: torch.device | str,
) -> tuple[torch.Tensor, list[str]]:
    """Restrict the unembedding dictionary to a concept's word atoms.

    Single-token words map directly to their unembedding row; multi-token words
    are averaged across their token rows and renormalized. Returns the
    device-placed atom matrix and the matching labels.
    """
    if concept not in CONCEPT_WORDS:
        options = ", ".join(sorted(CONCEPT_WORDS))
        raise ValueError(f"Unknown concept '{concept}'. Available concepts: {options}")

    single_labels: list[tuple[str, int]] = []
    multi_labels: list[str] = []
    multi_token_ids: list[list[int]] = []
    for w in CONCEPT_WORDS[concept]:
        tokens = tokenizer(w, add_special_tokens=False).input_ids
        if len(tokens) == 1:
            single_labels.append((w, tokens[0]))
        else:
            multi_labels.append(w)
            multi_token_ids.append(tokens)

    single_atoms = dictionary[[tid for _, tid in single_labels]].float()
    if multi_token_ids:
        multi_atoms = torch.stack(
            [dictionary[tids].mean(dim=0) for tids in multi_token_ids],
        ).float()
        multi_atoms = torch.nn.functional.normalize(multi_atoms, dim=1)
    else:
        multi_atoms = torch.empty(
            (0, dictionary.shape[1]),
            dtype=dictionary.dtype,
            device=dictionary.device,
        )

    dictionary = torch.cat([single_atoms, multi_atoms], dim=0).to(device)
    labels = [w for w, _ in single_labels] + multi_labels
    print(
        f"Concept '{concept}': {len(labels)} atoms "
        f"({len(multi_labels)} multi-token averaged)"
    )
    return dictionary, labels


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
        tokenizer: Tokenizer for decoding chosen atom indices. If None, loaded from
            AutoTokenizer using the model name stored in extractions_dir/metadata.json.

    Returns:
        Tuple of (results list, evr_matrix, count_matrix)
    """
    extractions_dir = Path(extractions_dir)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Determine device and move dictionary to it once — avoids 1024 redundant
    # host-to-device transfers of the 393 MB unembedding matrix. Device handling lives
    # in somp(): on MPS it falls back to CPU only for the float64 lstsq solve.
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
    dictionary = load_unembedding(
        get_unembedding_dir(metadata["model_name"]) / "dictionary.h5",
    ).float()
    token_ids = None
    if concept is not None:
        dictionary, labels = _build_concept_dictionary(
            concept, dictionary, tokenizer, device
        )
        base_vocab_size = 0
    else:
        labels = None
        base_vocab_size = None
        dictionary = dictionary.to(device)

    n_layers = metadata["n_layers"]
    n_experts = metadata["n_experts"]

    results = []
    evr_matrix = np.zeros((n_layers, n_experts))
    count_matrix = np.zeros((n_layers, n_experts))
    k = min(k, dictionary.shape[0])

    # Transpose the dictionary once and reuse it for every expert — otherwise
    # somp() rebuilds this ~400 MB contiguous copy on all 1024 expert calls.
    dict_t = dictionary.T.contiguous()

    jsonl_path = output_dir / "results.jsonl" if output_dir is not None else None
    with open(jsonl_path, "w") if jsonl_path else nullcontext() as jsonl_file:
        for layer_idx in tqdm(range(n_layers), desc="Projection pursuit"):
            expert_acts = load_layer_activations(
                extractions_dir, layer_idx, n_experts, min_activations
            )

            for expert_idx, acts in expert_acts.items():
                X = acts.float().to(device)
                n_acts = X.shape[0]
                tokens, evr = projection_pursuit(
                    X,
                    dictionary,
                    tokenizer,
                    device=device,
                    k=k,
                    token_ids=token_ids,
                    labels=labels,
                    base_vocab_size=base_vocab_size,
                    dict_t=dict_t,
                )
                # Free this expert's device tensors before the next expert
                # allocates a differently-sized `cross` matrix. Without this,
                # the caching allocator accumulates ~ one block per expert
                del X
                _empty_device_cache(device)
                if not tokens:
                    continue

                record = {
                    "layer": layer_idx,
                    "expert": expert_idx,
                    "n_activations": n_acts,
                    "tokens": tokens,
                    "evr": evr,
                }
                results.append(record)

                evr_matrix[layer_idx, expert_idx] = evr[-1]
                count_matrix[layer_idx, expert_idx] = n_acts

                if jsonl_file is not None:
                    jsonl_file.write(json.dumps(record) + "\n")
                    jsonl_file.flush()

    print(f"Analyzed {len(results)} experts")

    if output_dir is not None:
        np.save(output_dir / "evr_matrix.npy", evr_matrix)
        np.save(output_dir / "count_matrix.npy", count_matrix)

    return results, evr_matrix, count_matrix
