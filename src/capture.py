"""Expert activation extraction utilities."""

from math import ceil
from pathlib import Path
from typing import List, Optional, Tuple

import nnsight
import torch
from nnsight import LanguageModel
from tqdm import tqdm

from src.cache import MoETrace
from src.checkpoint import save_checkpoint, save_manifest


def process_batch(
    model: LanguageModel,
    batch_docs: List[List[int]],
    doc_source_ids: List[int],
) -> MoETrace:
    """Process a single batch and return expert trace.

    Args:
        model: nnsight LanguageModel
        batch_docs: List of document token ID lists
        doc_source_ids: Original dataset indices for each document

    Returns:
        MoETrace with expert routing information
    """
    doc_lengths = torch.tensor([len(doc) for doc in batch_docs])
    batch_boundaries = torch.cat([torch.tensor([0]), doc_lengths.cumsum(0)])

    with torch.no_grad(), model.trace(batch_docs) as tracer:
        layer_indices, layer_weights = [], []

        for layer in model.model.layers:
            _, weights, indices = layer.mlp.source.self_gate_0.output
            layer_indices.append(indices)
            layer_weights.append(weights)

        indices_stack = torch.stack(layer_indices, dim=0)
        weights_stack = torch.stack(layer_weights, dim=0)

        nnsight.save(indices_stack)
        nnsight.save(weights_stack)

    return MoETrace.build(
        docs=batch_docs,
        indices_stack=indices_stack,
        weights_stack=weights_stack,
        doc_boundaries=batch_boundaries,
        doc_source_ids=doc_source_ids,
    )


def capture_moe_activations(
    model: LanguageModel,
    docs: List[List[int]],
    doc_source_ids: List[int],
    batch_size: int = 8,
    save_dir: Optional[Path] = None,
) -> List[dict]:
    """Capture MoE activations for documents in batches.

    Args:
        model: nnsight LanguageModel
        docs: List of document token ID lists
        doc_source_ids: Original dataset indices for each document
        batch_size: Number of documents per batch
        save_dir: If provided, save each batch to disk

    Returns:
        List of dicts with batch info (idx, docs_range, filepath if saved)
    """
    n_batches = ceil(len(docs) / batch_size)
    batch_info = []

    for batch_idx in tqdm(range(n_batches), desc="Processing batches"):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(docs))
        batch_docs = docs[start:end]
        batch_source_ids = doc_source_ids[start:end]

        trace = process_batch(model, batch_docs, batch_source_ids)

        info = {
            "batch_idx": batch_idx,
            "docs_range": (start, end),
            "n_tokens": len(trace.token_ids),
        }

        if save_dir is not None:
            filepath = save_checkpoint(
                trace,
                batch_idx=batch_idx,
                docs_range=(start, end),
                data_dir=save_dir,
            )
            info["filepath"] = filepath
            tqdm.write(
                f"Batch {batch_idx + 1}/{n_batches} (docs {start}-{end - 1}): "
                f"{len(trace.token_ids)} tokens -> {filepath.name}"
            )
        else:
            tqdm.write(
                f"Batch {batch_idx + 1}/{n_batches} (docs {start}-{end - 1}): "
                f"{len(trace.token_ids)} tokens"
            )

        batch_info.append(info)

    if save_dir is not None:
        save_manifest(
            total_batches=n_batches,
            total_docs=len(docs),
            data_dir=save_dir,
        )

    return batch_info
