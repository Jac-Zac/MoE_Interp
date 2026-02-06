"""Expert activation extraction utilities."""

from math import ceil
from typing import List

import nnsight
import torch
from nnsight import LanguageModel
from tqdm import tqdm

from src.cache import MoETrace


def process_batch(model: LanguageModel, batch_docs: List[List[int]]) -> MoETrace:
    """Process a single batch and return expert trace.

    Args:
        model: nnsight LanguageModel
        batch_docs: List of document token ID lists

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
    )


def capture_moe_activations(
    model: LanguageModel,
    docs: List[List[int]],
    batch_size: int = 8,
) -> List[MoETrace]:
    """Capture MoE activations for documents in batches.

    Args:
        model: nnsight LanguageModel
        docs: List of document token ID lists
        batch_size: Number of documents per batch

    Returns:
        List of MoETrace objects, one per batch
    """
    n_batches = ceil(len(docs) / batch_size)
    all_traces = []

    for batch_idx in tqdm(range(n_batches), desc="Processing batches"):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(docs))
        batch_docs = docs[start:end]

        trace = process_batch(model, batch_docs)
        all_traces.append(trace)

        tqdm.write(
            f"Batch {batch_idx + 1}/{n_batches} (docs {start}-{end - 1}): "
            f"{len(trace.token_ids)} tokens"
        )

    return all_traces
