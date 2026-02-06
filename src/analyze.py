"""Simple analysis functions for expert activations."""

from collections import Counter

import torch

from src.cache import DocumentTrace


def get_expert_activations(
    trace: DocumentTrace,
    layer: int,
    expert_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find all token positions where a specific expert was activated.

    Args:
        trace: DocumentTrace to search
        layer: Layer index to search in
        expert_id: Expert ID to find

    Returns:
        Tuple of (positions [n_hits], weights [n_hits])
    """
    # Get expert indices for this layer: [seq_len, k]
    indices = trace.expert_indices[layer]

    # Find where this expert was active
    # Returns positions where expert_id appears
    positions, k_idx = (indices == expert_id).nonzero(as_tuple=True)

    # Get corresponding weights
    weights = trace.expert_weights[layer, positions, k_idx]

    return positions, weights


def get_expert_frequency(trace: DocumentTrace, layer: int) -> Counter:
    """Count how often each expert is activated in a layer.

    Args:
        trace: DocumentTrace to analyze
        layer: Layer index

    Returns:
        Counter mapping expert_id -> count
    """
    indices = trace.expert_indices[layer]  # [seq_len, k]
    flat_indices = indices.flatten().tolist()
    return Counter(flat_indices)


def get_top_experts(
    trace: DocumentTrace,
    layer: int,
    n: int = 10,
) -> list[tuple[int, int]]:
    """Get the top N most frequently activated experts in a layer.

    Args:
        trace: DocumentTrace to analyze
        layer: Layer index
        n: Number of top experts to return

    Returns:
        List of (expert_id, count) tuples, sorted by count descending
    """
    freq = get_expert_frequency(trace, layer)
    return freq.most_common(n)
