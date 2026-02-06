"""Pretty printing utilities for MoE traces."""

from pathlib import Path
from typing import Optional

from src.cache import DocumentTrace
from src.checkpoint import list_documents


def print_token(trace: DocumentTrace, layer: int, pos: int) -> None:
    """Print expert activations for a specific token.

    Args:
        trace: DocumentTrace to display
        layer: Layer index
        pos: Token position in sequence
    """
    expert_ids, weights = trace.get_token(layer, pos)

    print(f"Layer {layer}, Token {pos}")

    # Sort by weight descending
    sorted_indices = weights.argsort(descending=True)

    for rank, idx in enumerate(sorted_indices, 1):
        expert_id = expert_ids[idx].item()
        weight = weights[idx].item()
        bar = "█" * int(weight * 20)
        print(f"  {rank}. Expert {expert_id:2d}: {weight:.4f} {bar}")


def print_stats(trace: DocumentTrace) -> None:
    """Print detailed statistics for a document trace.

    Args:
        trace: DocumentTrace to analyze
    """
    import torch

    print(f"Document {trace.doc_id}")
    print(f"  Shape: [{trace.n_layers} layers, {trace.seq_len} tokens, k={trace.k}]")

    # Memory footprint
    total_bytes = (
        trace.expert_indices.numel() * trace.expert_indices.element_size()
        + trace.expert_weights.numel() * trace.expert_weights.element_size()
    )
    for unit in ["B", "KB", "MB", "GB"]:
        if total_bytes < 1024.0:
            size_str = f"{total_bytes:.1f} {unit}"
            break
        total_bytes /= 1024.0
    else:
        size_str = f"{total_bytes:.1f} TB"
    print(f"  Size: {size_str}")

    # Expert usage statistics
    print("\n  Expert Usage Statistics:")

    all_experts = trace.expert_indices.flatten()
    unique_experts = torch.unique(all_experts)
    n_unique = len(unique_experts)
    n_experts = int(trace.expert_indices.max().item()) + 1

    print(
        f"    Unique experts used: {n_unique}/{n_experts} ({100 * n_unique / n_experts:.1f}%)"
    )

    # Per-layer statistics
    print("\n    Per-layer distribution:")
    for layer_idx in range(min(3, trace.n_layers)):
        layer_experts = trace.expert_indices[layer_idx].flatten()
        layer_unique = len(torch.unique(layer_experts))
        print(f"      Layer {layer_idx:2d}: {layer_unique:2d}/{n_experts} experts")

    if trace.n_layers > 3:
        print(f"      ... ({trace.n_layers - 3} more layers)")

    # Top experts by total weight
    print("\n    Top 5 experts by total weight:")
    expert_weights_sum = torch.zeros(n_experts)
    for i in range(trace.n_layers):
        for j in range(trace.seq_len):
            experts, weights = trace.get_token(i, j)
            expert_weights_sum[experts] += weights

    top_k = min(5, n_experts)
    top_values, top_indices = torch.topk(expert_weights_sum, top_k)
    total_weight = expert_weights_sum.sum()

    for rank, (expert_id, weight) in enumerate(zip(top_indices, top_values), 1):
        pct = 100 * weight / total_weight if total_weight > 0 else 0
        bar = "█" * int(pct / 5)
        print(f"      {rank}. Expert {expert_id:2d}: {weight:8.3f} ({pct:5.1f}%) {bar}")


def print_doc_summary(data_dir: Optional[Path] = None, verbose: bool = False) -> None:
    """Print summary of all available documents.

    Args:
        data_dir: Directory containing trace files
        verbose: Whether to show detailed info for each document
    """
    from src.checkpoint import get_data_dir, load_document

    if data_dir is None:
        data_dir = get_data_dir()

    doc_ids = list_documents(data_dir)

    if not doc_ids:
        print("No documents found")
        return

    print(f"Available documents: {len(doc_ids)}")

    if verbose:
        print()
        for doc_id in doc_ids[:5]:
            try:
                trace = load_document(doc_id, data_dir)
                print(
                    f"  Doc {doc_id}: [{trace.n_layers}L x {trace.seq_len}T x k={trace.k}]"
                )
            except FileNotFoundError:
                print(f"  Doc {doc_id}: (not found)")

        if len(doc_ids) > 5:
            print(f"  ... and {len(doc_ids) - 5} more")
    else:
        ids_str = ", ".join(str(d) for d in doc_ids[:10])
        if len(doc_ids) > 10:
            ids_str += f", ... ({len(doc_ids) - 10} more)"
        print(f"  IDs: {ids_str}")
