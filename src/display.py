"""Display utilities for Expert Pursuit traces."""

from pathlib import Path
from typing import Optional

from src.cache import DocumentTrace, list_all


def print_trace(trace: DocumentTrace) -> None:
    """Print summary of a DocumentTrace.

    Args:
        trace: DocumentTrace to display
    """
    print(f"Document {trace.doc_id}")
    print(f"  Layers: {trace.n_layers}")
    print(f"  Sequence length: {trace.seq_len}")
    print(f"  Top-k: {trace.k}")

    # Count active experts per layer
    print("\n  Active experts per layer:")
    for layer_idx, layer_traces in enumerate(trace.expert_traces):
        n_active = len(layer_traces)
        print(f"    Layer {layer_idx:2d}: {n_active:2d} active experts")


def print_trace_summary(data_dir: Optional[Path] = None) -> None:
    """Print summary of all available traces.

    Args:
        data_dir: Directory containing trace files
    """
    if data_dir is None:
        data_dir = Path("./data")

    doc_ids = list_all(data_dir)

    if not doc_ids:
        print("No traces found")
        return

    print(f"Available traces: {len(doc_ids)}")

    ids_str = ", ".join(str(d) for d in doc_ids[:10])
    if len(doc_ids) > 10:
        ids_str += f", ... ({len(doc_ids) - 10} more)"
    print(f"  IDs: {ids_str}")
