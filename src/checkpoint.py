"""Checkpoint management for MoE trace batch processing."""

import json
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from safetensors.torch import load_file, save_file

from src.cache import MoETrace


def get_data_dir() -> Path:
    """Get data directory from MOE_DATA_DIR, DATA_DIR, or ./data."""
    data_dir = os.environ.get("MOE_DATA_DIR") or os.environ.get("DATA_DIR", "./data")
    path = Path(data_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_checkpoint(
    trace: MoETrace,
    batch_idx: int,
    docs_range: Tuple[int, int],
    data_dir: Optional[Path] = None,
) -> Path:
    """Save a MoETrace batch checkpoint to disk."""
    if data_dir is None:
        data_dir = get_data_dir()

    start_doc, end_doc = docs_range
    filename = f"moe_trace_batch_{batch_idx:04d}_docs_{start_doc}_{end_doc}.safetensors"
    filepath = data_dir / filename

    tensors = {
        "token_ids": trace.token_ids,
        "expert_indices": trace.expert_indices,
        "expert_weights": trace.expert_weights,
        "doc_boundaries": trace.doc_boundaries,
        "doc_source_ids": trace.doc_source_ids,
    }

    save_file(tensors, filepath)
    return filepath


def list_checkpoints(data_dir: Optional[Path] = None) -> List[Path]:
    """List all checkpoint files in data directory."""
    if data_dir is None:
        data_dir = get_data_dir()

    checkpoints = sorted(data_dir.glob("moe_trace_batch_*.safetensors"))
    return checkpoints


def _parse_checkpoint_filename(filename: str) -> Tuple[int, Tuple[int, int]]:
    """Parse batch index and docs range from checkpoint filename."""
    pattern = r"moe_trace_batch_(\d+)_docs_(\d+)_(\d+)\.safetensors"
    match = re.match(pattern, filename)
    if match:
        batch_idx = int(match.group(1))
        start_doc = int(match.group(2))
        end_doc = int(match.group(3))
        return batch_idx, (start_doc, end_doc)
    raise ValueError(f"Could not parse checkpoint filename: {filename}")


def load_checkpoint(filepath: Path) -> Tuple[MoETrace, int, Tuple[int, int]]:
    """Load a checkpoint file."""
    tensors = load_file(filepath)

    trace = MoETrace(
        token_ids=tensors["token_ids"],
        expert_indices=tensors["expert_indices"],
        expert_weights=tensors["expert_weights"],
        doc_boundaries=tensors["doc_boundaries"],
        doc_source_ids=tensors.get("doc_source_ids", torch.tensor([])),
    )

    batch_idx, docs_range = _parse_checkpoint_filename(filepath.name)
    return trace, batch_idx, docs_range


def merge_checkpoints(
    checkpoint_paths: Optional[List[Path]] = None,
    data_dir: Optional[Path] = None,
) -> MoETrace:
    """Merge multiple checkpoints into a single MoETrace."""
    if checkpoint_paths is None:
        checkpoint_paths = list_checkpoints(data_dir)

    if not checkpoint_paths:
        raise ValueError("No checkpoints found to merge")

    checkpoints_data = []
    for path in checkpoint_paths:
        trace, batch_idx, docs_range = load_checkpoint(path)
        checkpoints_data.append((batch_idx, trace, docs_range))

    checkpoints_data.sort(key=lambda x: x[0])

    all_token_ids = []
    all_expert_indices = []
    all_expert_weights = []
    all_doc_boundaries = [0]
    all_doc_source_ids = []

    current_total_tokens = 0

    for batch_idx, trace, (start_doc, end_doc) in checkpoints_data:
        all_token_ids.append(trace.token_ids)
        all_expert_indices.append(trace.expert_indices)
        all_expert_weights.append(trace.expert_weights)
        all_doc_source_ids.append(trace.doc_source_ids)

        adjusted_boundaries = trace.doc_boundaries[1:] + current_total_tokens
        all_doc_boundaries.extend(adjusted_boundaries.tolist())

        current_total_tokens += len(trace.token_ids)

    merged_token_ids = torch.cat(all_token_ids, dim=0)
    merged_expert_indices = torch.cat(all_expert_indices, dim=1)
    merged_expert_weights = torch.cat(all_expert_weights, dim=1)
    merged_doc_boundaries = torch.tensor(all_doc_boundaries)
    merged_doc_source_ids = torch.cat(all_doc_source_ids, dim=0)

    return MoETrace(
        token_ids=merged_token_ids,
        expert_indices=merged_expert_indices,
        expert_weights=merged_expert_weights,
        doc_boundaries=merged_doc_boundaries,
        doc_source_ids=merged_doc_source_ids,
    )


def save_manifest(
    total_batches: int,
    total_docs: int,
    data_dir: Optional[Path] = None,
) -> Path:
    """Save a manifest file tracking all batches."""
    if data_dir is None:
        data_dir = get_data_dir()

    manifest = {
        "total_batches": total_batches,
        "total_docs": total_docs,
        "checkpoints": [str(p.name) for p in list_checkpoints(data_dir)],
    }

    manifest_path = data_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest_path


def list_batch_info(data_dir: Optional[Path] = None) -> list[dict]:
    """List all available batches with metadata.

    Returns:
        List of dicts with batch_idx, doc_range, filename for each batch.
    """
    if data_dir is None:
        data_dir = get_data_dir()

    checkpoints = []
    for path in sorted(data_dir.glob("moe_trace_batch_*.safetensors")):
        try:
            batch_idx, docs_range = _parse_checkpoint_filename(path.name)
            checkpoints.append(
                {
                    "batch_idx": batch_idx,
                    "doc_range": docs_range,
                    "filename": path.name,
                    "filepath": str(path),
                }
            )
        except ValueError:
            continue

    return checkpoints


def load_batch(
    batch_idx: int, data_dir: Optional[Path] = None
) -> Tuple[MoETrace, Tuple[int, int]]:
    """Load a specific batch by index.

    Args:
        batch_idx: The batch index to load
        data_dir: Directory containing checkpoints

    Returns:
        Tuple of (MoETrace, doc_range)

    Raises:
        FileNotFoundError: If batch_idx doesn't exist
    """
    if data_dir is None:
        data_dir = get_data_dir()

    batches = list_batch_info(data_dir)

    for batch_info in batches:
        if batch_info["batch_idx"] == batch_idx:
            trace, _, docs_range = load_checkpoint(Path(batch_info["filepath"]))
            return trace, docs_range

    raise FileNotFoundError(f"Batch {batch_idx} not found in {data_dir}")
