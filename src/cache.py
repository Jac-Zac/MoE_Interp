"""Simple HDF5 storage for Expert Pursuit activations.

Per-layer files store [n_docs, n_experts, d_model] in float16.
Unembedding stored separately as float32.
"""

import json
from pathlib import Path

import h5py
import numpy as np
import torch


def save_layer(root_dir: Path, layer: int, data: torch.Tensor) -> None:
    """Save one layer's activations [n_docs, n_experts, d_model] to HDF5."""
    path = Path(root_dir) / "activations" / f"layer_{layer:02d}.h5"
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=data.cpu().numpy(), dtype=np.float16)


def load_layer(root_dir: Path, layer: int, device: str = "cpu") -> torch.Tensor:
    """Load one layer's activations [n_docs, n_experts, d_model]."""
    path = Path(root_dir) / "activations" / f"layer_{layer:02d}.h5"
    with h5py.File(path, "r") as f:
        data = f["data"][:]
    return torch.from_numpy(data).float().to(device)


def save_metadata(
    root_dir: Path,
    n_docs: int,
    n_layers: int,
    n_experts: int,
    d_model: int,
) -> None:
    """Save encoding metadata as JSON."""
    path = Path(root_dir) / "metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            {
                "n_docs": n_docs,
                "n_layers": n_layers,
                "n_experts": n_experts,
                "d_model": d_model,
            },
            f,
            indent=2,
        )


def load_metadata(root_dir: Path) -> dict:
    """Load encoding metadata."""
    with open(Path(root_dir) / "metadata.json") as f:
        return json.load(f)


def save_unembedding(root_dir: Path, unembed: torch.Tensor) -> Path:
    """Save unembedding matrix [vocab_size, d_model] to HDF5."""
    root_dir = Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    path = root_dir / "unembedding.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=unembed.cpu().numpy(), dtype=np.float32)
    return path


def load_unembedding(root_dir: Path, device: str = "cpu") -> torch.Tensor:
    """Load unembedding matrix [vocab_size, d_model]."""
    path = Path(root_dir) / "unembedding.h5"
    with h5py.File(path, "r") as f:
        data = f["data"][:]
    return torch.from_numpy(data).float().to(device)
