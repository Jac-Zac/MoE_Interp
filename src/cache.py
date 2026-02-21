"""Simple HDF5 storage for Expert Pursuit activations.

Per-layer files store [n_docs, n_experts, d_model] (default float16).
Per-expert files store variable-length activations (sparse, expert called multiple times).
Unembedding stored separately (default float32).
"""

import json
import logging
import warnings
from pathlib import Path

import h5py
import numpy as np
import torch

logger = logging.getLogger(__name__)


def save_metadata(
    output_dir: Path,
    n_docs: int,
    n_layers: int,
    n_experts: int,
    d_model: int,
    dtype: str = "float16",
) -> Path:
    """Save activation metadata to JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "metadata.json"
    meta = {
        "n_docs": n_docs,
        "n_layers": n_layers,
        "n_experts": n_experts,
        "d_model": d_model,
        "dtype": dtype,
    }
    with open(path, "w") as f:
        json.dump(meta, f)
    return path


def load_metadata(output_dir: Path) -> dict:
    """Load activation metadata from JSON."""
    path = Path(output_dir) / "metadata.json"
    with open(path) as f:
        return json.load(f)


def save_layer(
    output_dir: Path,
    layer: int,
    data: torch.Tensor,
    dtype: np.typing.DTypeLike = np.float16,
) -> Path:
    """Save a single layer's activations [n_docs, n_experts, d_model] to HDF5."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"layer_{layer:02d}.h5"
    np_dtype = np.dtype(dtype)
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "data", data=data.cpu().numpy().astype(np_dtype), dtype=np_dtype
        )
    return path


def load_layer(output_dir: Path, layer: int, device: str = "cpu") -> torch.Tensor:
    """Load a single layer's activations [n_docs, n_experts, d_model] from HDF5."""
    path = Path(output_dir) / f"layer_{layer:02d}.h5"
    with h5py.File(path, "r") as f:
        data = f["data"][:]
    return torch.from_numpy(data).float().to(device)


class ExpertActivationWriter:
    def __init__(
        self,
        path,
        n_experts: int,
        d_model: int,
        buffer_size: int = 512,
        dtype: np.typing.DTypeLike = np.float16,
    ):
        self.path = path
        self.n_experts = n_experts
        self.d_model = d_model
        self.buffer_size = buffer_size
        self.dtype = np.dtype(dtype)

        self.file = h5py.File(path, "w")
        self.datasets = {}
        self.buffers = {e: [] for e in range(n_experts)}

        for e in range(n_experts):
            self.datasets[e] = self.file.create_dataset(
                f"expert_{e:03d}",
                shape=(0, d_model),
                maxshape=(None, d_model),
                dtype=self.dtype,
                chunks=(buffer_size, d_model),
                compression="lzf",
            )

    def add(self, expert_idx: int, activation: torch.Tensor):
        arr = activation.cpu().numpy()
        if arr.dtype != self.dtype:
            warnings.warn(
                f"Activation dtype {arr.dtype} differs from stored dtype {self.dtype}, "
                f"converting automatically",
                RuntimeWarning,
                stacklevel=2,
            )
        self.buffers[expert_idx].append(arr)

        if len(self.buffers[expert_idx]) >= self.buffer_size:
            self.flush(expert_idx)

    def flush(self, expert_idx: int):
        if not self.buffers[expert_idx]:
            return

        data = np.stack(self.buffers[expert_idx]).astype(self.dtype)
        ds = self.datasets[expert_idx]

        old_size = ds.shape[0]
        new_size = old_size + data.shape[0]
        ds.resize((new_size, self.d_model))
        ds[old_size:new_size] = data

        self.buffers[expert_idx] = []

    def close(self):
        for e in range(self.n_experts):
            self.flush(e)
        self.file.close()


def save_unembedding(
    root_dir: Path,
    unembed: torch.Tensor,
    dtype: np.typing.DTypeLike = np.float32,
) -> Path:
    """Save unembedding matrix [vocab_size, d_model] to HDF5."""
    root_dir = Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    path = root_dir / "unembedding.h5"
    np_dtype = np.dtype(dtype)
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "data", data=unembed.cpu().numpy().astype(np_dtype), dtype=np_dtype
        )
    return path


def load_unembedding(root_dir: Path, device: str = "cpu") -> torch.Tensor:
    """Load unembedding matrix [vocab_size, d_model]."""
    path = Path(root_dir) / "unembedding.h5"
    with h5py.File(path, "r") as f:
        data = f["data"][:]
    return torch.from_numpy(data).float().to(device)
