"""HDF5-backed storage for Expert Pursuit activations.

Stores per-document mean gated expert outputs in memory-mapped HDF5 files,
organized per-layer for efficient SOMP iteration over experts.
"""

import json
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np
import torch


class ExpertActivationStore:
    """HDF5-backed store for aggregated expert activations.

    Storage layout:
        root_dir/
            expert_activations/layer_{ll}.h5  — [n_docs, n_experts, d_model] float16
            routing/routing_counts.h5         — [n_layers, n_docs, n_experts] int16
            composition.json                  — metadata
            doc_ids.json                      — row index -> dataset doc_id
    """

    def __init__(
        self,
        root_dir: Path,
        n_layers: int,
        n_experts: int,
        d_model: int,
        n_docs_estimate: int = 1000,
    ):
        self.root_dir = Path(root_dir)
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.d_model = d_model
        self._n_docs = 0

        # Create directories
        acts_dir = self.root_dir / "expert_activations"
        routing_dir = self.root_dir / "routing"
        acts_dir.mkdir(parents=True, exist_ok=True)
        routing_dir.mkdir(parents=True, exist_ok=True)

        # Create per-layer HDF5 files for expert activations
        chunk_docs = min(max(n_docs_estimate // 4, 1), 1000)
        self._layer_files: list[h5py.File] = []
        for layer_idx in range(n_layers):
            path = acts_dir / f"layer_{layer_idx:02d}.h5"
            f = h5py.File(path, "w")
            f.create_dataset(
                "data",
                shape=(0, n_experts, d_model),
                maxshape=(None, n_experts, d_model),
                dtype=np.float16,
                chunks=(chunk_docs, 1, d_model),
            )
            self._layer_files.append(f)

        # Routing counts: [n_layers, n_docs, n_experts]
        routing_path = routing_dir / "routing_counts.h5"
        self._routing_file = h5py.File(routing_path, "w")
        self._routing_file.create_dataset(
            "data",
            shape=(n_layers, 0, n_experts),
            maxshape=(n_layers, None, n_experts),
            dtype=np.int16,
            chunks=(n_layers, chunk_docs, 1),
        )

        # Doc ID tracking
        self._doc_ids: list[int] = []

    @property
    def n_docs(self) -> int:
        return self._n_docs

    def add_document(
        self,
        expert_means: torch.Tensor,
        routing_counts: torch.Tensor,
        doc_id: int,
    ) -> None:
        """Append one document's aggregated data.

        Args:
            expert_means: [n_layers, n_experts, d_model] mean gated outputs
            routing_counts: [n_layers, n_experts] token counts per expert
            doc_id: Original dataset document index
        """
        idx = self._n_docs
        self._n_docs += 1
        self._doc_ids.append(doc_id)

        # Resize and write per-layer activation data
        for layer_idx in range(self.n_layers):
            ds = self._layer_files[layer_idx]["data"]
            ds.resize(self._n_docs, axis=0)
            ds[idx] = expert_means[layer_idx].cpu().numpy().astype(np.float16)

        # Resize and write routing counts
        rds = self._routing_file["data"]
        rds.resize(self._n_docs, axis=1)
        rds[:, idx, :] = routing_counts.cpu().numpy().astype(np.int16)

    def flush(self) -> None:
        """Force write all buffered data to disk."""
        for f in self._layer_files:
            f.flush()
        self._routing_file.flush()

    def close(self) -> None:
        """Finalize: save metadata and close all files."""
        # Save composition metadata
        metadata = {
            "n_layers": self.n_layers,
            "n_experts": self.n_experts,
            "d_model": self.d_model,
            "n_docs": self._n_docs,
        }
        with open(self.root_dir / "composition.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save doc ID mapping
        with open(self.root_dir / "doc_ids.json", "w") as f:
            json.dump(self._doc_ids, f)

        # Close HDF5 files
        for f in self._layer_files:
            f.close()
        self._routing_file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # --- Read API (class methods for loading saved data) ---

    @staticmethod
    def load_expert(
        root_dir: Path,
        layer: int,
        expert_id: int,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Load one expert's activations across all documents.

        Args:
            root_dir: Store root directory
            layer: Layer index
            expert_id: Expert index
            device: Target device

        Returns:
            Tensor [n_docs, d_model]
        """
        path = root_dir / "expert_activations" / f"layer_{layer:02d}.h5"
        with h5py.File(path, "r") as f:
            data = f["data"][:, expert_id, :]  # [n_docs, d_model]
        return torch.from_numpy(data).float().to(device)

    @staticmethod
    def load_layer(
        root_dir: Path,
        layer: int,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Load all expert activations for one layer.

        Args:
            root_dir: Store root directory
            layer: Layer index
            device: Target device

        Returns:
            Tensor [n_docs, n_experts, d_model]
        """
        path = root_dir / "expert_activations" / f"layer_{layer:02d}.h5"
        with h5py.File(path, "r") as f:
            data = f["data"][:]
        return torch.from_numpy(data).float().to(device)

    @staticmethod
    def stream_experts(
        root_dir: Path,
        layer: int,
        device: str = "cpu",
    ) -> Iterator[tuple[int, torch.Tensor]]:
        """Iterate over experts in a layer, yielding (expert_id, activations).

        Args:
            root_dir: Store root directory
            layer: Layer index
            device: Target device

        Yields:
            (expert_id, Tensor [n_docs, d_model])
        """
        path = root_dir / "expert_activations" / f"layer_{layer:02d}.h5"
        with h5py.File(path, "r") as f:
            n_experts = f["data"].shape[1]
            for expert_id in range(n_experts):
                data = f["data"][:, expert_id, :]
                yield expert_id, torch.from_numpy(data).float().to(device)

    @staticmethod
    def load_routing_counts(
        root_dir: Path,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Load routing counts [n_layers, n_docs, n_experts].

        Args:
            root_dir: Store root directory
            device: Target device

        Returns:
            Tensor [n_layers, n_docs, n_experts]
        """
        path = root_dir / "routing" / "routing_counts.h5"
        with h5py.File(path, "r") as f:
            data = f["data"][:]
        return torch.from_numpy(data).long().to(device)

    @staticmethod
    def load_metadata(root_dir: Path) -> dict:
        """Load composition metadata.

        Args:
            root_dir: Store root directory

        Returns:
            Metadata dictionary
        """
        with open(root_dir / "composition.json") as f:
            return json.load(f)

    @staticmethod
    def load_doc_ids(root_dir: Path) -> list[int]:
        """Load document ID mapping.

        Args:
            root_dir: Store root directory

        Returns:
            List of original dataset doc IDs
        """
        with open(root_dir / "doc_ids.json") as f:
            return json.load(f)


# --- Unembedding matrix (HDF5) ---


def save_unembedding(root_dir: Path, unembed: torch.Tensor) -> Path:
    """Save the unembedding matrix (lm_head.weight) to HDF5.

    Args:
        root_dir: Store root directory
        unembed: [vocab_size, d_model] unembedding matrix

    Returns:
        Path to the saved HDF5 file
    """
    root_dir = Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    path = root_dir / "unembedding.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "data",
            data=unembed.cpu().numpy(),
            dtype=np.float32,
        )
    return path


def load_unembedding(
    root_dir: Path,
    device: str = "cpu",
) -> torch.Tensor:
    """Load the unembedding matrix from HDF5.

    Args:
        root_dir: Store root directory
        device: Target device

    Returns:
        Tensor [vocab_size, d_model]
    """
    path = Path(root_dir) / "unembedding.h5"
    with h5py.File(path, "r") as f:
        data = f["data"][:]
    return torch.from_numpy(data).float().to(device)
