"""Expert Pursuit data structures."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import load_file, save_file


@dataclass
class ExpertTrace:
    """Per-expert activation data for a specific layer.

    Simple data container with no save/load methods.

    Attributes:
        token_indices: [n_tokens] positions in sequence routed to this expert
        raw_outputs: [n_tokens, hidden_dim] down-projection outputs (before weighting)
        top_k_positions: [n_tokens] which of the k slots this expert was in (0 to k-1)
    """

    token_indices: torch.Tensor
    raw_outputs: torch.Tensor
    top_k_positions: torch.Tensor


@dataclass
class DocumentTrace:
    """All expert activations for one document across all layers.

    Storage layout:
      - doc_id: original document index from dataset
      - n_layers: number of model layers
      - expert_indices: [n_layers, seq_len, k] expert IDs per token
      - expert_weights: [n_layers, seq_len, k] gate weights per token
      - expert_traces: list[dict] where expert_traces[layer_idx][expert_id] = ExpertTrace

    Attributes:
        doc_id: Original document index from dataset
        n_layers: Number of model layers
        expert_indices: Expert routing indices per token
        expert_weights: Gate weights per token
        expert_traces: Per-layer, per-expert activation data
    """

    doc_id: int
    n_layers: int
    expert_indices: torch.Tensor  # [n_layers, seq_len, k]
    expert_weights: torch.Tensor  # [n_layers, seq_len, k]
    expert_traces: list[
        dict[int, ExpertTrace]
    ]  # [n_layers] dict[expert_id -> ExpertTrace]

    @property
    def seq_len(self) -> int:
        return self.expert_indices.shape[1]

    @property
    def k(self) -> int:
        return self.expert_indices.shape[2]

    def __str__(self) -> str:
        total_experts = sum(len(layer_acts) for layer_acts in self.expert_traces)
        return (
            f"DocumentTrace(doc_id={self.doc_id}, "
            f"layers={self.n_layers}, seq_len={self.seq_len}, k={self.k}, "
            f"total_expert_traces={total_experts})"
        )

    def __repr__(self) -> str:
        return self.__str__()

    def save(self, output_dir: Path) -> Path:
        """Save this document trace to disk in safetensors format.

        All tensors are detached, moved to CPU, and cloned to ensure
        contiguous memory layout before saving.

        Args:
            output_dir: Directory to save the file

        Returns:
            Path to saved file
        """
        filename = f"doc_{self.doc_id:06d}.safetensors"
        filepath = output_dir / filename
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build tensor dict with detached, CPU, contiguous tensors
        tensors = {
            "expert_indices": self.expert_indices.detach().cpu().contiguous(),
            "expert_weights": self.expert_weights.detach().cpu().contiguous(),
            "doc_id": torch.tensor(self.doc_id, dtype=torch.int64),
            "n_layers": torch.tensor(self.n_layers, dtype=torch.int64),
        }

        # Flatten expert_traces: expert_traces[layer_idx][expert_id] = ExpertTrace
        # Store as: "act_{layer_idx}_{expert_id}_tokens", "act_{layer_idx}_{expert_id}_output", "act_{layer_idx}_{expert_id}_pos"
        acts_metadata = []  # List of (layer_idx, expert_id) pairs for reconstruction

        for layer_idx, layer_dict in enumerate(self.expert_traces):
            for expert_id, trace in layer_dict.items():
                base_key = f"act_{layer_idx}_{expert_id}"
                tensors[f"{base_key}_tokens"] = (
                    trace.token_indices.detach().cpu().contiguous()
                )
                tensors[f"{base_key}_output"] = (
                    trace.raw_outputs.detach().cpu().contiguous()
                )
                tensors[f"{base_key}_pos"] = (
                    trace.top_k_positions.detach().cpu().contiguous()
                )
                acts_metadata.append((layer_idx, expert_id))

        # Store metadata as tensor: [n_entries, 2] where each row is (layer_idx, expert_id)
        if acts_metadata:
            tensors["acts_metadata"] = torch.tensor(acts_metadata, dtype=torch.int64)
        else:
            tensors["acts_metadata"] = torch.empty((0, 2), dtype=torch.int64)

        save_file(tensors, filepath)
        return filepath

    @classmethod
    def load(cls, doc_id: int, data_dir: Optional[Path] = None) -> "DocumentTrace":
        """Load a DocumentTrace by document ID.

        Args:
            doc_id: Document ID to load
            data_dir: Directory containing trace files (default: ./data)

        Returns:
            DocumentTrace for the requested document

        Raises:
            FileNotFoundError: If document not found
        """
        if data_dir is None:
            data_dir = Path("./data")

        filepath = data_dir / f"doc_{doc_id:06d}.safetensors"

        if not filepath.exists():
            raise FileNotFoundError(f"Document {doc_id} not found at {filepath}")

        tensors = load_file(filepath)

        # Reconstruct expert_traces from flattened structure
        n_layers = int(tensors["n_layers"].item())

        # Initialize list of empty dicts for each layer
        expert_traces: list[dict[int, ExpertTrace]] = [dict() for _ in range(n_layers)]

        # Reconstruct from metadata
        acts_metadata = tensors["acts_metadata"]
        for layer_idx, expert_id in acts_metadata.tolist():
            base_key = f"act_{layer_idx}_{expert_id}"
            trace = ExpertTrace(
                token_indices=tensors[f"{base_key}_tokens"],
                raw_outputs=tensors[f"{base_key}_output"],
                top_k_positions=tensors[f"{base_key}_pos"],
            )
            expert_traces[layer_idx][expert_id] = trace

        return cls(
            doc_id=int(tensors["doc_id"].item()),
            n_layers=n_layers,
            expert_indices=tensors["expert_indices"],
            expert_weights=tensors["expert_weights"],
            expert_traces=expert_traces,
        )


def list_all(data_dir: Optional[Path] = None) -> list[int]:
    """List all available document IDs in a directory.

    Args:
        data_dir: Directory containing trace files (default: ./data)

    Returns:
        List of document IDs sorted numerically
    """
    if data_dir is None:
        data_dir = Path("./data")

    doc_ids = []
    for filepath in sorted(data_dir.glob("doc_*.safetensors")):
        try:
            doc_id = int(filepath.stem.split("_")[1])
            doc_ids.append(doc_id)
        except (ValueError, IndexError):
            continue

    return doc_ids
