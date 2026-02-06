"""Token-level MoE activation trace per document."""

from dataclasses import dataclass

import torch


@dataclass
class DocumentTrace:
    """MoE routing trace for a single document.

    Storage layout: [n_layers, seq_len, k] - variable length per document,
    no padding required.

    Attributes:
        expert_indices: Expert IDs selected for each token [layers, seq, k]
        expert_weights: Gate weights for each expert [layers, seq, k]
        doc_id: Original document index from dataset
    """

    expert_indices: torch.Tensor  # [layers, seq, k] - expert IDs
    expert_weights: torch.Tensor  # [layers, seq, k] - gate weights
    doc_id: int  # original dataset index

    @property
    def n_layers(self) -> int:
        return self.expert_indices.shape[0]

    @property
    def seq_len(self) -> int:
        return self.expert_indices.shape[1]

    @property
    def k(self) -> int:
        return self.expert_indices.shape[2]

    def get_token(self, layer: int, pos: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get expert IDs and weights for a specific token.

        Args:
            layer: Layer index
            pos: Token position in sequence

        Returns:
            Tuple of (expert_ids [k], expert_weights [k])
        """
        return (
            self.expert_indices[layer, pos],
            self.expert_weights[layer, pos],
        )
