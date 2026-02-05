"""
Sparse-first MoE activation trace.

Routing tensors use token-major layout: [batch, seq, layer, k]
Expert outputs use event-based sparse storage.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch


@dataclass
class MoETrace:
    """MoE routing trace for expert activation analysis."""

    token_ids: torch.Tensor  # [batch, seq]
    expert_indices: torch.Tensor  # [batch, seq, layer, k] expert IDs
    expert_weights: torch.Tensor  # [batch, seq, layer, k] gate weights
    expert_token_idx: Dict[Tuple[int, int], torch.Tensor] = field(default_factory=dict)
    expert_outputs: Dict[Tuple[int, int], torch.Tensor] = field(default_factory=dict)

    @classmethod
    def from_tensors(
        cls,
        token_ids: torch.Tensor,
        indices_list: List[torch.Tensor],
        weights_list: List[torch.Tensor],
        num_experts: int,
    ) -> "MoETrace":
        """
        Build MoETrace from nnsight-captured tensors.

        Args:
            token_ids: [batch, seq] token IDs
            indices_list: [n_layers] list of [batch, seq, k] expert indices
            weights_list: [n_layers] list of [batch, seq, k] expert weights
            num_experts: Total number of experts per layer

        Returns:
            MoETrace with sparse expert-to-token mapping
        """

        batch_size, seq_len = token_ids.shape
        n_layers = len(indices_list)

        # Stack to [batch, seq, layer, k]
        indices_tensor = torch.stack(indices_list, dim=2)
        weights_tensor = torch.stack(weights_list, dim=2)

        # Build sparse (layer, expert) -> token_indices mapping
        expert_token_idx = {}
        for layer in range(n_layers):
            for expert in range(num_experts):
                mask = indices_tensor[:, :, layer, :] == expert
                batch_idx, seq_idx, _ = mask.nonzero(as_tuple=True)
                flat_idx = batch_idx * seq_len + seq_idx
                if len(flat_idx) > 0:
                    expert_token_idx[(layer, expert)] = flat_idx.unique()

        return cls(
            token_ids=token_ids,
            expert_indices=indices_tensor,
            expert_weights=weights_tensor,
            expert_token_idx=expert_token_idx,
            expert_outputs={},
        )
