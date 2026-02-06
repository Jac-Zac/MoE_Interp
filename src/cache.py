"""Token-level MoE activation trace.

Each token's expert activations are stored without padding.
Document boundaries are tracked to map tokens back to their source.
"""

from dataclasses import dataclass
from typing import List, Tuple

import torch

# HACK: This structure is currently vibecoded and needs revision


@dataclass
class MoETrace:
    """Token-level MoE routing trace for expert activation analysis.

    Storage layout: [n_layers, total_tokens, k] for contiguous layer access.
    Document boundaries allow mapping tokens back to source documents.
    """

    token_ids: torch.Tensor  # [total_tokens] - all tokens concatenated
    expert_indices: torch.Tensor  # [n_layers, total_tokens, k] - expert IDs
    expert_weights: torch.Tensor  # [n_layers, total_tokens, k] - gate weights
    doc_boundaries: torch.Tensor  # [n_docs+1] - cumulative token counts per doc
    doc_source_ids: torch.Tensor  # [n_docs] - original dataset indices for each doc

    @classmethod
    def build(
        cls,
        docs: List[List[int]],
        indices_stack: torch.Tensor,  # [n_layers, batch, seq, k]
        weights_stack: torch.Tensor,  # [n_layers, batch, seq, k]
        doc_boundaries: torch.Tensor,
        doc_source_ids: List[int],
    ) -> "MoETrace":
        """Build MoETrace from stacked routing tensors.

        Args:
            docs: List of document token ID lists
            indices_stack: Stacked expert indices from nnsight trace
            weights_stack: Stacked expert weights from nnsight trace
            doc_boundaries: Pre-computed cumulative token counts
            doc_source_ids: Original dataset indices for each document

        Returns:
            MoETrace with layer-first storage layout
        """
        # Concatenate all tokens
        token_ids = torch.tensor([tid for doc in docs for tid in doc], dtype=torch.long)

        # Flatten batch and seq: [n_layers, batch, seq, k] -> [n_layers, total_tokens, k]
        n_layers = indices_stack.shape[0]
        flat_shape = (n_layers, -1, indices_stack.shape[-1])

        return cls(
            token_ids=token_ids,
            expert_indices=indices_stack.reshape(flat_shape),
            expert_weights=weights_stack.reshape(flat_shape),
            doc_boundaries=doc_boundaries,
            doc_source_ids=torch.tensor(doc_source_ids, dtype=torch.long),
        )

    def doc_slice(self, doc_idx: int) -> slice:
        """Get slice for token indices of a specific document."""
        return slice(
            self.doc_boundaries[doc_idx].item(),
            self.doc_boundaries[doc_idx + 1].item(),
        )

    def experts_for_token(
        self, layer: int, token_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get expert IDs and weights for a specific layer and token.

        Args:
            layer: Layer index
            token_idx: Global token index (flattened across all docs)

        Returns:
            Tuple of (expert_ids [k], expert_weights [k])
        """
        return (
            self.expert_indices[layer, token_idx],
            self.expert_weights[layer, token_idx],
        )

    @property
    def n_docs(self) -> int:
        return len(self.doc_boundaries) - 1

    @property
    def n_layers(self) -> int:
        return self.expert_indices.shape[0]

    @property
    def k(self) -> int:
        return self.expert_indices.shape[2]
