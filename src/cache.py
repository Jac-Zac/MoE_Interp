"""
Dataclass for MoE expert activation tracing.

Usage:
    trace = MoETrace(
        prompts=["text1", "text2"],
        expert_indices=indices_tensor,  # [batch, seq_len, n_layers, top_k]
        expert_weights=weights_tensor,  # [batch, seq_len, n_layers, top_k]
        expert_outputs={(layer, expert_id): activations_tensor, ...}
    )
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


@dataclass
class MoETrace:
    prompts: List[str]
    expert_indices: torch.Tensor  # [batch, seq_len, n_layers, top_k]
    expert_weights: torch.Tensor  # [batch, seq_len, n_layers, top_k]

    # (layer, expert) -> [n_tokens, hidden]
    expert_outputs: Dict[Tuple[int, int], torch.Tensor]
