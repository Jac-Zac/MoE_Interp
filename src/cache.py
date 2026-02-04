"""
Dataclass definitions for MoE activation tracing.

Hierarchical structure:
    TraceCache (per run)
     └── LayerCache (per layer)
          ├── routing (topk_weights, topk_indices)
          └── experts (Dict[expert_id, ExpertCache])
               └── ExpertCache (sparse by prompt)
"""

# HACK: Cache has been vibecoded (by opencode cloud opus)

from dataclasses import dataclass, field
from typing import Dict, List

import torch


@dataclass
class ExpertCache:
    """
    Per-expert activation storage (sparse by prompt).

    Only stores data for prompts where this expert was activated.
    """

    # prompt_idx -> which tokens hit this expert
    token_indices: Dict[int, torch.Tensor] = field(default_factory=dict)
    # prompt_idx -> [n_tokens, d_model] input to expert
    inputs: Dict[int, torch.Tensor] = field(default_factory=dict)
    # prompt_idx -> [n_tokens, d_model] post-down-proj output
    outputs: Dict[int, torch.Tensor] = field(default_factory=dict)


@dataclass
class LayerCache:
    """
    Per-layer storage for routing + expert internals.

    Routing tensors are stored as lists (one per prompt) to handle
    variable sequence lengths without padding.
    """

    # [seq_len, k] per prompt
    topk_weights: List[torch.Tensor] = field(default_factory=list)
    topk_indices: List[torch.Tensor] = field(default_factory=list)

    # expert_id -> ExpertCache (sparse, only populated for experts that fired)
    experts: Dict[int, ExpertCache] = field(default_factory=dict)


@dataclass
class TraceCache:
    """
    Top-level trace container for a run.

    Stores prompts and per-layer caches.
    """

    prompts: List[str]
    layers: Dict[int, LayerCache] = field(default_factory=dict)
