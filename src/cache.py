"""
Dataclass definitions for MoE activation tracing.

Hierarchical structure:
    TraceCache (per run)
     └── LayerCache (per layer)
          ├── routing (topk_weights, topk_indices)
          └── expert_outputs (Dict[expert_id, Dict[prompt_idx, outputs]])
"""

# HACK: Cache has been vibecoded (by opencode cloud opus)

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

import torch


@dataclass
class LayerCache:
    # prompt_idx -> [seq_len, d_model]
    layer_inputs: Dict[int, torch.Tensor] = field(default_factory=dict)

    # The routing (The Map)
    topk_weights: List[torch.Tensor] = field(default_factory=list)
    topk_indices: List[torch.Tensor] = field(default_factory=list)

    # expert_id -> prompt_idx -> [n_expert_tokens, d_model]
    expert_outputs: Dict[int, Dict[int, torch.Tensor]] = field(
        default_factory=lambda: defaultdict(dict)
    )


@dataclass
class TraceCache:
    """
    Top-level trace container for a run.

    Stores prompts and per-layer caches.
    """

    # NOTE: Field calls that function each time an instance of that class is created
    # Not a reference to the dict but a new dict every time -> good for mutable to use fields

    prompts: List[str]
    layers: Dict[int, LayerCache] = field(
        default_factory=lambda: defaultdict(LayerCache)
    )
