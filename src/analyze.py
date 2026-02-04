"""
Simple analysis functions for expert activations.
"""

from collections import Counter
from typing import Any, Dict


# TODO: This is just an idea
# HACK: Vibecoded for later
def get_expert_data(layer_cache, prompt_idx, expert_id):
    # 1. Get the map
    indices = layer_cache.topk_indices[prompt_idx]  # [seq_len, k]

    # 2. Find where this expert was active
    # logical_idx is [N_hits], pointing to which row in seq_len
    row_idx, k_idx = (indices == expert_id).nonzero(as_tuple=True)

    # 3. Gather the inputs
    global_input = layer_cache.layer_inputs[prompt_idx]
    expert_inputs = global_input[row_idx]

    return row_idx, expert_inputs
