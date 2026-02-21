"""Expert Pursuit activation capture.

Batched last-token capture: traces a batch of documents through the model
via nnsight, extracts per-expert gated outputs at the LAST TOKEN position,
and stores per-layer HDF5 files.

With left-padding (nnsight default), the last token is always at seq_len - 1,
making last-token extraction trivial.

NOTE: Token aggregation strategies
==================================
1. LAST-TOKEN (default): Captures output at generation-ready position.
   Semantic: "What does this expert contribute to predicting the answer?"

2. CONTENT-TOKEN AVERAGING (HeadPursuit style, not yet implemented):
   Average over question tokens between <BOS> unwitting|>\\n and \\n<|assistant|">.
   Semantic: "Which experts are consistently used for this question type?"
   MoE consideration: each token routes to only top-k experts (k=8 for OLMoE).
   Use zero contribution where expert was not selected:

   # Pseudocode:
   for tok in content_tokens:
       if expert_id in top_k_indices[tok]:
           contribution = gate_weight[tok] * down_proj[tok]
       else:
           contribution = zero_vector
   expert_output = mean(contributions)
"""

from pathlib import Path

import numpy as np
import torch
from nnsight import LanguageModel
from tqdm import tqdm

from src.cache import save_layer, save_metadata


def capture_batch(
    model: LanguageModel,
    batch: list[list[int]],
) -> torch.Tensor:
    """Capture last-token expert outputs for a single batch.

    Returns [batch_size, n_layers, n_experts, d_model] tensor.
    Expert not selected at last token -> zero vector.
    """
    n_layers = model.config.num_hidden_layers
    n_experts = model.config.num_experts
    d_model = model.config.hidden_size
    batch_size = len(batch)
    seq_len = len(batch[0])

    batch_data = {}

    with torch.no_grad(), model.trace(batch) as tracer:
        for layer_idx, layer in enumerate(model.model.layers):
            # Get routing info from the gate
            # top_k_weights: weight for each expert
            # top_k_indices: expert id active for each token
            # self_gate_0 outputs: (_, top_k_weights, top_k_indices)
            _, weights, indices = layer.mlp.source.self_gate_0.output
            top_k_weights = weights.save()
            top_k_indices = indices.save()

            token_indices_list: list[torch.Tensor] = []
            down_projs_list: list[torch.Tensor] = []
            top_k_pos_list: list[torch.Tensor] = []

            # NOTE: One must be very careful of what to get
            # I need to get expert_hit after the nonzero_0
            # expert_mask_sum_0  ->  9 expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            # torch_greater_0    ->  + ...
            # nonzero_0          ->  + ...
            expert_hit = layer.mlp.experts.source.nonzero_0.output
            active_experts = (
                expert_hit[expert_hit != model.config.num_experts].squeeze(-1).save()
            )
            num_iters = active_experts.numel()

            with tracer.iter[:num_iters]:
                top_k_pos, token_idx = layer.mlp.experts.source.torch_where_0.output
                down_proj = layer.mlp.experts.source.nn_functional_linear_1.output
                token_indices_list.append(token_idx.save())
                down_projs_list.append(down_proj.save())
                top_k_pos_list.append(top_k_pos.save())

            batch_data[layer_idx] = {
                "active_experts": active_experts,
                "token_indices": token_indices_list,
                "down_projs": down_projs_list,
                "top_k_pos": top_k_pos_list,
                "weights": top_k_weights,
            }

    # Aggregate: [batch_size, n_layers, n_experts, d_model]
    device = batch_data[0]["weights"].device
    result = torch.zeros(
        batch_size, n_layers, n_experts, d_model, dtype=torch.float32, device=device
    )

    for layer_idx in range(n_layers):
        d = batch_data[layer_idx]
        active_experts = d["active_experts"]
        token_indices = d["token_indices"]
        down_projs = d["down_projs"]
        top_k_positions = d["top_k_pos"]
        weights = d["weights"]

        for i, expert_id in enumerate(active_experts.tolist()):
            token_idx = token_indices[i]
            down_proj = down_projs[i]
            top_k_pos = top_k_positions[i]

            # Filter to last token only
            last_token_mask = (token_idx % seq_len) == (seq_len - 1)
            if not last_token_mask.any():
                continue

            last_token_idx = token_idx[last_token_mask]
            last_down_proj = down_proj[last_token_mask]
            last_top_k_pos = top_k_pos[last_token_mask]

            # weights is [total_tokens, top_k], indexed by [token_position, top_k_position]
            doc_idx = last_token_idx // seq_len
            gate_weights = weights[last_token_idx, last_top_k_pos]
            gated_output = gate_weights.unsqueeze(-1) * last_down_proj

            for j in range(gated_output.shape[0]):
                result[doc_idx[j], layer_idx, expert_id] += gated_output[j].float()

    return result.cpu()


def encode_dataset(
    model: LanguageModel,
    prompts: list[list[int]],
    output_dir: Path,
    batch_size: int = 8,
    dtype: np.typing.DTypeLike = np.float16,
) -> Path:
    """Encode dataset: capture last-token expert outputs and save per-layer."""
    n_layers = model.config.num_hidden_layers
    n_experts = model.config.num_experts
    d_model = model.config.hidden_size
    output_dir = Path(output_dir)
    dtype_str = np.dtype(dtype).name

    all_batches: list[torch.Tensor] = []

    for start in tqdm(range(0, len(prompts), batch_size), desc="Encoding"):
        batch = prompts[start : start + batch_size]
        batch_result = capture_batch(model, batch)
        all_batches.append(batch_result)

    activations = torch.cat(all_batches, dim=0)

    for li in tqdm(range(n_layers), desc="Saving layers"):
        save_layer(output_dir, li, activations[:, li], dtype=dtype)

    save_metadata(
        output_dir, len(prompts), n_layers, n_experts, d_model, dtype=dtype_str
    )

    return output_dir
