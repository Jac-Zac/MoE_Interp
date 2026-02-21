"""Expert Pursuit activation capture.

Batched last-token capture: traces a batch of documents through the model
via nnsight, extracts per-expert gated outputs at the LAST TOKEN position,
and stores per-layer HDF5 files.

With left-padding (nnsight default), the last token is always at seq_len - 1,
making last-token extraction trivial.
"""

from pathlib import Path

import nnsight
import torch
from nnsight import LanguageModel
from tqdm import tqdm

from src.cache import save_layer, save_metadata


def capture_batch(
    model: LanguageModel,
    batch: list[list[int]],
) -> torch.Tensor:
    """Capture last-token gated expert outputs for a batch.

    Uses nnsight tracing to extract per-expert down_proj outputs and gate
    weights. For each expert in each document, extracts the gated output
    at the LAST TOKEN position (seq_len - 1 with left-padding).

    Args:
        model: nnsight LanguageModel instance.
        batch: List of token ID lists.

    Returns:
        [batch_size, n_layers, n_experts, d_model] dense tensor (float32).
    """
    n_layers = model.config.num_hidden_layers
    n_experts_total = model.config.num_experts
    d_model = model.config.hidden_size
    batch_size = len(batch)

    layer_active_experts: list = []
    layer_token_indices: list[list[torch.Tensor]] = []
    layer_down_projs: list[list[torch.Tensor]] = []
    layer_top_k_pos: list[list[torch.Tensor]] = []
    layer_weights: list = []
    layer_indices: list = []

    with torch.no_grad(), model.trace(batch) as tracer:
        for layer in model.model.layers:
            _, weights, indices = layer.mlp.source.self_gate_0.output
            layer_weights.append(weights)
            layer_indices.append(indices)

            expert_hit = layer.mlp.experts.source.nonzero_0.output
            active_experts = expert_hit[expert_hit != n_experts_total].squeeze(-1)
            num_iters = active_experts.numel()

            token_idx_list: list[torch.Tensor] = []
            down_proj_list: list[torch.Tensor] = []
            top_k_pos_list: list[torch.Tensor] = []

            with tracer.iter[:num_iters]:
                top_k_pos, token_idx = layer.mlp.experts.source.torch_where_0.output
                down_proj = layer.mlp.experts.source.nn_functional_linear_1.output
                token_idx_list.append(token_idx)
                down_proj_list.append(down_proj)
                top_k_pos_list.append(top_k_pos)

            layer_active_experts.append(active_experts)
            layer_token_indices.append(token_idx_list)
            layer_down_projs.append(down_proj_list)
            layer_top_k_pos.append(top_k_pos_list)

        nnsight.save(layer_active_experts)
        nnsight.save(layer_token_indices)
        nnsight.save(layer_down_projs)
        nnsight.save(layer_top_k_pos)
        weights_t = torch.stack(layer_weights, dim=0)
        indices_t = torch.stack(layer_indices, dim=0)
        nnsight.save(weights_t)
        nnsight.save(indices_t)

    seq_len = weights_t.shape[2]
    result = torch.zeros(batch_size, n_layers, n_experts_total, d_model)

    for li in range(n_layers):
        active_experts = layer_active_experts[li]
        for i, expert_id_tensor in enumerate(active_experts):
            expert_id = int(expert_id_tensor.item())
            token_idxs = layer_token_indices[li][i]
            down_proj = layer_down_projs[li][i]
            top_k_positions = layer_top_k_pos[li][i]

            if token_idxs.numel() == 0:
                continue

            doc_idxs = token_idxs // seq_len
            padded_positions = token_idxs % seq_len

            last_token_mask = padded_positions == (seq_len - 1)
            if not last_token_mask.any():
                continue

            token_idxs_last = token_idxs[last_token_mask]
            down_proj_last = down_proj[last_token_mask]
            top_k_positions_last = top_k_positions[last_token_mask]
            doc_idxs_last = doc_idxs[last_token_mask]

            for d_tensor in doc_idxs_last.unique():
                d = int(d_tensor.item())
                doc_mask = doc_idxs_last == d_tensor
                doc_down_proj = down_proj_last[doc_mask]
                doc_top_k = top_k_positions_last[doc_mask]

                for idx in range(doc_down_proj.shape[0]):
                    gate_w = weights_t[li, d, seq_len - 1, doc_top_k[idx]]
                    gated = gate_w.unsqueeze(-1) * doc_down_proj[idx]
                    result[d, li, expert_id] = gated
                    break

    return result


def encode_dataset(
    model: LanguageModel,
    prompts: list[list[int]],
    output_dir: Path,
    batch_size: int = 8,
) -> Path:
    """Encode dataset: capture last-token expert outputs and save per-layer.

    Args:
        model: nnsight LanguageModel.
        prompts: List of token ID lists.
        output_dir: Root directory for output (per-layer HDF5 + metadata).
        batch_size: Documents per batch for nnsight tracing.

    Returns:
        Path to output directory.
    """
    n_layers = model.config.num_hidden_layers
    n_experts = model.config.num_experts
    d_model = model.config.hidden_size
    output_dir = Path(output_dir)

    all_batches: list[torch.Tensor] = []

    for start in tqdm(range(0, len(prompts), batch_size), desc="Encoding"):
        batch = prompts[start : start + batch_size]
        batch_result = capture_batch(model, batch)
        all_batches.append(batch_result)

    activations = torch.cat(all_batches, dim=0)

    for li in tqdm(range(n_layers), desc="Saving layers"):
        save_layer(output_dir, li, activations[:, li])

    save_metadata(output_dir, len(prompts), n_layers, n_experts, d_model)

    return output_dir
