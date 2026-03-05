"""Expert activation capture for Expert Pursuit."""

from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.cache import append_expert_h5, save_metadata, save_unembedding


def capture_expert_activations(
    model,
    prompts: list[list[int]],
    batch_size: int,
    output_dir: Path,
    data_dir: Path | None = None,
    model_name: str | None = None,
) -> dict:
    """Capture expert activations for all prompts using nnsight tracing.

    Args:
        model: NNsight LanguageModel
        prompts: List of tokenized prompts (list of token IDs)
        batch_size: Batch size for processing
        output_dir: Directory to save encodings
        data_dir: Parent data directory (for saving unembedding). If None, derived from output_dir.
        model_name: Model name to store in metadata. If None, extracted from model.config._name_or_path.

    Returns:
        Metadata dict with model_name, n_docs, n_layers, n_experts, d_model
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if data_dir is None:
        data_dir = output_dir.parent

    if model_name is None:
        model_name = model.config._name_or_path

    n_layers = model.config.num_hidden_layers
    n_experts = model.config.num_experts
    d_model = model.config.hidden_size

    total_docs = 0
    for start in tqdm(range(0, len(prompts), batch_size), desc="Capturing batches"):
        batch = prompts[start : start + batch_size]
        total_docs += len(batch)
        batch_data = {}

        with torch.no_grad(), model.trace(batch) as tracer:
            input_ids = model.inputs[1]["input_ids"].save()

            for layer_idx, layer in enumerate(model.model.layers):
                _, weights, indices = layer.mlp.source.self_gate_0.output
                top_k_weights = weights.save()
                top_k_indices = indices.save()

                token_indices_list: list[torch.Tensor] = []
                down_projs_list: list[torch.Tensor] = []
                top_k_pos_list: list[torch.Tensor] = []

                expert_hit = layer.mlp.experts.source.nonzero_0.output
                active_experts = (
                    expert_hit[expert_hit != model.config.num_experts]
                    .squeeze(-1)
                    .save()
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

        seq_len = input_ids.shape[1]

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

                last_token_mask = (token_idx % seq_len) == (seq_len - 1)
                if not last_token_mask.any():
                    continue

                last_token_idx = token_idx[last_token_mask]
                last_down_proj = down_proj[last_token_mask]
                last_top_k_pos = top_k_positions[i][last_token_mask]

                gate_weights = weights[last_token_idx, last_top_k_pos]
                gated_output = gate_weights.unsqueeze(-1) * last_down_proj

                last_doc_indices = last_token_idx // seq_len
                last_token_ids = input_ids[last_doc_indices, -1]

                if gated_output.shape[0] == 0:
                    continue
                layer_path = output_dir / f"layer_{layer_idx:02d}.h5"
                append_expert_h5(
                    layer_path,
                    expert_id,
                    gated_output.half(),
                    last_token_ids,
                )

    metadata = {
        "model_name": model_name,
        "n_docs": total_docs,
        "n_layers": n_layers,
        "n_experts": n_experts,
        "d_model": d_model,
    }
    save_metadata(output_dir, **metadata)

    unembedding_dir = data_dir / "unembedding"
    dictionary = F.normalize(model.lm_head.weight.detach().float(), dim=1).cpu()
    save_unembedding(unembedding_dir / "dictionary.h5", dictionary)
    print(f"Saved unembedding to {unembedding_dir}")
    print(f"Saved activations to {output_dir}")

    return metadata
