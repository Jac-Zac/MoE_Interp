"""Expert activation capture for Expert Pursuit."""

from pathlib import Path

import torch
import torch.nn.functional as F
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from src.cache import append_expert_h5, save_metadata, save_unembedding
from src.environment import get_unembedding_dir
from src.model_adapter import get_model_adapter


def capture_expert_activations(
    model,
    prompts: list[list[int]],
    output_dir: Path,
    data_dir: Path | None = None,
    model_name: str | None = None,
) -> dict:
    """Capture expert activations for all prompts using nnsight tracing.

    Processes one prompt at a time. Batching is intentionally avoided: nnsight
    left-pads shorter sequences to match the longest in the batch, which shifts
    positional embeddings for all padded documents. Because OLMoE uses RoPE,
    positional encoding directly affects attention and expert routing, so a token
    that sits at position 5 in a standalone forward pass would appear at a
    different position inside a padded batch, producing different activations.
    Processing one prompt at a time guarantees no padding is ever introduced.

    Args:
        model: NNsight LanguageModel
        prompts: List of tokenized prompts (list of token IDs)
        output_dir: Directory to save extractions
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
        model_name: str = model.config._name_or_path

    adapter = get_model_adapter(model=model)
    n_layers = adapter.n_layers
    n_experts = adapter.n_experts
    d_model = adapter.d_model

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Capturing prompts", total=len(prompts))

        # HACK: No batching of prompt is done to avoid incorrect results due to the
        # effect of left padding on the positional embeddings
        for prompt in prompts:
            with torch.no_grad(), model.trace(prompt) as tracer:
                input_ids = model.inputs[1]["input_ids"].save()

                for layer_idx, layer in enumerate(model.model.layers):
                    _, weights, indices = adapter.get_router_output(layer)
                    top_k_weights = weights.save()

                    token_indices_list: list[torch.Tensor] = []
                    down_projs_list: list[torch.Tensor] = []
                    top_k_pos_list: list[torch.Tensor] = []

                    expert_hit = adapter.get_expert_hit(layer)
                    active_experts = (
                        expert_hit[expert_hit != adapter.n_experts].squeeze(-1).save()
                    )
                    num_iters = active_experts.numel()

                    with tracer.iter[:num_iters]:
                        top_k_pos, token_idx = adapter.get_top_k_pos_token_idx(layer)
                        down_proj = adapter.get_expert_output(layer)

                        # NOTE: Similarly to logit lens we apply the last normalization to the expert activations here
                        token_indices_list.append(token_idx.save())
                        down_projs_list.append(model.model.norm(down_proj).save())
                        top_k_pos_list.append(top_k_pos.save())

                    layer_data = {
                        "active_experts": active_experts,
                        "token_indices": token_indices_list,
                        "down_projs": down_projs_list,
                        "top_k_pos": top_k_pos_list,
                        "weights": top_k_weights,
                    }

                    # Single prompt: no padding, last real token is always at seq_len - 1
                    seq_len = input_ids.shape[1]
                    # FIX: Here I take the last token but average over tokens can also be performed instaed
                    last_token_id = input_ids[0, -1]

                    for i, expert_id in enumerate(active_experts.tolist()):
                        token_idx = layer_data["token_indices"][i]
                        down_proj = layer_data["down_projs"][i]
                        top_k_pos = layer_data["top_k_pos"][i]

                        last_token_mask = token_idx == (seq_len - 1)
                        if not last_token_mask.any():
                            continue

                        # Multi-GPU: ensure mask is on same device as target tensors
                        last_down_proj = down_proj[last_token_mask.to(down_proj.device)]
                        last_top_k_pos = top_k_pos[last_token_mask.to(top_k_pos.device)]

                        # Get gate weights and compute weighted output
                        # weights is [seq_len, top_k], indexed by [token_position, top_k_position]
                        gate_weights = top_k_weights[
                            seq_len - 1, last_top_k_pos.to(top_k_weights.device)
                        ]
                        gated_output = (
                            gate_weights.unsqueeze(-1).to(last_down_proj.device)
                            * last_down_proj
                        )

                        if gated_output.shape[0] == 0:
                            continue

                        layer_path = output_dir / f"layer_{layer_idx:02d}.h5"
                        append_expert_h5(
                            layer_path,
                            expert_id,
                            gated_output.half(),
                            last_token_id.unsqueeze(0).expand(gated_output.shape[0]),
                        )

            progress.advance(task)

    metadata = {
        "model_name": model_name,
        "n_docs": len(prompts),
        "n_layers": n_layers,
        "n_experts": n_experts,
        "d_model": d_model,
    }
    save_metadata(output_dir, **metadata)

    unembedding_dir = get_unembedding_dir(model_name)
    dictionary = F.normalize(model.lm_head.weight.detach().float(), dim=1).cpu()
    save_unembedding(unembedding_dir / "dictionary.h5", dictionary)
    print(f"Saved unembedding to {unembedding_dir}")
    print(f"Saved activations to {output_dir}")

    return metadata
