"""Expert activation capture for Expert Pursuit."""

from pathlib import Path

import h5py
import torch
import torch.nn.functional as F
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from src.cache import (
    _append_to_file,
    get_model_unembedding,
    save_metadata,
    save_unembedding,
)
from src.environment import get_unembedding_dir
from src.model_adapter import get_model_adapter


def capture_expert_activations(
    model,
    prompts: list[list[int]],
    output_dir: Path,
    model_name: str | None = None,
    batch_size: int = 8,
) -> dict:
    """Capture expert activations for all prompts using nnsight tracing.

    Processes prompts in batches with right-padding so each prompt's tokens
    stay at their true positions, preserving RoPE positional encodings.

    Args:
        model: NNsight LanguageModel
        prompts: List of tokenized prompts (list of token IDs)
        output_dir: Directory to save extractions
        model_name: Model name to store in metadata. If None, extracted from model.config._name_or_path.
        batch_size: Number of prompts per batch.

    Returns:
        Metadata dict with model_name, n_docs, n_layers, n_experts, d_model
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if model_name is None:
        model_name = model.config._name_or_path

    adapter = get_model_adapter(model=model)
    n_layers = adapter.n_layers
    n_experts = adapter.n_experts
    d_model = adapter.d_model

    # Sort by length so similar-length prompts land in the same batches
    sorted_prompts = sorted(prompts, key=len, reverse=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Capturing prompts", total=len(prompts))

        layer_files = {
            i: h5py.File(output_dir / f"layer_{i:02d}.h5", "a") for i in range(n_layers)
        }

        # Right-pad so token positions are preserved (RoPE stays correct)
        model.tokenizer.padding_side = "right"

        # NOTE: Sorted prompts favours batching of prompt of similar size together
        for batch_start in range(0, len(sorted_prompts), batch_size):
            batch = sorted_prompts[batch_start : batch_start + batch_size]
            prompt_lengths = [len(p) for p in batch]
            b_size = len(batch)

            with torch.no_grad(), model.trace(batch) as tracer:
                input_ids = model.inputs[1]["input_ids"].save().detach().cpu()
                norm_layer = model.model.norm

                for layer_idx, layer in enumerate(model.model.layers):
                    _, weights, indices = adapter.get_router_output(layer)
                    top_k_weights = weights.save().detach().cpu()

                    token_indices_list: list = []
                    down_projs_list: list = []
                    top_k_pos_list: list = []

                    expert_hit = adapter.get_expert_hit(layer)
                    active_experts = (
                        expert_hit[expert_hit != adapter.n_experts]
                        .squeeze(-1)
                        .save()
                        .detach()
                        .cpu()
                    )
                    num_iters = active_experts.numel()

                    with tracer.iter[:num_iters]:
                        top_k_pos, token_idx = adapter.get_top_k_pos_token_idx(layer)
                        down_proj = adapter.get_expert_output(layer)

                        # NOTE: Similarly to logit lens we apply the last normalization to the expert activations here
                        token_indices_list.append(token_idx.save().detach().cpu())
                        down_projs_list.append(
                            norm_layer(down_proj).save().detach().cpu()
                        )
                        top_k_pos_list.append(top_k_pos.save().detach().cpu())

                    layer_data = {
                        "active_experts": active_experts,
                        "token_indices": token_indices_list,
                        "down_projs": down_projs_list,
                        "top_k_pos": top_k_pos_list,
                        "weights": top_k_weights,
                    }

                    max_len = input_ids.shape[1]

                    # NOTE: Here we take the last token but averaging over content
                    # tokens can also be performed instead

                    # Pre-compute last-token positions for all batches (vectorized)
                    batch_offsets = torch.arange(b_size, device="cpu") * max_len
                    actual_lens_tensor = torch.tensor(
                        prompt_lengths, device="cpu", dtype=torch.long
                    )
                    last_positions = batch_offsets + actual_lens_tensor - 1

                    for i, expert_id in enumerate(active_experts.tolist()):
                        token_idx = layer_data["token_indices"][i]
                        down_proj = layer_data["down_projs"][i]
                        top_k_pos = layer_data["top_k_pos"][i]

                        # Vectorized: single mask instead of inner loop over batch
                        is_last = torch.isin(token_idx, last_positions)
                        if not is_last.any():
                            continue

                        # Extract all last-token data at once
                        last_down_proj = down_proj[is_last]
                        last_top_k_pos = top_k_pos[is_last]
                        last_token_idx_flat = token_idx[is_last]

                        # Compute gate weights and weighted output
                        gate_weights = top_k_weights[
                            last_token_idx_flat, last_top_k_pos
                        ]
                        gated_output = gate_weights.unsqueeze(-1) * last_down_proj

                        if gated_output.shape[0] == 0:
                            continue

                        # Map flat indices back to get token IDs
                        batch_indices = last_token_idx_flat // max_len
                        pos_in_batch = last_token_idx_flat % max_len
                        last_token_ids = input_ids[batch_indices, pos_in_batch]

                        # Single write per expert (was b_size writes)
                        _append_to_file(
                            layer_files[layer_idx],
                            expert_id,
                            gated_output.half(),
                            last_token_ids,
                        )

            progress.advance(task, len(batch))

        model.tokenizer.padding_side = "left"

        for f in layer_files.values():
            f.close()

    metadata = {
        "model_name": model_name,
        "n_docs": len(prompts),
        "n_layers": n_layers,
        "n_experts": n_experts,
        "d_model": d_model,
    }
    save_metadata(output_dir, **metadata)

    unembedding_dir = get_unembedding_dir(model_name)
    dictionary = F.normalize(get_model_unembedding(model), dim=1)
    save_unembedding(unembedding_dir / "dictionary.h5", dictionary)
    print(f"Saved unembedding to {unembedding_dir}")
    print(f"Saved activations to {output_dir}")

    return metadata
