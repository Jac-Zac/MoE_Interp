"""Expert activation capture for Expert Pursuit."""

from pathlib import Path

import h5py
import torch
import torch.nn.functional as F
from datasets import Dataset
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
from src.environment import get_unembedding_dir, is_rank0
from src.model_adapter import get_model_adapter


def apply_component_rmsnorm(
    hidden_states: torch.Tensor,
    second_moment: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Approximate component RMSNorm using the residual stream second moment.

    This keeps the expert output on the same scale as the final model norm while
    avoiding recomputing the full residual-stream normalization.
    """
    input_dtype = hidden_states.dtype
    # NOTE: Keep float32 here for stability (HF issue #33133).
    hidden_states = hidden_states.to(torch.float32)
    hidden_states = hidden_states * torch.rsqrt(second_moment.unsqueeze(-1) + eps)
    return weight * hidden_states.to(input_dtype)


def capture_expert_activations(
    model,
    prompts: Dataset | list[list[int]],
    output_dir: Path,
    model_name: str | None = None,
    dataset_name: str | None = None,
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
    if is_rank0():
        output_dir.mkdir(parents=True, exist_ok=True)

    if model_name is None:
        model_name = model.config._name_or_path

    adapter = get_model_adapter(model=model)
    n_layers = adapter.n_layers
    n_experts = adapter.n_experts
    d_model = adapter.d_model

    # Normalise to HF Dataset so both paths use .iter() for batched iteration.
    # The Dataset stays memory-mapped (Arrow) — no full Python list is materialised.
    if not isinstance(prompts, Dataset):
        prompts = Dataset.from_dict({"input_ids": prompts})

    # Pre-compute prompt lengths and sort so similar-length prompts land together
    # (minimises right-padding waste per batch, preserving RoPE positions).
    ds = prompts.map(lambda x: {"length": len(x["input_ids"])})  # type: ignore[index]
    ds = ds.sort("length", reverse=True)
    n_docs = len(ds)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Capturing prompts", total=n_docs)

        layer_files = {}
        if is_rank0():
            layer_files = {
                i: h5py.File(output_dir / f"layer_{i:02d}.h5", "a")
                for i in range(n_layers)
            }

        # Right-pad so token positions are preserved (RoPE stays correct)
        original_padding_side = model.tokenizer.padding_side
        model.tokenizer.padding_side = "right"
        try:
            # .iter() yields dicts where batch["input_ids"] is list[list[int]] — exactly the format nnsight's model.trace() expects.
            for batch in ds.iter(batch_size=batch_size):
                batch_tokens = batch["input_ids"]  # type: ignore[index]
                prompt_lengths = batch["length"]  # type: ignore[index]
                b_size = len(batch_tokens)

                with torch.no_grad(), model.trace(batch_tokens) as tracer:
                    input_ids = model.inputs[1]["input_ids"].save().detach()
                    pre_norm_hidden = model.model.norm.input[0].save().detach()
                    norm_weight = model.model.norm.weight
                    norm_eps = model.model.norm.variance_epsilon

                    max_len = input_ids.shape[1]

                    # NOTE: Here we take the last token but averaging over content
                    # tokens can also be performed instead

                    # Pre-compute last-token positions for all batches (vectorized)
                    batch_offsets = torch.arange(b_size) * max_len
                    actual_lens_tensor = torch.tensor(prompt_lengths, dtype=torch.long)
                    last_positions = batch_offsets + actual_lens_tensor - 1
                    sample_indices = torch.arange(b_size)
                    pre_norm_last = pre_norm_hidden[
                        sample_indices, actual_lens_tensor - 1
                    ]
                    second_moment_last = torch.atleast_1d(
                        pre_norm_last.float().pow(2).mean(dim=-1)
                    )

                    for layer_idx, layer in enumerate(model.model.layers):
                        _, weights, indices = adapter.get_router_output(layer)
                        top_k_weights = weights.save().detach()

                        token_indices_list: list = []
                        down_projs_list: list = []
                        top_k_pos_list: list = []

                        expert_hit = adapter.get_expert_hit(layer)
                        active_experts = (
                            expert_hit[expert_hit != adapter.n_experts]
                            .squeeze(-1)
                            .save()
                            .detach()
                        )
                        num_iters = active_experts.numel()

                        with tracer.iter[:num_iters]:
                            top_k_pos, token_idx = adapter.get_top_k_pos_token_idx(
                                layer
                            )
                            down_proj = adapter.get_expert_output(layer)

                            token_indices_list.append(token_idx.save().detach())
                            down_projs_list.append(down_proj.save().detach())
                            top_k_pos_list.append(top_k_pos.save().detach())

                        layer_data = {
                            "active_experts": active_experts,
                            "token_indices": token_indices_list,
                            "down_projs": down_projs_list,
                            "top_k_pos": top_k_pos_list,
                            "weights": top_k_weights,
                        }

                        for i, expert_id in enumerate(active_experts.tolist()):
                            token_idx = layer_data["token_indices"][i]
                            down_proj = layer_data["down_projs"][i]
                            top_k_pos = layer_data["top_k_pos"][i]

                            target_device = down_proj.device
                            lp = last_positions.to(target_device)
                            ids = input_ids.to(target_device)
                            tw = top_k_weights.to(target_device)
                            token_idx = token_idx.to(target_device)
                            top_k_pos = top_k_pos.to(target_device)

                            # Vectorized: single mask instead of inner loop over batch
                            is_last = torch.isin(token_idx, lp)
                            if not is_last.any():
                                continue

                            # Extract all last-token data at once
                            last_down_proj = down_proj[is_last]
                            last_top_k_pos = top_k_pos[is_last]
                            last_token_idx_flat = token_idx[is_last]

                            # Compute gate weights and weighted output
                            gate_weights = tw[last_token_idx_flat, last_top_k_pos]
                            gated_output = gate_weights.unsqueeze(-1) * last_down_proj
                            batch_indices = (last_token_idx_flat // max_len).long()
                            gated_output = apply_component_rmsnorm(
                                hidden_states=gated_output,
                                second_moment=second_moment_last.to(target_device)[
                                    batch_indices
                                ],
                                weight=norm_weight.to(target_device),
                                eps=norm_eps,
                            )

                            if gated_output.shape[0] == 0:
                                continue

                            # Map flat indices back to get token IDs
                            batch_indices = last_token_idx_flat // max_len
                            pos_in_batch = last_token_idx_flat % max_len
                            last_token_ids = ids[batch_indices, pos_in_batch]

                            # Single write per expert (was b_size writes)
                            if is_rank0():
                                _append_to_file(
                                    layer_files[layer_idx],
                                    expert_id,
                                    gated_output.half().cpu(),
                                    last_token_ids.cpu(),
                                )

                progress.advance(task, b_size)
        finally:
            model.tokenizer.padding_side = original_padding_side
            if is_rank0():
                for f in layer_files.values():
                    f.close()

    metadata = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "n_docs": n_docs,
        "n_layers": n_layers,
        "n_experts": n_experts,
        "d_model": d_model,
    }
    if is_rank0() and model_name is not None:
        save_metadata(output_dir, **metadata)

        unembedding_dir = get_unembedding_dir(model_name)
        dictionary = F.normalize(get_model_unembedding(model), dim=1)
        save_unembedding(unembedding_dir / "dictionary.h5", dictionary)
        print(f"Saved unembedding to {unembedding_dir}")
        print(f"Saved activations to {output_dir}")

    return metadata
