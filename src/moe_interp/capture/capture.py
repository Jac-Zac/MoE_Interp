"""Expert activation capture for Expert Pursuit."""

from pathlib import Path
from typing import Any

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

from moe_interp.capture.cache import (
    append_to_file,
    get_model_unembedding,
    save_metadata,
    save_unembedding,
)
from moe_interp.capture.model_adapter import MoEAdapter, get_model_adapter
from moe_interp.config import get_unembedding_dir, is_rank0


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


def build_pending_writes(
    layer_data_list: list,
    last_token_ids: torch.Tensor,
    last_positions: torch.Tensor,
    second_moment_last: torch.Tensor,
    max_len: int,
    norm_weight: torch.Tensor,
    norm_eps: float,
) -> dict[tuple[int, int], list[tuple[torch.Tensor, torch.Tensor]]]:
    """Pass 2: filter to last-token positions, normalise, stage writes.

    Takes the per-layer traced data collected during Pass 1 and produces a dict
    keyed by ``(layer_idx, expert_id)`` containing ``(expert_output, last_token_ids)``
    tensor pairs on CPU (float16 activations). Pure tensor math — no nnsight
    dependency — so the notebook can share this with `capture_expert_activations`.

    The ``down_projs`` tensors are already routing-weight-scaled contributions to
    the residual stream, captured right before each expert's ``index_add_`` call:
      OLMoE:   (act_fn(gate) * up) @ down_proj * routing_weight
      GPT-oss: (gated_output @ down_proj + down_proj_bias) * routing_weight
    """
    pending_writes: dict[tuple[int, int], list[tuple[torch.Tensor, torch.Tensor]]] = {}

    for layer_idx, layer_data in enumerate(layer_data_list):
        active_experts = layer_data["active_experts"]
        if not layer_data["down_projs"]:
            continue

        # All experts in the same layer share a device (pipeline/tensor parallel).
        target_device = layer_data["down_projs"][0].device
        lp = last_positions.to(target_device)
        sm_last = second_moment_last.to(target_device)
        nw = norm_weight.to(target_device)

        n_iters = len(layer_data["token_indices"])
        if not (
            active_experts.numel()
            == n_iters
            == len(layer_data["down_projs"])
            == len(layer_data["top_k_pos"])
        ):
            raise RuntimeError(
                "Mismatched traced event counts in capture Pass 2: "
                f"active_experts={active_experts.numel()}, "
                f"token_indices={n_iters}, "
                f"down_projs={len(layer_data['down_projs'])}, "
                f"top_k_pos={len(layer_data['top_k_pos'])}"
            )

        for i in range(n_iters):
            token_idx = layer_data["token_indices"][i].to(target_device).flatten()
            expert_output = layer_data["down_projs"][i].to(target_device)
            if expert_output.ndim == 1:
                expert_output = expert_output.unsqueeze(0)
            expert_id = int(active_experts[i].item())

            # Filter to last-token positions only
            is_last = torch.isin(token_idx, lp)
            if not is_last.any():
                continue

            expert_output = expert_output[is_last]
            last_token_idx_flat = token_idx[is_last]

            # Map flat indices back to batch positions and token IDs
            batch_indices = last_token_idx_flat // max_len
            batch_last_token_ids = last_token_ids[batch_indices]

            # Apply component RMSNorm using residual stream stats
            expert_output = apply_component_rmsnorm(
                hidden_states=expert_output,
                second_moment=sm_last[batch_indices],
                weight=nw,
                eps=norm_eps,
            )

            key = (layer_idx, expert_id)
            pending_writes.setdefault(key, []).append(
                (expert_output.half().cpu(), batch_last_token_ids.cpu())
            )

    return pending_writes


def flush_pending_writes(
    pending_writes: dict[tuple[int, int], list[tuple[torch.Tensor, torch.Tensor]]],
    layer_files: dict[int, h5py.File],
) -> None:
    """Concatenate staged per-expert tensors and append them to the HDF5 files."""
    for (layer_idx, expert_id), writes in pending_writes.items():
        all_activations = torch.cat([act for act, _ in writes], dim=0)
        all_tokens = torch.cat([tok for _, tok in writes], dim=0)
        append_to_file(layer_files[layer_idx], expert_id, all_activations, all_tokens)


def prepare_prompts_dataset(prompts: Dataset | list[list[int]]) -> Dataset:
    """Normalise to a HF Dataset, add a ``length`` column, sort descending."""
    if not isinstance(prompts, Dataset):
        prompts = Dataset.from_dict({"input_ids": prompts})
    ds = prompts.map(lambda x: {"length": len(x["input_ids"])})  # type: ignore[index]
    return ds.sort("length", reverse=True)


def save_capture_artifacts(
    model: Any,
    model_name: str,
    output_dir: Path,
    metadata: dict,
) -> None:
    """Save metadata and the normalized unembedding dictionary."""
    if not is_rank0():
        return
    save_metadata(output_dir, **metadata)
    unembedding_dir = get_unembedding_dir(model_name)
    dictionary = F.normalize(get_model_unembedding(model), dim=1)
    save_unembedding(unembedding_dir / "dictionary.h5", dictionary)
    print(f"Saved unembedding to {unembedding_dir}")
    print(f"Saved activations to {output_dir}")


def _capture_batch(
    model: Any,
    adapter: MoEAdapter,
    batch: dict,
    layer_files: dict[int, h5py.File],
    norm_weight: torch.Tensor,
    norm_eps: float,
) -> int:
    """Trace one batch, normalise Pass-2 outputs, flush to HDF5. Returns batch size."""
    batch_tokens = batch["input_ids"]
    prompt_lengths = batch["length"]
    b_size = len(batch_tokens)
    max_len = max(len(tokens) for tokens in batch_tokens)
    last_token_ids = torch.tensor([int(tokens[-1]) for tokens in batch_tokens], dtype=torch.long)

    with torch.no_grad(), model.trace(batch_tokens) as tracer:
        # NOTE: Here we take the last token but averaging over content
        # tokens can also be performed instead

        # Pre-compute last-token positions for all batches (vectorized)
        batch_offsets = torch.arange(b_size) * max_len
        actual_lens_tensor = torch.tensor(prompt_lengths, dtype=torch.long)
        last_positions = batch_offsets + actual_lens_tensor - 1
        sample_indices = torch.arange(b_size)

        # Pass 1: collect per-layer expert data
        layer_data_list: list = []
        for layer in model.model.layers:
            _, weights, _ = adapter.get_router_output(layer)
            top_k_weights = weights.save().detach()

            expert_hit = adapter.get_expert_hit(layer)
            active_experts = (
                expert_hit[expert_hit != adapter.n_experts].reshape(-1).save().detach()
            )

            token_indices_list: list = []
            down_projs_list: list = []
            top_k_pos_list: list = []
            for _step in tracer.iter[: active_experts.numel()]:
                top_k_pos, token_idx = adapter.get_top_k_pos_token_idx(layer)
                down_proj = adapter.get_expert_output(layer)
                token_indices_list.append(token_idx.save().detach())
                down_projs_list.append(down_proj.save().detach())
                top_k_pos_list.append(top_k_pos.save().detach())

            layer_data_list.append(
                {
                    "active_experts": active_experts,
                    "token_indices": token_indices_list,
                    "down_projs": down_projs_list,
                    "top_k_pos": top_k_pos_list,
                    "weights": top_k_weights,
                }
            )

        # Access the final norm input only after all layer nodes have been
        # materialized; nnsight traces are order-sensitive.
        pre_norm_last = model.model.norm.input[sample_indices, actual_lens_tensor - 1].save().detach()
        second_moment_last = torch.atleast_1d(pre_norm_last.float().pow(2).mean(dim=-1))

        # Pass 2: apply normalisation and stage per-expert writes. Kept inside
        # the trace context to match the working semantics with nnsight
        # saved-tensor materialisation.
        if is_rank0():
            pending_writes = build_pending_writes(
                layer_data_list=layer_data_list,
                last_token_ids=last_token_ids,
                last_positions=last_positions,
                second_moment_last=second_moment_last,
                max_len=max_len,
                norm_weight=norm_weight,
                norm_eps=norm_eps,
            )

    if is_rank0():
        flush_pending_writes(pending_writes, layer_files)
    return b_size


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
    """
    output_dir = Path(output_dir)
    if is_rank0():
        output_dir.mkdir(parents=True, exist_ok=True)

    if model_name is None:
        model_name = model.config._name_or_path

    adapter = get_model_adapter(model=model)
    ds = prepare_prompts_dataset(prompts)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Capturing prompts", total=len(ds))

        layer_files: dict[int, h5py.File] = {}
        if is_rank0():
            layer_files = {
                i: h5py.File(output_dir / f"layer_{i:02d}.h5", "a")
                for i in range(adapter.n_layers)
            }

        # Right-pad so token positions are preserved (RoPE stays correct)
        original_padding_side = model.tokenizer.padding_side
        model.tokenizer.padding_side = "right"
        try:
            for batch in ds.iter(batch_size=batch_size):
                b_size = _capture_batch(
                    model=model,
                    adapter=adapter,
                    batch=batch,
                    layer_files=layer_files,
                    norm_weight=model.model.norm.weight,
                    norm_eps=model.model.norm.variance_epsilon,
                )
                progress.advance(task, b_size)
        finally:
            model.tokenizer.padding_side = original_padding_side
            for f in layer_files.values():
                f.close()

    metadata = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "n_docs": len(ds),
        "n_layers": adapter.n_layers,
        "n_experts": adapter.n_experts,
        "d_model": adapter.d_model,
    }
    save_capture_artifacts(model, model_name, output_dir, metadata)
    return metadata
