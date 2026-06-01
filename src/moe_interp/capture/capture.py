"""Expert activation capture for Expert Pursuit."""

from pathlib import Path
from typing import Any, Literal

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


def token_real_mask(prompt_lengths, max_len: int) -> torch.Tensor:
    """Boolean mask over the flattened (b_size * max_len) token axis, True for real
    (non-padding) tokens. Right-padding means token t is real iff (t % max_len) is below
    that row's prompt length."""
    lengths = torch.as_tensor(prompt_lengths, dtype=torch.long)
    flat = torch.arange(len(lengths) * max_len)
    return (flat % max_len) < lengths[flat // max_len]


def flush_pending_writes(
    pending_writes: dict[
        tuple[int, int],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ],
    layer_files: dict[int, h5py.File],
    max_rows_per_expert: int | None = None,
) -> None:
    """Append each (layer, expert)'s staged tensors to its HDF5 file.

    ``max_rows_per_expert`` caps the rows kept per expert (truncating once full) so
    all-token captures stay bounded on disk."""
    for (layer_idx, expert_id), (
        acts,
        tokens,
        weights,
        positions,
    ) in pending_writes.items():
        append_to_file(
            layer_files[layer_idx],
            expert_id,
            acts,
            tokens,
            routing_weights=weights,
            positions=positions,
            max_rows=max_rows_per_expert,
        )


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
    layer_files: dict[int, h5py.File],
    batch: dict,
    norm_weight: torch.Tensor,
    norm_eps: float,
    token_selection: Literal["last", "all"],
    max_rows_per_expert: int | None = None,
) -> int:
    """Trace one batch, reconstruct every expert's contribution, flush. Returns batch size.

    Pass 1 (one trace): tap each MoE block's boundary inputs (hidden_states,
    top_k_index, top_k_weights) plus the final-norm input. nnsight 0.7's ``tracer.iter``
    does not step a module's internal per-expert loop and fused kernels (grouped_mm)
    never materialise per-expert tensors, so we do NOT trace inside the experts.
    Pass 2 (outside the trace): ``adapter.reconstruct_expert_contributions`` re-derives
    each expert's per-token output from the block inputs and weight params, using the
    model-specific expert math.

    ``max_rows_per_expert`` caps the rows kept per expert (truncating once full) to keep
    all-token captures bounded on disk."""
    batch_tokens = batch["input_ids"]
    prompt_lengths = batch["length"]
    b_size = len(batch_tokens)
    max_len = max(len(tokens) for tokens in batch_tokens)
    padded_token_ids = torch.full((b_size, max_len), -1, dtype=torch.long)
    for i, tokens in enumerate(batch_tokens):
        padded_token_ids[i, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)

    # Pass 1 — tap boundary tensors. Use an explicit loop (not a comprehension): a
    # comprehension's scope doesn't bind saved proxies back after the trace on nnsight 0.7.
    expert_inputs: list = []
    with torch.no_grad(), model.trace(batch_tokens):
        input_ids = model.inputs[1]["input_ids"].save()
        for layer in model.model.layers:
            expert_inputs.append(adapter.tap_layer(layer).save())
        pre_norm_hidden = model.model.norm.input.save()

    if not is_rank0():
        return b_size

    # Real-token mask over the flattened (b_size * max_len) axis.
    lengths = torch.as_tensor(prompt_lengths, dtype=torch.long)
    if token_selection == "last":
        flat = torch.arange(b_size * max_len)
        keep_mask = flat == ((flat // max_len) * max_len + lengths[flat // max_len] - 1)
    else:
        keep_mask = token_real_mask(prompt_lengths, max_len)
    token_ids = padded_token_ids.reshape(-1)
    second_moment = pre_norm_hidden.float().pow(2).mean(-1).reshape(-1)

    # Pass 2 — reconstruct & stage per-(layer, expert) writes.
    for layer_idx, layer in enumerate(model.model.layers):
        hidden_states, top_k_index, top_k_weights = adapter.unpack_boundary(
            expert_inputs[layer_idx]
        )
        contributions = adapter.reconstruct_expert_contributions(
            layer.mlp.experts,
            hidden_states,
            top_k_index,
            top_k_weights,
            real_mask=keep_mask,
            second_moment=second_moment,
            token_ids=token_ids,
            max_len=max_len,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
        )
        pending = {(layer_idx, e): rows for e, rows in contributions.items()}
        flush_pending_writes(pending, layer_files, max_rows_per_expert)
    return b_size


def capture_expert_activations(
    model,
    prompts: Dataset | list[list[int]],
    output_dir: Path,
    model_name: str | None = None,
    dataset_name: str | None = None,
    batch_size: int = 8,
    token_selection: Literal["last", "all"] = "last",
    max_rows_per_expert: int | None = None,
) -> dict:
    """Capture expert activations for all prompts using nnsight tracing.

    Processes prompts in batches with right-padding so each prompt's tokens stay
    at their true positions, preserving RoPE positional encodings.

    ``max_rows_per_expert`` caps the rows kept per expert (recommended for
    ``token_selection="all"`` to bound disk); once an expert is full, extra rows are
    dropped.
    """
    if token_selection not in {"last", "all"}:
        raise ValueError("token_selection must be 'last' or 'all'")

    output_dir = Path(output_dir)
    if is_rank0():
        output_dir.mkdir(parents=True, exist_ok=True)

    if model_name is None:
        model_name = model.config._name_or_path

    adapter = get_model_adapter(model)
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
                    layer_files=layer_files,
                    batch=batch,
                    norm_weight=model.model.norm.weight,
                    norm_eps=model.model.norm.variance_epsilon,
                    token_selection=token_selection,
                    max_rows_per_expert=max_rows_per_expert,
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
        "token_selection": token_selection,
        "stores_routing_weights": True,
        "stores_positions": True,
        "max_rows_per_expert": max_rows_per_expert,
    }
    save_capture_artifacts(model, model_name, output_dir, metadata)
    return metadata
