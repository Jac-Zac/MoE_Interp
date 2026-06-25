"""Expert activation capture for Expert Pursuit."""

from pathlib import Path
from typing import Any

import h5py
import torch
import torch.nn.functional as F
from datasets import Dataset
from tqdm import tqdm

from moe_interp.capture.cache import (
    append_to_file,
    get_model_unembedding,
    save_metadata,
    save_unembedding,
)
from moe_interp.capture.model_adapter import MoEAdapter, get_model_adapter
from moe_interp.config import get_unembedding_dir


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
) -> int:
    """Trace one batch, reconstruct every expert's last-token contribution, flush.

    Returns the batch size.

    Pass 1 (one trace): tap each MoE block's boundary inputs (hidden_states,
    top_k_index, top_k_weights) plus the final-norm input. nnsight 0.7's ``tracer.iter``
    does not step a module's internal per-expert loop and fused kernels (grouped_mm)
    never materialise per-expert tensors, so we do NOT trace inside the experts.
    Pass 2 (outside the trace): ``adapter.reconstruct_expert_contributions`` re-derives
    each expert's per-token output from the block inputs and weight params, using the
    model-specific expert math. We keep only each prompt's last real token (one row per
    prompt per routed expert), which is plenty for SOMP over a large document set."""
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
        for layer in model.model.layers:
            expert_inputs.append(adapter.tap_layer(layer).save())
        pre_norm_hidden = model.model.norm.input.save()

    # Keep only the last real token of each prompt, over the flattened (b_size * max_len)
    # axis (right-padding: prompt row r's last token sits at r*max_len + length_r - 1).
    lengths = torch.as_tensor(prompt_lengths, dtype=torch.long)
    flat = torch.arange(b_size * max_len)
    keep_mask = flat == ((flat // max_len) * max_len + lengths[flat // max_len] - 1)
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
            norm_weight=norm_weight,
            norm_eps=norm_eps,
        )
        for e, (acts, tokens, weights) in contributions.items():
            append_to_file(
                layer_files[layer_idx], e, acts, tokens, routing_weights=weights
            )
    return b_size


def capture_expert_activations(
    model,
    prompts: Dataset | list[list[int]],
    output_dir: Path,
    model_name: str | None = None,
    dataset_name: str | None = None,
    batch_size: int = 8,
) -> dict:
    """Capture each prompt's last-token expert contributions using nnsight tracing.

    Processes prompts in batches with right-padding so each prompt's tokens stay
    at their true positions, preserving RoPE positional encodings. Stores one row per
    prompt per routed expert (the prompt's last real token); over a large document set
    this gives each expert plenty of rows for SOMP.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if model_name is None:
        model_name = model.config._name_or_path

    adapter = get_model_adapter(model)
    ds = prepare_prompts_dataset(prompts)

    layer_files: dict[int, h5py.File] = {
        i: h5py.File(output_dir / f"layer_{i:02d}.h5", "a")
        for i in range(adapter.n_layers)
    }

    # Right-pad so token positions are preserved (RoPE stays correct)
    original_padding_side = model.tokenizer.padding_side
    model.tokenizer.padding_side = "right"
    try:
        with tqdm(total=len(ds), desc="Capturing prompts") as progress:
            for batch in ds.iter(batch_size=batch_size):
                b_size = _capture_batch(
                    model=model,
                    adapter=adapter,
                    layer_files=layer_files,
                    batch=batch,
                    norm_weight=model.model.norm.weight,
                    norm_eps=model.model.norm.variance_epsilon,
                )
                progress.update(b_size)
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
        "token_selection": "last",
        "stores_routing_weights": True,
    }
    save_capture_artifacts(model, model_name, output_dir, metadata)
    return metadata
