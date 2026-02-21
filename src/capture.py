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

import numpy as np

from src.cache import save_layer, save_metadata


def encode_dataset(
    model: LanguageModel,
    prompts: list[list[int]],
    output_dir: Path,
    batch_size: int = 8,
    dtype: np.typing.DTypeLike = np.float16,
) -> Path:
    """Encode dataset: capture last-token expert outputs and save per-layer.

    Args:
        model: nnsight LanguageModel.
        prompts: List of token ID lists.
        output_dir: Root directory for output (per-layer HDF5 + metadata).
        batch_size: Documents per batch for nnsight tracing.
        dtype: Storage dtype for activations (default: float16).

    Returns:
        Path to output directory.
    """
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
