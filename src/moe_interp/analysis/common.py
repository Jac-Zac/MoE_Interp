"""Shared loading helpers for the post-hoc analyses.

Everything here reads stored tensors only (HDF5 extractions + SOMP ``results.jsonl``);
no model is loaded. Inputs default to the standard local layout for a model/dataset;
pass explicit ``extractions_dir`` / ``pursuit_dir`` to read from somewhere else.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import torch

from moe_interp.capture.cache import (
    load_layer_activations,
    load_metadata,
    load_unembedding,
)
from moe_interp.config import get_extractions_dir, get_pursuit_dir, get_unembedding_dir


def load_somp_results(pursuit_dir: Path) -> dict[tuple[int, int], dict]:
    """Map ``(layer, expert) -> full SOMP result record`` from ``results.jsonl``."""
    out: dict[tuple[int, int], dict] = {}
    with open(Path(pursuit_dir) / "results.jsonl") as f:
        for line in f:
            r = json.loads(line)
            out[(r["layer"], r["expert"])] = r
    return out


def load_analysis_inputs(
    model_name: str,
    dataset: str,
    extractions_dir: Path | str | None = None,
    pursuit_dir: Path | str | None = None,
) -> tuple[Path, dict, torch.Tensor, Path]:
    """Return ``(extractions_dir, metadata, dictionary, pursuit_dir)``.

    ``extractions_dir`` / ``pursuit_dir`` default to the standard local layout for
    ``model_name`` / ``dataset``; pass explicit paths to read from elsewhere. ``dictionary``
    is the L2-normalized unembedding (rows = token directions), float32 on CPU.
    """
    extractions_dir = (
        Path(extractions_dir)
        if extractions_dir
        else get_extractions_dir(model_name, dataset)
    )
    pursuit_dir = (
        Path(pursuit_dir) if pursuit_dir else get_pursuit_dir(model_name, dataset)
    )
    metadata = load_metadata(extractions_dir / "metadata.json")
    dictionary = load_unembedding(
        get_unembedding_dir(model_name) / "dictionary.h5"
    ).float()
    return extractions_dir, metadata, dictionary, pursuit_dir


def iter_expert_activations(
    extractions_dir: Path,
    n_layers: int,
    n_experts: int,
    min_activations: int = 5,
    max_rows: int | None = None,
    seed: int = 1337,
) -> Iterator[tuple[int, int, torch.Tensor]]:
    """Yield ``(layer, expert, A)`` for every sufficiently-sampled expert.

    ``A`` is ``(n, d)`` float32. If ``max_rows`` is set and an expert has more rows,
    a fixed random subsample is drawn (keeps the per-expert EVR solve fast and bounded).
    """
    gen = torch.Generator().manual_seed(seed)
    for layer_idx in range(n_layers):
        acts = load_layer_activations(
            extractions_dir, layer_idx, n_experts, min_activations
        )
        for expert_idx, A in acts.items():
            A = A.float()
            if max_rows is not None and A.shape[0] > max_rows:
                idx = torch.randperm(A.shape[0], generator=gen)[:max_rows]
                A = A[idx]
            yield layer_idx, expert_idx, A
