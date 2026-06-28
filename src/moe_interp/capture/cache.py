"""Simple storage for Expert Pursuit activations."""

import json
from pathlib import Path
from typing import Any, cast

import h5py
import torch

_OPTIONAL_EXPERT_FIELDS = ("routing_weights",)


def _metadata_path(path: Path) -> Path:
    path = Path(path)
    return path if path.suffix == ".json" else path / "metadata.json"


def _expert_group_name(expert_id: int) -> str:
    return f"expert_{expert_id:03d}"


def _read_expert_group(group: h5py.Group) -> dict[str, torch.Tensor]:
    entry = {
        "activations": torch.from_numpy(cast(h5py.Dataset, group["activations"])[:]),
        "tokens": torch.from_numpy(cast(h5py.Dataset, group["tokens"])[:]),
    }
    for name in _OPTIONAL_EXPERT_FIELDS:
        if name in group:
            entry[name] = torch.from_numpy(cast(h5py.Dataset, group[name])[:])
    return entry


def save_metadata(path: Path, **kwargs) -> None:
    path = _metadata_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(kwargs, indent=2))


def load_metadata(path: Path) -> dict:
    return json.loads(_metadata_path(path).read_text())


def _append_dataset(group: h5py.Group, name: str, data: torch.Tensor) -> None:
    """Create a resizable dataset on first write, else extend it along axis 0."""
    if name not in group:
        group.create_dataset(
            name,
            data=data,
            maxshape=(None, *data.shape[1:]),
            chunks=(max(data.shape[0], 1), *data.shape[1:]),
            dtype=data.numpy().dtype,
        )
        return
    ds = cast(h5py.Dataset, group[name])
    n = data.shape[0]
    ds.resize(ds.shape[0] + n, axis=0)
    ds[-n:] = data


def append_to_file(
    f: h5py.File,
    expert_id: int,
    activations: torch.Tensor,
    tokens: torch.Tensor,
    routing_weights: torch.Tensor | None = None,
) -> None:
    acts = activations.detach().cpu()
    toks = tokens.detach().cpu()
    weights = routing_weights.detach().cpu() if routing_weights is not None else None
    if acts.numel() == 0:
        return
    group = f.require_group(_expert_group_name(expert_id))
    # Legacy groups predate routing_weights: backfill the existing rows with NaN so the
    # column stays aligned with activations before appending this batch's weights.
    if (
        weights is not None
        and "activations" in group
        and "routing_weights" not in group
    ):
        old = cast(h5py.Dataset, group["activations"]).shape[0]
        group.create_dataset(
            "routing_weights",
            data=torch.full((old,), float("nan")).numpy(),
            maxshape=(None,),
            chunks=(max(old, 1),),
        )
    _append_dataset(group, "activations", acts)
    _append_dataset(group, "tokens", toks)
    if weights is not None:
        _append_dataset(group, "routing_weights", weights)


def save_unembedding(path: Path, tensor: torch.Tensor) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("weight", data=tensor.detach().cpu().numpy())


def load_unembedding(path: Path) -> torch.Tensor:
    with h5py.File(path, "r") as f:
        return torch.from_numpy(cast(h5py.Dataset, f["weight"])[:])


def get_model_unembedding(model: Any) -> torch.Tensor:
    """Extract lm_head weight from a model, handling meta-device tensors.

    When a model is loaded with device_map="auto", some parameters (e.g. lm_head)
    may remain on the meta device. This safely moves them to CPU before use.
    """
    weight = model.lm_head.weight
    if weight.device.type == "meta":
        weight = weight.to("cpu")
    return weight.detach().float()


def load_layer_h5(
    extractions_dir: Path,
    layer_idx: int,
    n_experts: int,
    min_activations: int = 0,
) -> dict[int, dict[str, torch.Tensor]]:
    """Return {expert_id: {activations, tokens, [routing_weights]}}.

    Experts with fewer than min_activations rows are excluded.
    Returns an empty dict if the layer file does not exist.
    """
    layer_path = Path(extractions_dir) / f"layer_{layer_idx:02d}.h5"
    if not layer_path.exists():
        return {}
    result: dict[int, dict[str, torch.Tensor]] = {}
    with h5py.File(layer_path, "r") as f:
        for ei in range(n_experts):
            group_name = _expert_group_name(ei)
            if group_name not in f:
                continue
            group = cast(h5py.Group, f[group_name])
            acts_ds = cast(h5py.Dataset, group["activations"])
            if acts_ds.shape[0] < min_activations:
                continue
            result[ei] = _read_expert_group(group)
    return result


def load_layer_activations(
    extractions_dir: Path,
    layer_idx: int,
    n_experts: int,
    min_activations: int = 0,
) -> dict[int, torch.Tensor]:
    """Return only activations for callers that do not need token metadata."""
    layer = load_layer_h5(extractions_dir, layer_idx, n_experts, min_activations)
    return {ei: entry["activations"] for ei, entry in layer.items()}
