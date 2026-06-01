"""Simple storage for Expert Pursuit activations."""

import json
from pathlib import Path
from typing import Any, cast

import h5py
import torch

_OPTIONAL_EXPERT_FIELDS = ("routing_weights", "positions")


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


def append_to_file(
    f: h5py.File,
    expert_id: int,
    activations: torch.Tensor,
    tokens: torch.Tensor,
    routing_weights: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    max_rows: int | None = None,
) -> None:
    group_name = _expert_group_name(expert_id)
    acts = activations.detach().cpu()
    toks = tokens.detach().cpu()
    weights = routing_weights.detach().cpu() if routing_weights is not None else None
    pos = positions.detach().cpu() if positions is not None else None
    if acts.numel() == 0:
        return
    group = f.require_group(group_name)
    if max_rows is not None:
        # Cap rows per expert: keep at most max_rows, truncating the incoming batch to
        # whatever space is left (keeps disk bounded for all-token captures).
        existing = group["activations"].shape[0] if "activations" in group else 0
        room = max_rows - existing
        if room <= 0:
            return
        if acts.shape[0] > room:
            acts, toks = acts[:room], toks[:room]
            weights = weights[:room] if weights is not None else None
            pos = pos[:room] if pos is not None else None
    if "activations" not in group:
        group.create_dataset(
            "activations",
            data=acts,
            maxshape=(None, acts.shape[1]),
            chunks=(max(acts.shape[0], 1), acts.shape[1]),
            dtype=acts.numpy().dtype,
        )
        group.create_dataset(
            "tokens",
            data=toks,
            maxshape=(None,),
            chunks=(max(toks.shape[0], 1),),
            dtype=toks.numpy().dtype,
        )
        if weights is not None:
            group.create_dataset(
                "routing_weights",
                data=weights,
                maxshape=(None,),
                chunks=(max(weights.shape[0], 1),),
                dtype=weights.numpy().dtype,
            )
        if pos is not None:
            group.create_dataset(
                "positions",
                data=pos,
                maxshape=(None,),
                chunks=(max(pos.shape[0], 1),),
                dtype=pos.numpy().dtype,
            )
        return
    act_ds = cast(h5py.Dataset, group["activations"])
    tok_ds = cast(h5py.Dataset, group["tokens"])
    old_size = act_ds.shape[0]
    new_size = act_ds.shape[0] + acts.shape[0]
    act_ds.resize((new_size, act_ds.shape[1]))
    act_ds[-acts.shape[0] :] = acts
    tok_ds.resize((new_size,))
    tok_ds[-toks.shape[0] :] = toks
    if weights is not None:
        if "routing_weights" not in group:
            group.create_dataset(
                "routing_weights",
                data=torch.full((old_size,), float("nan")).numpy(),
                maxshape=(None,),
                chunks=(max(old_size, 1),),
            )
        weight_ds = cast(h5py.Dataset, group["routing_weights"])
        weight_ds.resize((new_size,))
        weight_ds[-weights.shape[0] :] = weights
    if pos is not None:
        if "positions" not in group:
            group.create_dataset(
                "positions",
                data=torch.full((old_size,), -1, dtype=pos.dtype).numpy(),
                maxshape=(None,),
                chunks=(max(old_size, 1),),
                dtype=pos.numpy().dtype,
            )
        pos_ds = cast(h5py.Dataset, group["positions"])
        pos_ds.resize((new_size,))
        pos_ds[-pos.shape[0] :] = pos


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
    """Return {expert_id: {activations, tokens, [routing_weights], [positions]}}.

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
