"""Simple storage for Expert Pursuit activations."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import h5py
import torch


def _metadata_path(path: Path) -> Path:
    path = Path(path)
    return path if path.suffix == ".json" else path / "metadata.json"


def _expert_group_name(expert_id: int) -> str:
    return f"expert_{expert_id:03d}"


def save_metadata(path: Path, **kwargs) -> None:
    path = _metadata_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(kwargs))


def load_metadata(path: Path) -> dict:
    return json.loads(_metadata_path(path).read_text())


def append_expert_h5(
    path: Path,
    expert_id: int,
    activations: torch.Tensor,
    tokens: torch.Tensor,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    group_name = _expert_group_name(expert_id)
    acts = activations.detach().cpu()
    toks = tokens.detach().cpu()
    if acts.numel() == 0:
        return
    with h5py.File(path, "a") as f:
        group = f.require_group(group_name)
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
            return
        act_ds = cast(h5py.Dataset, group["activations"])
        tok_ds = cast(h5py.Dataset, group["tokens"])
        new_size = act_ds.shape[0] + acts.shape[0]
        act_ds.resize((new_size, act_ds.shape[1]))
        act_ds[-acts.shape[0] :] = acts
        tok_ds.resize((new_size,))
        tok_ds[-toks.shape[0] :] = toks


def load_expert_h5(path: Path, expert_id: int) -> dict[str, torch.Tensor]:
    path = Path(path)
    group_name = _expert_group_name(expert_id)
    with h5py.File(path, "r") as f:
        if group_name not in f:
            return {"activations": torch.empty(0), "tokens": torch.empty(0)}
        group = cast(h5py.Group, f[group_name])
        return {
            "activations": torch.from_numpy(
                cast(h5py.Dataset, group["activations"])[:]
            ),
            "tokens": torch.from_numpy(cast(h5py.Dataset, group["tokens"])[:]),
        }


def save_unembedding(path: Path, tensor: torch.Tensor) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("weight", data=tensor.detach().cpu().numpy())


def load_unembedding(path: Path) -> torch.Tensor:
    with h5py.File(path, "r") as f:
        return torch.from_numpy(cast(h5py.Dataset, f["weight"])[:])


def iter_layer_activations(
    encodings_dir: Path,
    n_layers: int,
    n_experts: int,
    min_activations: int = 0,
) -> Iterator[tuple[int, int, torch.Tensor]]:
    """Yield (layer_idx, expert_idx, activations) for every populated expert.

    Streams one expert at a time — peak memory is one expert's activations.
    Skips experts with fewer than min_activations rows.
    """
    encodings_dir = Path(encodings_dir)
    for li in range(n_layers):
        layer_path = encodings_dir / f"layer_{li:02d}.h5"
        if not layer_path.exists():
            continue
        for ei in range(n_experts):
            data = load_expert_h5(layer_path, ei)
            acts = data["activations"]
            if acts.shape[0] >= min_activations:
                yield li, ei, acts


def iter_layers(
    encodings_dir: Path,
    n_layers: int,
    n_experts: int,
    min_activations: int = 0,
) -> Iterator[tuple[int, dict[int, torch.Tensor]]]:
    """Yield (layer_idx, {expert_id: activations}) for each layer.

    All experts for a layer are loaded together (one HDF5 file open at a time),
    so peak memory is one full layer's activations. Experts below min_activations
    are excluded from the dict.
    """
    encodings_dir = Path(encodings_dir)
    for li in range(n_layers):
        layer_path = encodings_dir / f"layer_{li:02d}.h5"
        if not layer_path.exists():
            continue
        expert_acts: dict[int, torch.Tensor] = {}
        for ei in range(n_experts):
            data = load_expert_h5(layer_path, ei)
            acts = data["activations"]
            if acts.shape[0] >= min_activations:
                expert_acts[ei] = acts
        if expert_acts:
            yield li, expert_acts
