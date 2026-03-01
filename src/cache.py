"""Simple storage for Expert Pursuit activations."""

import json
from pathlib import Path
from typing import cast

import h5py
import torch


def save_metadata(path: Path, **kwargs) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    (path if path.suffix == ".json" else path / "metadata.json").write_text(
        json.dumps(kwargs)
    )


def load_metadata(path: Path) -> dict:
    path = Path(path)
    p = path if path.suffix == ".json" else path / "metadata.json"
    return json.loads(p.read_text())


def save_expert(path: Path, activations: torch.Tensor, tokens: torch.Tensor) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("activations", data=activations.detach().cpu().numpy())
        f.create_dataset("tokens", data=tokens.detach().cpu().numpy())


def load_expert(path: Path) -> dict[str, torch.Tensor]:
    with h5py.File(path, "r") as f:
        return {
            "activations": torch.from_numpy(cast(h5py.Dataset, f["activations"])[:]),
            "tokens": torch.from_numpy(cast(h5py.Dataset, f["tokens"])[:]),
        }


def append_expert_h5(
    path: Path,
    expert_id: int,
    activations: torch.Tensor,
    tokens: torch.Tensor,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    group_name = f"expert_{expert_id:03d}"
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
    group_name = f"expert_{expert_id:03d}"
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
