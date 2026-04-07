"""Simple storage for Expert Pursuit activations."""

import json
from pathlib import Path
from typing import Any, cast

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
    path.write_text(json.dumps(kwargs, indent=2))


def load_metadata(path: Path) -> dict:
    return json.loads(_metadata_path(path).read_text())


def _append_to_file(
    f: h5py.File,
    expert_id: int,
    activations: torch.Tensor,
    tokens: torch.Tensor,
) -> None:
    group_name = _expert_group_name(expert_id)
    acts = activations.detach().cpu()
    toks = tokens.detach().cpu()
    if acts.numel() == 0:
        return
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
) -> dict[int, torch.Tensor]:
    """Return {expert_id: activations} for one layer.

    Experts with fewer than min_activations rows are excluded.
    Returns an empty dict if the layer file does not exist.
    """
    layer_path = Path(extractions_dir) / f"layer_{layer_idx:02d}.h5"
    if not layer_path.exists():
        return {}
    result: dict[int, torch.Tensor] = {}
    with h5py.File(layer_path, "r") as f:
        for ei in range(n_experts):
            group_name = _expert_group_name(ei)
            if group_name not in f:
                continue
            group = cast(h5py.Group, f[group_name])
            acts_ds = cast(h5py.Dataset, group["activations"])
            if acts_ds.shape[0] < min_activations:
                continue
            result[ei] = torch.from_numpy(acts_ds[:])
    return result
