"""Simple safetensor storage for Expert Pursuit activations."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


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
    save_file({"activations": activations, "tokens": tokens}, path)


def load_expert(path: Path) -> dict[str, torch.Tensor]:
    return load_file(path)


def save_unembedding(path: Path, tensor: torch.Tensor) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"weight": tensor}, path)


def load_unembedding(path: Path) -> torch.Tensor:
    return load_file(path)["weight"]
