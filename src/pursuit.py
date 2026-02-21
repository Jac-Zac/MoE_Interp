"""Expert Pursuit: projection-based concept decomposition of MoE experts.

Projects expert activations onto full unembedding dictionary and finds
top-k tokens by explained variance ratio (EVR).
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.cache import load_layer, load_metadata


@dataclass
class ExpertConceptResult:
    """Decomposition result for a single expert."""

    layer: int
    expert_id: int
    n_activations: int
    tokens: list[str]
    token_ids: list[int]
    evr: list[float]


@dataclass
class PursuitResult:
    """Full Expert Pursuit analysis results."""

    n_layers: int
    n_experts: int
    k: int
    experts: list[ExpertConceptResult] = field(default_factory=list)
    evr_matrix: torch.Tensor = field(default_factory=lambda: torch.zeros(0))

    def concept_frequency(self, top_n: int = 5) -> Counter:
        """Aggregate top-N concepts across all experts."""
        counter: Counter = Counter()
        for e in self.experts:
            counter.update(e.tokens[:top_n])
        return counter

    def save(self, path: Path) -> None:
        """Save results to JSON + tensors."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        torch.save(self.evr_matrix, path / "evr.pt")

        data = {
            "n_layers": self.n_layers,
            "n_experts": self.n_experts,
            "k": self.k,
            "experts": [
                {
                    "layer": e.layer,
                    "expert_id": e.expert_id,
                    "n_activations": e.n_activations,
                    "tokens": e.tokens,
                    "token_ids": e.token_ids,
                    "evr": e.evr,
                }
                for e in self.experts
            ],
        }
        with open(path / "pursuit_results.json", "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load(path: Path) -> "PursuitResult":
        """Load saved results from disk."""
        path = Path(path)
        with open(path / "pursuit_results.json") as f:
            data = json.load(f)

        experts = [
            ExpertConceptResult(
                layer=e["layer"],
                expert_id=e["expert_id"],
                n_activations=e["n_activations"],
                tokens=e["tokens"],
                token_ids=e["token_ids"],
                evr=e["evr"],
            )
            for e in data["experts"]
        ]

        return PursuitResult(
            n_layers=data["n_layers"],
            n_experts=data["n_experts"],
            k=data["k"],
            experts=experts,
            evr_matrix=torch.load(path / "evr.pt", weights_only=True),
        )


def projection_pursuit(
    X: torch.Tensor,
    dictionary: torch.Tensor,
    tokenizer: Any,
    k: int = 50,
) -> tuple[list[str], list[int], list[float]]:
    """Project expert activations onto dictionary, return top-k by EVR.

    EVR per token = var(projection) / total_var, clamped to [0, 1].
    """
    if X.shape[0] <= 1:
        return [], [], []

    X_centered = X - X.mean(dim=0, keepdim=True)
    projections = X_centered @ dictionary.T

    total_var = X_centered.var(dim=0).sum()
    if total_var < 1e-10:
        return [], [], []

    var_per_token = projections.var(dim=0)
    evr = (var_per_token / total_var).clamp(0, 1)

    valid_mask = evr > 1e-6
    if not valid_mask.any():
        return [], [], []

    top_k = evr.topk(min(k, int(valid_mask.sum().item())))

    token_ids = top_k.indices.tolist()
    tokens = [tokenizer.decode([i]).strip() for i in token_ids]
    return tokens, token_ids, top_k.values.tolist()


def run_expert_pursuit(
    activations_dir: Path,
    unembed: torch.Tensor,
    tokenizer: Any,
    k: int = 50,
    min_activations: int = 5,
) -> PursuitResult:
    """Run projection pursuit on all experts across all layers.

    Args:
        activations_dir: Path to directory with per-layer HDF5 files.
        unembed: [vocab_size, d_model] unembedding matrix.
        tokenizer: HuggingFace tokenizer for decoding.
        k: Number of top atoms to return per expert.
        min_activations: Minimum non-zero activations required.

    Returns:
        PursuitResult with per-expert decompositions and EVR.
    """
    meta = load_metadata(activations_dir)
    n_layers = meta["n_layers"]
    n_experts = meta["n_experts"]
    k = min(k, unembed.shape[0])

    # L2-normalize dictionary
    dictionary = F.normalize(unembed, dim=1)

    evr_matrix = torch.zeros(n_layers, n_experts, k)
    expert_results: list[ExpertConceptResult] = []

    for li in tqdm(range(n_layers), desc="Expert Pursuit"):
        layer_data = load_layer(activations_dir, li)  # [n_docs, n_experts, d_model]

        for ei in range(n_experts):
            X = layer_data[:, ei, :]  # [n_docs, d_model]

            n_activations = int((X.norm(dim=1) > 1e-6).sum().item())
            if n_activations < min_activations:
                continue

            tokens, token_ids, evr = projection_pursuit(X, dictionary, tokenizer, k)
            if not tokens:
                continue

            evr_matrix[li, ei, : len(evr)] = torch.tensor(evr)
            expert_results.append(
                ExpertConceptResult(
                    layer=li,
                    expert_id=ei,
                    n_activations=n_activations,
                    tokens=tokens,
                    token_ids=token_ids,
                    evr=evr,
                )
            )

    return PursuitResult(
        n_layers=n_layers,
        n_experts=n_experts,
        k=k,
        experts=expert_results,
        evr_matrix=evr_matrix,
    )
