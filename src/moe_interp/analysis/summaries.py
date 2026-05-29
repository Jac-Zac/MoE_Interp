"""Per-expert summary statistics for unsupervised analysis.

For each active ``(layer, expert)`` we compute a compact description of its captured
activations: the mean vector (the cleanest first baseline), norm statistics, and a
centered PCA spectrum that distinguishes monosemantic-looking experts (one dominant
direction) from polysemantic ones (variance spread across many directions).
"""

from dataclasses import dataclass
from pathlib import Path

import torch

from moe_interp.capture.cache import load_layer_activations

_EPS = 1e-12


@dataclass
class ExpertSummary:
    """Summary of one expert's captured activations within a single layer."""

    layer: int
    expert: int
    count: int
    mean: torch.Tensor  # (d_model,)
    mean_norm: float
    row_norm_mean: float
    row_norm_std: float
    pc1_evr: float  # S[0]^2 / sum(S^2): fraction of variance in the top direction
    effective_rank: float  # participation ratio (sum S^2)^2 / sum S^4, in [1, min(n,d)]
    singular_values: torch.Tensor  # centered spectrum, length min(n, d)
    top_pc_directions: torch.Tensor  # (top_pcs, d_model)

    def to_record(self) -> dict:
        """JSON-friendly scalar summary (drops the large mean / direction tensors)."""
        return {
            "layer": self.layer,
            "expert": self.expert,
            "count": self.count,
            "mean_norm": self.mean_norm,
            "row_norm_mean": self.row_norm_mean,
            "row_norm_std": self.row_norm_std,
            "pc1_evr": self.pc1_evr,
            "effective_rank": self.effective_rank,
        }


def _spectrum(
    centered: torch.Tensor, top_pcs: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Centered singular values and top right-singular vectors.

    Computed via eigendecomposition of the (n x n) Gram matrix rather than a direct
    SVD: with n <= ~242 << d = 2048 this is cheaper and, crucially, robust — LAPACK's
    gesdd SVD fails to converge on the near-duplicate float16 rows that show up in
    sparse captures. Falls back to a degenerate spectrum if even eigh fails.
    """
    n, d = centered.shape
    if n < 2:
        return torch.zeros(1), torch.zeros(min(top_pcs, 1), d)
    try:
        gram = (centered @ centered.T).double()
        evals, evecs = torch.linalg.eigh(gram)
        evals = evals.flip(0).clamp_min(0.0)  # descending, non-negative
        evecs = evecs.flip(1)
        S = evals.sqrt().float()
        r = min(top_pcs, n)
        Sr = S[:r].clamp_min(_EPS).unsqueeze(0)
        # Right singular vectors: V = X^T U / S, shape (r, d).
        top_directions = ((centered.T @ evecs[:, :r].float()) / Sr).T.contiguous()
        return S, top_directions
    except Exception:  # noqa: BLE001 - degrade to a degenerate spectrum
        return torch.zeros(min(n, d)), torch.zeros(min(top_pcs, n), d)


def compute_expert_summary(
    acts: torch.Tensor,
    layer: int,
    expert: int,
    top_pcs: int = 3,
) -> ExpertSummary:
    """Summarize a single expert's activation matrix ``acts`` of shape (n, d)."""
    X = torch.nan_to_num(acts.float())
    n, d = X.shape
    mu = X.mean(dim=0)
    row_norms = X.norm(dim=1)

    centered = X - mu
    S, top_directions = _spectrum(centered, top_pcs)

    sq = S**2
    total = sq.sum()
    if total <= _EPS:
        pc1_evr = 0.0
        effective_rank = 1.0
    else:
        pc1_evr = (sq[0] / total).item()
        # Participation ratio: smooth count of "effective" dimensions.
        effective_rank = (total**2 / (sq**2).sum().clamp_min(_EPS)).item()

    return ExpertSummary(
        layer=layer,
        expert=expert,
        count=int(n),
        mean=mu,
        mean_norm=mu.norm().item(),
        row_norm_mean=row_norms.mean().item(),
        row_norm_std=(row_norms.std().item() if n >= 2 else 0.0),
        pc1_evr=pc1_evr,
        effective_rank=effective_rank,
        singular_values=S,
        top_pc_directions=top_directions,
    )


def summarize_layer(
    extractions_dir: Path,
    layer_idx: int,
    n_experts: int,
    min_activations: int = 0,
    top_pcs: int = 3,
) -> dict[int, ExpertSummary]:
    """Summarize every active expert in one layer. Maps expert_id -> ExpertSummary."""
    expert_acts = load_layer_activations(
        extractions_dir, layer_idx, n_experts, min_activations
    )
    return {
        ei: compute_expert_summary(acts, layer_idx, ei, top_pcs=top_pcs)
        for ei, acts in expert_acts.items()
    }


def summarize_all(
    extractions_dir: Path,
    metadata: dict,
    min_activations: int = 0,
    top_pcs: int = 3,
) -> dict[int, dict[int, ExpertSummary]]:
    """Summarize all layers. Maps layer_idx -> {expert_id -> ExpertSummary}."""
    n_layers = metadata["n_layers"]
    n_experts = metadata["n_experts"]
    return {
        layer_idx: summarize_layer(
            extractions_dir, layer_idx, n_experts, min_activations, top_pcs
        )
        for layer_idx in range(n_layers)
    }
