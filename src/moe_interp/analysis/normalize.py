"""Feature normalization variants for expert-level clustering.

Different normalizations answer different questions, and the one that clusters best is
not necessarily the one that decodes best. We expose each as an explicit, comparable
condition rather than baking one in as a hidden default. All functions take a
``(n_experts, n_features)`` tensor and return a tensor of the same shape.
"""

import torch

_EPS = 1e-8


def normalize_raw(X: torch.Tensor) -> torch.Tensor:
    """Identity: use features as-is."""
    return X.float()


def normalize_layer_centered(X: torch.Tensor) -> torch.Tensor:
    """Subtract the per-feature mean across experts in the layer."""
    X = X.float()
    return X - X.mean(dim=0, keepdim=True)


def normalize_l2(X: torch.Tensor) -> torch.Tensor:
    """L2-normalize each expert row (puts features on the unit sphere → cosine geometry)."""
    X = X.float()
    return X / X.norm(dim=1, keepdim=True).clamp_min(_EPS)


def normalize_standardized(X: torch.Tensor) -> torch.Tensor:
    """Z-score each feature column across experts."""
    X = X.float()
    mean = X.mean(dim=0, keepdim=True)
    std = X.std(dim=0, keepdim=True).clamp_min(_EPS)
    return (X - mean) / std


NORMALIZATIONS = {
    "raw": normalize_raw,
    "layer_centered": normalize_layer_centered,
    "l2": normalize_l2,
    "standardized": normalize_standardized,
}


def normalize_features(X: torch.Tensor, method: str) -> torch.Tensor:
    """Apply a named normalization. Composable names are joined with '+', e.g.
    ``"layer_centered+l2"`` centers then L2-normalizes (the clustering default)."""
    out = X
    for step in method.split("+"):
        if step not in NORMALIZATIONS:
            options = ", ".join(sorted(NORMALIZATIONS))
            raise ValueError(f"Unknown normalization '{step}'. Available: {options}")
        out = NORMALIZATIONS[step](out)
    return out
