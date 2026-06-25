"""Shared helpers for ``(n_layers, n_experts)`` score grids.

Patching and gate-AtP both produce a layer×expert grid and then need its top
cells. ``top_experts`` is the one place that turns a flat argmax into ``(layer, expert)``
coordinates so the ``i // n_experts`` / ``i % n_experts`` arithmetic isn't re-derived per
caller.
"""

import numpy as np


def top_experts(grid, k: int = 20, *, by: str = "abs") -> list[tuple[int, int, float]]:
    """Top-``k`` ``(layer, expert, value)`` cells of a grid, best first.

    ``by="abs"`` ranks by ``|value|`` (largest effect of either sign — promoters and
    suppressors); ``by="signed"`` ranks by value descending (largest positive). ``NaN``
    cells (unsampled experts) are dropped. Accepts a tensor or array.
    """
    g = grid.detach().cpu().numpy() if hasattr(grid, "detach") else np.asarray(grid)
    n_experts = g.shape[1]
    flat = g.ravel()
    keys = np.abs(flat) if by == "abs" else flat
    keys = np.where(np.isnan(flat), -np.inf, keys)
    order = np.argsort(-keys)[:k]
    return [
        (int(i // n_experts), int(i % n_experts), float(flat[i]))
        for i in order
        if not np.isnan(flat[i])
    ]
