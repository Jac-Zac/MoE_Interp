"""Expert activation patching — the per-(layer, expert) causal effect grid.

The whole-set ablation in :mod:`moe_interp.circuit.toxicity` answers "does *this* expert
set matter?"; this sweeps the intervention over *every* expert to draw the full causal map
the driver-experts literature uses (arXiv:2601.10159): for each ``(layer, expert)`` we zero
its router gate (an ablation patch) on a toxic-eliciting prompt set and record the change in
the toxic-logit metric. Positive effect ⇒ the expert pushes toxicity up (ablating it lowers
the score). The result is an ``(n_layers, n_experts)`` grid, plotted as a heatmap.

Cost note: a brute-force grid is one forward per expert. We skip experts that are *never*
selected at any position across the prompt batch — their gate ablation is identically a
no-op — which removes the bulk of the 64-way fan-out for short prompts at zero cost to
correctness. For the cheap one-backward-pass estimate of this same grid see
:func:`moe_interp.circuit.attribution.gate_attribution` (AtP).
"""

from __future__ import annotations

import torch

from moe_interp.circuit.toxicity import (
    Metric,
    relative_logit_score,
    right_padded,
    scorer,
)


def selected_experts(
    model, prompts: list[list[int]], batch_size: int = 6
) -> dict[int, set[int]]:
    """Experts selected at any token/position across the batch, per layer (one trace).

    An expert never routed anywhere cannot affect the output when its gate is zeroed, so
    it is safe (and much cheaper) to skip it in the grid sweep.
    """
    n_layers = model.config.num_hidden_layers
    out: dict[int, set[int]] = {layer: set() for layer in range(n_layers)}
    with right_padded(model):
        for i in range(0, len(prompts), batch_size):
            idx_saved = []
            with torch.no_grad(), model.trace(prompts[i : i + batch_size]):
                for layer in model.model.layers:
                    _, top_k_index, _ = layer.mlp.experts.inputs[0]
                    idx_saved.append(top_k_index.save())
            for layer in range(n_layers):
                out[layer].update(int(e) for e in idx_saved[layer].flatten().tolist())
    return out


def expert_patching_grid(
    model,
    prompts: list[list[int]],
    toxic_ids: list[int],
    *,
    metric: Metric = relative_logit_score,
    batch_size: int = 6,
    layers: list[int] | None = None,
) -> torch.Tensor:
    """``(n_layers, n_experts)`` causal-effect grid from single-expert gate ablation.

    ``grid[l, e] = mean_prompts(metric_base - metric_ablate(l, e))`` — positive means
    ablating expert ``e`` in layer ``l`` lowers the toxic score (the expert promotes it).
    Experts never routed in the batch are left at 0. Restrict ``layers`` for a faster run.
    """
    n_layers = model.config.num_hidden_layers
    n_experts = model.config.num_local_experts
    want = set(layers) if layers is not None else set(range(n_layers))
    grid = torch.zeros(n_layers, n_experts)

    active = selected_experts(model, prompts, batch_size)
    with right_padded(model):
        score = scorer(model, prompts, toxic_ids, metric, batch_size)
        base = score(None)
        for layer in sorted(want):
            for expert in sorted(active[layer]):
                delta = base - score([(layer, expert)])
                grid[layer, expert] = float(delta.mean())
    return grid


def top_grid_experts(grid: torch.Tensor, k: int = 20) -> list[dict]:
    """The ``k`` experts with the largest |effect|, as JSON-friendly records."""
    from moe_interp.grids import top_experts

    return [
        {"layer": layer, "expert": e, "effect": v}
        for layer, e, v in top_experts(grid, k, by="abs")
    ]


def plot_expert_effect_grid(grid: torch.Tensor, output_path, *, title: str) -> None:
    """Save a layer×expert heatmap of the causal effect (diverging, centred at 0)."""
    from moe_interp.io.plots import diverging_expert_heatmap

    diverging_expert_heatmap(
        grid,
        title=title,
        colorbar_title="Δ toxic score<br>(ablation effect)",
        output_path=output_path,
    )
