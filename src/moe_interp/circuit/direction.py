"""Gather last-token residuals for the diff-of-means toxicity *direction*.

Toxicity is isolated as a *direction* in the residual stream rather than a single
expert::

    v = mean(h | toxic-eliciting prompts) - mean(h | neutral prompts)   (diff-of-means)

This module only collects the last-token residuals; ``steer.py`` differences them into
``v`` and the causal edits on ``v`` (steering, project-out) live in ``intervene.py``.
"""

from __future__ import annotations

import torch

from moe_interp.circuit.toxicity import right_padded


def collect_last_token_residuals(
    model, prompts: list[list[int]], layer: int, batch_size: int = 8
) -> torch.Tensor:
    """Residual stream at ``layer`` output, gathered at each prompt's last real token."""
    chunks: list[torch.Tensor] = []
    with right_padded(model):
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            lengths = torch.tensor([len(t) for t in batch])
            with torch.no_grad(), model.trace(batch):
                # decoder layer .output is the bare hidden-states tensor (B, T, D)
                hs = model.model.layers[layer].output.save()
            rows = torch.arange(hs.shape[0])
            chunks.append(hs[rows, lengths - 1].float().cpu())
    return torch.cat(chunks)
