"""Shared primitives for the causal gate interventions.

- ``relative_logit_score`` — the concept-logit probe (see its docstring).
- ``right_padded`` — trace plumbing shared with ``attribution`` (gate-AtP).

The boundary tap mirrors ``capture.py``: ``layer.mlp.experts.inputs[0]`` yields
``(hidden_states, top_k_index, top_k_weights)`` for the fused experts module, which is
the only point where per-expert routing is exposed on transformers >= 5.9.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

import torch


@contextmanager
def right_padded(model) -> Generator[None, None, None]:
    """Force right-padding for the block so each prompt's last real token sits at
    ``len - 1`` and (causally downstream) padding cannot leak into it; restore on exit.
    """
    original = model.tokenizer.padding_side
    model.tokenizer.padding_side = "right"
    try:
        yield
    finally:
        model.tokenizer.padding_side = original


def relative_logit_score(
    logits_last: torch.Tensor, concept_ids: list[int]
) -> torch.Tensor:
    """Mean logit on the concept tokens minus the row mean logit (relative elevation).

    More sensitive than a raw ``P(next token in concept set)`` probe: the softmax floor
    over a 50k vocab makes the absolute probability of any specific content word tiny, so
    small causal shifts are easier to read on the (centred) logits. Concept-agnostic — the
    harmful-content proxy is this score with an offensive-word token set.
    """
    logits = logits_last.float()
    return logits[:, concept_ids].mean(dim=-1) - logits.mean(dim=-1)
