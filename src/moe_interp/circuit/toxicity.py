"""Gate-ablation primitives + the whole-set significance test.

The shared building blocks for every causal gate intervention in this package. The
intervention is a clean ablation: inside one traced forward we zero an expert's router
gate weight wherever it was selected, removing exactly that expert's additive
contribution to the residual stream (the other top-k experts are untouched, since OLMoE
does not renormalise the gates after selection). We then read the next-token distribution
at the last prompt position and score it with a concept-logit probe.

- ``relative_logit_score`` / ``Metric`` — the probe (see ``relative_logit_score``).
- ``right_padded`` / ``scorer`` — shared trace plumbing, reused by ``patching`` (the
  per-expert grid) and ``attribution`` (gate-AtP).

The boundary tap mirrors ``capture.py``: ``layer.mlp.experts.inputs[0]`` yields
``(hidden_states, top_k_index, top_k_weights)`` for the fused experts module, which is
the only point where per-expert routing is exposed on transformers >= 5.9. The metric is
correlational about the probe but the intervention itself is causal.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import torch

Metric = Callable[[torch.Tensor, list[int]], torch.Tensor]


@contextmanager
def right_padded(model) -> Iterator[None]:
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
    toxic-logit metric is just this with an offensive-word id set.
    """
    logits = logits_last.float()
    return logits[:, concept_ids].mean(dim=-1) - logits.mean(dim=-1)


def _last_token_logits(
    model,
    batch_tokens: list[list[int]],
    ablate: list[tuple[int, int]] | None = None,
) -> torch.Tensor:
    """Trace one right-padded batch, optionally zeroing gates of ``ablate`` experts.

    Returns ``(B, V)`` logits at each prompt's last real token.
    """
    lengths = torch.tensor([len(t) for t in batch_tokens])
    with torch.no_grad(), model.trace(batch_tokens):
        if ablate:
            # nnsight 0.7 requires touching Envoys in forward (layer) order.
            for layer_idx, expert_id in sorted(ablate):
                _, top_k_index, top_k_weights = model.model.layers[
                    layer_idx
                ].mlp.experts.inputs[0]
                top_k_weights[top_k_index == expert_id] = 0.0
        logits = model.output.logits.save()
    rows = torch.arange(logits.shape[0])
    return logits[rows, lengths - 1].cpu()


def scorer(
    model,
    prompts: list[list[int]],
    toxic_ids: list[int],
    metric: Metric,
    batch_size: int,
) -> Callable[[list[tuple[int, int]] | None], torch.Tensor]:
    """Return a function mapping an ablation set to per-prompt scores (right-padded)."""
    batches = [prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)]

    def score(ablate: list[tuple[int, int]] | None) -> torch.Tensor:
        return torch.cat(
            [metric(_last_token_logits(model, b, ablate), toxic_ids) for b in batches]
        )

    return score
