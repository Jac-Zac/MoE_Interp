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
- ``run_set_ablation`` — ablate a *whole flagged set jointly* vs matched random control
  sets, reporting a z-score: the significance test for an identified circuit (e.g. the
  top-k gate-AtP experts), which the marginal per-expert patching grid cannot give.

The boundary tap mirrors ``capture.py``: ``layer.mlp.experts.inputs[0]`` yields
``(hidden_states, top_k_index, top_k_weights)`` for the fused experts module, which is
the only point where per-expert routing is exposed on transformers >= 5.9. The metric is
correlational about the probe but the intervention itself is causal.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

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


@dataclass
class SetAblationResult:
    """Ablating a flagged expert set vs ``n`` matched random control sets."""

    flagged_delta: float
    control_deltas: list[float]
    control_mean: float
    control_std: float
    zscore: float  # (flagged - control_mean) / control_std
    percentile: float  # fraction of control sets the flagged set beats


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


def run_set_ablation(
    model,
    prompts: list[list[int]],
    flagged: list[tuple[int, int]],
    toxic_ids: list[int],
    n_controls: int = 30,
    batch_size: int = 8,
    metric: Metric = relative_logit_score,
    seed: int = 1337,
) -> SetAblationResult:
    """Ablate the whole flagged set vs random control sets matched in size & layers.

    Each control draws, for every flagged ``(layer, expert)``, a different random expert
    in the *same layer* — so layer position and set size are held fixed and only the
    identity of the experts varies. A large positive z-score means the flagged experts
    drive the toxic probe well beyond what arbitrary same-layer experts do.
    """
    n_experts = model.config.num_local_experts
    gen = torch.Generator().manual_seed(seed)
    flagged_layers = [layer for layer, _ in flagged]
    flagged_set = {(layer, e) for layer, e in flagged}

    with right_padded(model):
        score = scorer(model, prompts, toxic_ids, metric, batch_size)
        base = score(None)

        flagged_delta = float((base - score(flagged)).mean())

        control_deltas: list[float] = []
        for _ in range(n_controls):
            ctrl: list[tuple[int, int]] = []
            for layer in flagged_layers:
                while True:
                    e = int(torch.randint(n_experts, (1,), generator=gen).item())
                    if (layer, e) not in flagged_set and (layer, e) not in ctrl:
                        break
                ctrl.append((layer, e))
            control_deltas.append(float((base - score(ctrl)).mean()))

        cd = torch.tensor(control_deltas)
        std = float(cd.std(unbiased=False)) or 1e-9
        return SetAblationResult(
            flagged_delta=flagged_delta,
            control_deltas=control_deltas,
            control_mean=float(cd.mean()),
            control_std=float(cd.std(unbiased=False)),
            zscore=(flagged_delta - float(cd.mean())) / std,
            percentile=float((cd < flagged_delta).float().mean()),
        )
