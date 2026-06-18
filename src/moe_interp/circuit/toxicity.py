"""Single-expert gate-ablation on toxic-eliciting prompts.

We test whether the experts Expert Pursuit flags as *toxicity specialists* are also
*causally* responsible for toxic continuations. The intervention is a clean ablation:
inside one traced forward we zero an expert's router gate weight wherever it was
selected, removing exactly that expert's additive contribution to the residual stream
(the other top-k experts are untouched, since OLMoE does not renormalise the gates
after selection). We then read the next-token distribution at the last prompt position
and measure the drop in probability mass on a toxic-token probe.

The boundary tap mirrors ``capture.py``: ``layer.mlp.experts.inputs[0]`` yields
``(hidden_states, top_k_index, top_k_weights)`` for the fused experts module, which is
the only point where per-expert routing is exposed on transformers >= 5.9.

A toxicity specialist should show a *positive* effect (ablation lowers P(toxic));
matched random control experts should cluster near zero. The metric is correlational
about the probe but the intervention itself is causal.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from moe_interp.pursuit.concepts import CONCEPT_WORDS

Metric = Callable[[torch.Tensor, list[int]], torch.Tensor]


@contextmanager
def right_padded(model) -> Iterator[None]:
    """Force right-padding for the block so each prompt's last real token sits at
    ``len - 1`` and (causally downstream) padding cannot leak into it; restore on exit."""
    original = model.tokenizer.padding_side
    model.tokenizer.padding_side = "right"
    try:
        yield
    finally:
        model.tokenizer.padding_side = original


@dataclass
class ExpertAblationResult:
    layer: int
    expert: int
    group: str  # "candidate" | "control"
    delta_mean: float  # mean over prompts of (score_base - score_ablated)
    delta_sem: float  # standard error of the mean of the per-prompt deltas


@dataclass
class SetAblationResult:
    """Ablating a flagged expert set vs ``n`` matched random control sets."""

    flagged_delta: float
    control_deltas: list[float]
    control_mean: float
    control_std: float
    zscore: float  # (flagged - control_mean) / control_std
    percentile: float  # fraction of control sets the flagged set beats


def build_toxic_token_ids(tokenizer, words: list[str] | None = None) -> list[int]:
    """Vocabulary ids of single-token toxic words (with and without a leading space)."""
    words = words or CONCEPT_WORDS["offensive"]
    ids: set[int] = set()
    for w in words:
        for variant in (w, " " + w):
            toks = tokenizer(variant, add_special_tokens=False).input_ids
            if len(toks) == 1:
                ids.add(int(toks[0]))
    return sorted(ids)


def toxic_probability(logits_last: torch.Tensor, toxic_ids: list[int]) -> torch.Tensor:
    """P(next token in toxic set) for each row of ``logits_last`` ``(B, V)``."""
    probs = F.softmax(logits_last.float(), dim=-1)
    return probs[:, toxic_ids].sum(dim=-1)


def toxic_logit_score(logits_last: torch.Tensor, toxic_ids: list[int]) -> torch.Tensor:
    """Mean logit on toxic tokens minus the row mean logit (relative elevation).

    More sensitive than ``toxic_probability``: the softmax floor over a 50k vocab makes
    the absolute probability of any specific content word tiny, so small causal shifts
    are easier to read on the (centred) logits.
    """
    logits = logits_last.float()
    return logits[:, toxic_ids].mean(dim=-1) - logits.mean(dim=-1)


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


def _scorer(
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


def run_expert_ablation(
    model,
    prompts: list[list[int]],
    candidates: list[tuple[int, int]],
    controls: list[tuple[int, int]],
    toxic_ids: list[int],
    batch_size: int = 8,
    metric: Metric = toxic_logit_score,
) -> list[ExpertAblationResult]:
    """Ablate each (layer, expert) one at a time; report Δ score vs controls.

    Prompts are pre-tokenised id lists.
    """
    with right_padded(model):
        score = _scorer(model, prompts, toxic_ids, metric, batch_size)
        base = score(None)
        results: list[ExpertAblationResult] = []
        for group, experts in (("candidate", candidates), ("control", controls)):
            for layer_idx, expert_id in experts:
                delta = base - score([(layer_idx, expert_id)])
                results.append(
                    ExpertAblationResult(
                        layer=layer_idx,
                        expert=expert_id,
                        group=group,
                        delta_mean=float(delta.mean()),
                        delta_sem=float(delta.std(unbiased=False) / (len(delta) ** 0.5)),
                    )
                )
        return results


def run_set_ablation(
    model,
    prompts: list[list[int]],
    flagged: list[tuple[int, int]],
    toxic_ids: list[int],
    n_controls: int = 30,
    batch_size: int = 8,
    metric: Metric = toxic_logit_score,
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
        score = _scorer(model, prompts, toxic_ids, metric, batch_size)
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
