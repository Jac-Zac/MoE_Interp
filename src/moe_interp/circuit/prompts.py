"""Prompts for the circuit / intervention experiments — a RealToxicityPrompts split.

:func:`rtp_prompts` partitions real RealToxicityPrompts by their own per-prompt toxicity
score into high-toxicity (eliciting) and low-toxicity (neutral) sets. A diff-of-means over
the two isolates the toxic direction, and the eliciting set drives the knockout/project-out
comparison. This is the single prompt source for the whole circuit study; the notebooks use
the same function with a small ``n`` and print a few examples for clarity.
"""

from __future__ import annotations


def rtp_prompts(
    tokenizer,
    *,
    n: int = 48,
    hi: float = 0.5,
    lo: float = 0.1,
    seed: int = 0,
) -> tuple[list[list[int]], list[list[int]]]:
    """Tokenised ``(eliciting, neutral)`` prompts from RealToxicityPrompts.

    Streams the shuffled dataset and partitions prompts by their own toxicity score:
    ``toxicity >= hi`` goes to the eliciting set, ``<= lo`` to the neutral set, until each
    holds ``n`` prompts. Each is the bare prompt text (the model completes it). Decode an
    id-list with ``tokenizer.decode(ids)`` to inspect a prompt. Requires the
    ``allenai/real-toxicity-prompts`` dataset to be available (cached when offline).
    """
    from datasets import load_dataset

    ds = load_dataset("allenai/real-toxicity-prompts", split="train").shuffle(seed=seed)
    eliciting: list[list[int]] = []
    neutral: list[list[int]] = []
    for ex in ds:
        prompt = ex["prompt"]
        tox = prompt.get("toxicity")
        text = (prompt.get("text") or "").strip()
        if not text or tox is None:
            continue
        if tox >= hi and len(eliciting) < n:
            eliciting.append(tokenizer(text).input_ids)
        elif tox <= lo and len(neutral) < n:
            neutral.append(tokenizer(text).input_ids)
        if len(eliciting) >= n and len(neutral) >= n:
            break
    if not eliciting or not neutral:
        raise RuntimeError(
            f"RealToxicityPrompts yielded {len(eliciting)} eliciting / {len(neutral)} "
            "neutral prompts; loosen hi/lo or check the dataset is available."
        )
    return eliciting, neutral
