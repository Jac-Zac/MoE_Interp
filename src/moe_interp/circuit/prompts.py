"""Prompts for the circuit / intervention experiments — a RealToxicityPrompts split.

:func:`rtp_split` partitions real RealToxicityPrompts by each prompt's own toxicity score into
high-toxicity (eliciting) and low-toxicity (neutral) sets, then into disjoint train (identify)
and test (score) halves. The eliciting set drives the expert knockout / expert-output steering
comparison and the neutral set is the collateral control. This is the single prompt source for
the whole circuit study; the notebooks call ``rtp_split`` with a small ``n`` for a quick look.
"""

from __future__ import annotations


def _rtp_prompts(
    tokenizer,
    *,
    n: int = 48,
    hi: float = 0.5,
    lo: float = 0.1,
    seed: int = 0,
    challenging: bool = False,
) -> tuple[list[list[int]], list[list[int]]]:
    """Tokenised ``(eliciting, neutral)`` prompts from RealToxicityPrompts.

    Streams the shuffled dataset and partitions prompts by their own toxicity score:
    ``toxicity >= hi`` goes to the eliciting set, ``<= lo`` to the neutral set, until each
    holds ``n`` prompts. Each is the bare prompt text (the model completes it). Decode an
    id-list with ``tokenizer.decode(ids)`` to inspect a prompt. Requires the
    ``allenai/real-toxicity-prompts`` dataset to be available (cached when offline).

    ``challenging=True`` restricts the *eliciting* set to RealToxicityPrompts' curated
    ``challenging`` subset (prompts that reliably trigger toxic degeneration), which raises the
    base toxicity rate so a lexical word-fraction metric has real dynamic range.
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
        if tox >= hi and len(eliciting) < n and not (challenging and not ex.get("challenging")):
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


def rtp_split(
    tokenizer,
    *,
    n_train: int = 24,
    n_test: int = 24,
    hi: float = 0.5,
    lo: float = 0.1,
    seed: int = 0,
    challenging: bool = False,
) -> tuple[list[list[int]], list[list[int]], list[list[int]], list[list[int]]]:
    """Disjoint ``(elic_train, elic_test, neut_train, neut_test)`` RTP partitions.

    Experts and the diff-of-means direction are identified on the *train* split; the
    intervention is then scored on the held-out *test* split, so "causal knockout suppresses
    toxicity, correlational knockout does not" is measured out-of-sample. The two halves are
    a deterministic slice of one shuffled stream, so they never overlap.

    Raise ``hi`` and/or set ``challenging=True`` for a higher-toxicity regime where a lexical
    word-fraction metric has real dynamic range (so "can we drive toxicity to ~0 on toxic
    sentences" is measurable, not lost in noise).
    """
    elic, neut = _rtp_prompts(
        tokenizer, n=n_train + n_test, hi=hi, lo=lo, seed=seed, challenging=challenging
    )
    return (
        elic[:n_train],
        elic[n_train:],
        neut[:n_train],
        neut[n_train:],
    )
