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


# Minimal pairs for counterfactual editing: same template + length, the answer differs by one
# number. Used to test whether splicing the number experts' activity from the cf run into the
# factual run flips the predicted number (see circuit/editing.py). Each tuple is
# (factual_text, factual_answer, counterfactual_text, counterfactual_answer).
_NUMBER_PAIRS: list[tuple[str, str, str, str]] = [
    ("The sum of 2 and 3 is", "5", "The sum of 2 and 4 is", "6"),
    ("The sum of 1 and 4 is", "5", "The sum of 1 and 5 is", "6"),
    ("The sum of 3 and 4 is", "7", "The sum of 3 and 5 is", "8"),
    ("The sum of 2 and 2 is", "4", "The sum of 2 and 5 is", "7"),
    ("The sum of 4 and 4 is", "8", "The sum of 4 and 3 is", "7"),
    ("Counting up: 1, 2, 3, 4,", "5", "Counting up: 2, 3, 4, 5,", "6"),
    ("Counting up: 3, 4, 5, 6,", "7", "Counting up: 4, 5, 6, 7,", "8"),
    ("The number right after 6 is", "7", "The number right after 7 is", "8"),
    ("The number right after 4 is", "5", "The number right after 8 is", "9"),
    ("She had 2 apples and ate 1, leaving", "1", "She had 5 apples and ate 1, leaving", "4"),
]


def numbers_counterfactual_pairs(tokenizer) -> list[dict]:
    """Tokenised, validated minimal pairs for the numbers editing experiment.

    Keeps only pairs where (i) the factual and counterfactual prompts tokenise to the **same
    length** (so residual positions align 1:1 for the interchange) and (ii) both answers are a
    **single** token with a leading space (single-token log-prob eval). Each kept item is
    ``{"fact": ids, "cf": ids, "fact_ans": id, "cf_ans": id, "text": (fact_text, cf_text)}``.
    Extend ``_NUMBER_PAIRS`` (or write a sibling for another concept) to try other capabilities.
    """
    out: list[dict] = []
    for fact_text, fact_ans, cf_text, cf_ans in _NUMBER_PAIRS:
        fact = tokenizer(fact_text).input_ids
        cf = tokenizer(cf_text).input_ids
        a = tokenizer(" " + fact_ans, add_special_tokens=False).input_ids
        b = tokenizer(" " + cf_ans, add_special_tokens=False).input_ids
        if len(fact) != len(cf) or len(a) != 1 or len(b) != 1:
            continue
        out.append(
            {
                "fact": fact,
                "cf": cf,
                "fact_ans": a[0],
                "cf_ans": b[0],
                "text": (fact_text, cf_text),
            }
        )
    if not out:
        raise RuntimeError("no valid numbers minimal pairs survived tokenisation")
    return out


def rtp_split(
    tokenizer,
    *,
    n_train: int = 24,
    n_test: int = 24,
    hi: float = 0.5,
    lo: float = 0.1,
    seed: int = 0,
) -> tuple[list[list[int]], list[list[int]], list[list[int]], list[list[int]]]:
    """Disjoint ``(elic_train, elic_test, neut_train, neut_test)`` RTP partitions.

    Experts and the diff-of-means direction are identified on the *train* split; the
    intervention is then scored on the held-out *test* split, so "causal knockout suppresses
    toxicity, correlational knockout does not" is measured out-of-sample. The two halves are
    a deterministic slice of one shuffled stream, so they never overlap.
    """
    elic, neut = rtp_prompts(tokenizer, n=n_train + n_test, hi=hi, lo=lo, seed=seed)
    return (
        elic[:n_train],
        elic[n_train:],
        neut[:n_train],
        neut[n_train:],
    )
