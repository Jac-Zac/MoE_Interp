"""TriviaQA data loading for Expert Pursuit.

Simple loader following HeadPursuit's approach: returns tokenized prompts
as list of token ID lists. No content boundary tracking needed since we
capture at the last token position.
"""

from __future__ import annotations

from typing import Any

from datasets import Dataset, load_dataset


def load_triviaqa(
    tokenizer: Any,
    n_docs: int = 5000,
    split: str = "train",
    dataset: Dataset | None = None,
) -> list[list[int]]:
    """Load and tokenize TriviaQA questions with the model's chat template.

    Args:
        tokenizer: HuggingFace tokenizer with apply_chat_template support.
        n_docs: Number of questions to load.
        split: Dataset split ("train" for encoding, "validation" for eval).
        dataset: Pre-loaded HF Dataset to skip download.

    Returns:
        List of token ID lists, each wrapped in the model's chat template.
    """
    if dataset is None:
        dataset = load_dataset("mandarjoshi/trivia_qa", "rc", split=split)

    prompts: list[list[int]] = []

    for row in dataset:
        if len(prompts) >= n_docs:
            break

        question = dict(row).get("question", "").strip()
        if not question:
            continue

        messages = [{"role": "user", "content": question}]
        token_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )

        if hasattr(token_ids, "input_ids"):
            token_ids = token_ids.input_ids
            if isinstance(token_ids[0], list):
                token_ids = token_ids[0]
        elif not isinstance(token_ids, list):
            token_ids = list(token_ids)

        if token_ids:
            prompts.append(token_ids)

    return prompts
