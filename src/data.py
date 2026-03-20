"""TriviaQA data loading for Expert Pursuit.

Uses HF datasets.map() for faster tokenization instead of a Python loop.
"""

from typing import Any

from datasets import Dataset, load_dataset

# NOTE: I could add something like this:
# messages = [{"role": "user", "content":
#     "Answer the following question in 1–3 words only. Do not provide any additional explanation for your answer. "
#     "Question: " + question + " Answer:"
# }]

# TODO: Review the chat template for triviaQA


def _normalize_token_ids(raw: Any) -> list[int]:
    """Normalize apply_chat_template output to a flat list of ints."""
    if hasattr(raw, "input_ids"):
        raw = raw.input_ids
    if isinstance(raw, list) and raw and isinstance(raw[0], list):
        raw = raw[0]
    if not isinstance(raw, list):
        raw = list(raw)
    return raw


def load_triviaqa(
    tokenizer: Any,
    n_docs: int = 5000,
    split: str = "train",
    dataset: Dataset | None = None,
) -> Dataset:
    """Load and tokenize TriviaQA questions with the model's chat template.

    Args:
        tokenizer: HuggingFace tokenizer with apply_chat_template support.
        n_docs: Number of questions to load.
        split: Dataset split ("train" for encoding, "validation" for eval).
        dataset: Pre-loaded HF Dataset to skip download.

    Returns:
        HF Dataset with an ``input_ids`` column (list[int] per row).
    """
    if dataset is None:
        dataset = load_dataset(
            "mandarjoshi/trivia_qa", "rc", split=f"{split}[:{n_docs}]"
        )

    dataset = dataset.filter(
        lambda q: bool(q and q.strip()), input_columns=["question"]
    )

    def _tokenize(example: dict) -> dict:
        out = tokenizer.apply_chat_template(
            [{"role": "user", "content": example["question"].strip()}],
            add_generation_prompt=True,
            tokenize=True,
        )
        return {"input_ids": _normalize_token_ids(out)}

    return dataset.map(_tokenize).select(range(min(n_docs, len(dataset))))
