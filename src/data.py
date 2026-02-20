"""Data loading for Expert Pursuit.

Loads TriviaQA questions and tokenizes them using the model's chat template.
Following HeadPursuit, raw questions are used for encoding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import Dataset, load_dataset


@dataclass
class TokenizedQuestion:
    """A tokenized TriviaQA question with content-token boundaries.

    Attributes:
        token_ids: Full token sequence (with chat template markers).
        content_start: Index of the first question-content token.
        content_end: Index one past the last question-content token.
        source_idx: Original index in the HF dataset.
    """

    token_ids: list[int]
    content_start: int
    content_end: int
    source_idx: int


def _find_content_boundaries(
    token_ids: list[int],
    tokenizer: Any,
) -> tuple[int, int]:
    """Find start/end indices of question-content tokens within chat template.

    The OLMoE chat template produces:
        <|endoftext|> <|user|> \\n {question tokens} \\n <|assistant|> \\n
    We want only the {question tokens} portion.

    Returns:
        (content_start, content_end) indices into token_ids.
    """
    # Encode the special tokens to find their IDs
    user_token = tokenizer.encode("<|user|>", add_special_tokens=False)
    assistant_token = tokenizer.encode("<|assistant|>", add_special_tokens=False)

    # Find <|user|> token position
    user_pos = None
    if user_token:
        uid = user_token[0]
        for i, tid in enumerate(token_ids):
            if tid == uid:
                user_pos = i
                break

    # Find <|assistant|> token position (search from end)
    assistant_pos = None
    if assistant_token:
        aid = assistant_token[0]
        for i in range(len(token_ids) - 1, -1, -1):
            if token_ids[i] == aid:
                assistant_pos = i
                break

    # Content starts after <|user|> + newline token
    if user_pos is not None:
        content_start = user_pos + 2  # skip <|user|> and \n
    else:
        content_start = 0

    # Content ends before \n <|assistant|>
    if assistant_pos is not None:
        content_end = assistant_pos - 1  # exclude \n before <|assistant|>
    else:
        content_end = len(token_ids)

    # Sanity: ensure valid range
    content_start = max(0, min(content_start, len(token_ids)))
    content_end = max(content_start, min(content_end, len(token_ids)))

    return content_start, content_end


def load_triviaqa(
    tokenizer: Any,
    n_docs: int = 5000,
    split: str = "train",
    dataset: Dataset | None = None,
) -> list[TokenizedQuestion]:
    """Load and tokenize TriviaQA questions with the model's chat template.

    Each question is wrapped in the OLMoE chat template (no QA prompt).
    Token boundaries are computed so that only question-content tokens
    (excluding <|user|>, <|assistant|>, etc.) are used for aggregation.

    Args:
        tokenizer: HuggingFace tokenizer with apply_chat_template support.
        n_docs: Number of questions to load.
        split: Dataset split ("train" for encoding, "validation" for eval).
        dataset: Pre-loaded HF Dataset to skip download. Must have a
            "question" column. Pass this in notebooks to avoid re-downloading.

    Returns:
        List of TokenizedQuestion with token IDs and content boundaries.
    """
    if dataset is None:
        dataset = load_dataset("mandarjoshi/trivia_qa", "rc", split=split)

    questions: list[TokenizedQuestion] = []

    for idx in range(len(dataset)):
        if len(questions) >= n_docs:
            break

        question_text = dataset[idx].get("question", "").strip()
        if not question_text:
            continue

        # Wrap in chat template (raw question, no QA prompt)
        messages = [{"role": "user", "content": question_text}]
        token_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
        )

        content_start, content_end = _find_content_boundaries(token_ids, tokenizer)

        questions.append(
            TokenizedQuestion(
                token_ids=token_ids,
                content_start=content_start,
                content_end=content_end,
                source_idx=idx,
            )
        )

    return questions
