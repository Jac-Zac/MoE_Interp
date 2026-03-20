"""Utilities for augmenting pursuit dictionaries with common words."""

import re
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from datasets import load_dataset


@dataclass(frozen=True)
class WordDictionary:
    embeddings: torch.Tensor
    labels: list[str]
    base_vocab_size: int


def load_common_words(top_k: int = 10000, source: str = "default") -> list[str]:
    """Load common words from HuggingFace dataset, sorted by frequency.

    Args:
        top_k: Number of words to return
        source: Dataset source - "default" for jaagli/common-words-79k (missing common words),
                "full" for Maximax67/English-Valid-Words (includes the, a, is, etc.)
    """
    if source == "default":
        ds = load_dataset("jaagli/common-words-79k")["whole"]
        sorted_ds = ds.sort("frequency", reverse=True)
        return [w.strip().lower() for w in sorted_ds["alias"][:top_k] if w.strip()]
    elif source == "full":
        ds = load_dataset(
            "Maximax67/English-Valid-Words",
            subset="sorted_by_frequency",
        )["train"]
        return [w.strip().lower() for w in ds["Word"][:top_k] if w.strip()]
    else:
        raise ValueError(f"Unknown source: {source}. Use 'default' or 'full'")


def build_word_dictionary(
    tokenizer,
    dictionary: torch.Tensor,
    words: Sequence[str] | None = None,
    top_k: int = 10000,
    source: str = "default",
) -> WordDictionary:
    """Append averaged word atoms for words that split into multiple tokens,
    then remove the constituent sub-tokens from the base vocabulary.

    Uses a two-pass approach: first identifies multi-token words and their sub-tokens,
    then filters out sub-tokens that are also standalone promoted words.

    Args:
        tokenizer: Tokenizer for encoding words
        dictionary: Unembedding matrix (L2-normalized)
        words: Custom word list (if None, loads from source)
        top_k: Number of words to load from source
        source: "default" (jaagli) or "full" (Maximax67 with all common words)
    """
    if words is None:
        words = load_common_words(top_k, source=source)
    elif isinstance(words, (list, tuple)):
        words = list(words)[:top_k]

    # First pass: identify multi-token words and track their sub-tokens
    labels: list[str] = []
    seen: set[str] = set()
    subtoken_ids: set[int] = set()

    for word in words:
        cleaned = re.sub(r"[^\w]+$", "", word.strip()).lower()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)

        token_ids = tokenizer(" " + cleaned, add_special_tokens=False)["input_ids"]
        if len(token_ids) <= 1:
            continue

        subtoken_ids.update(token_ids)
        labels.append(cleaned)

    # Filter: remove subtokens that are also standalone promoted words
    promoted_words = set(labels)
    tokens_to_remove: set[int] = set()
    for tid in subtoken_ids:
        decoded = tokenizer.decode([tid]).strip()
        if decoded not in promoted_words:
            tokens_to_remove.add(tid)

    # Build filtered base dictionary (remove sub-tokens)
    base_vocab_size = dictionary.shape[0]
    keep_mask = torch.tensor(
        [i not in tokens_to_remove for i in range(base_vocab_size)], dtype=torch.bool
    )
    filtered_base = dictionary[keep_mask]

    # Second pass: build rows with filtered base + promoted words
    rows = [filtered_base]
    for word in labels:
        token_ids = tokenizer(" " + word, add_special_tokens=False)["input_ids"]
        # NOTE: Re-normalize after averaging — averaging k unit vectors produces
        # a vector with norm < 1, which would bias SOMP against merged words.
        rows.append(F.normalize(dictionary[token_ids].mean(dim=0, keepdim=True), dim=1))

    added = len(labels)
    removed = base_vocab_size - filtered_base.shape[0]
    new_size = filtered_base.shape[0] + added
    pct_added = (added / base_vocab_size) * 100 if base_vocab_size > 0 else 0
    pct_removed = (removed / base_vocab_size) * 100 if base_vocab_size > 0 else 0
    print(
        f"Added {added} multi-token words ({pct_added:.1f}% of original), "
        f"removed {removed} sub-tokens ({pct_removed:.1f}%), "
        f"final vocabulary: {new_size} rows"
    )

    return WordDictionary(
        embeddings=torch.cat(rows, dim=0) if len(labels) > 0 else filtered_base,
        labels=labels,
        base_vocab_size=filtered_base.shape[0],
    )
