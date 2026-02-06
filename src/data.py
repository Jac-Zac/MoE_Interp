"""Data loading utilities for MoE interpretability."""

from typing import Any, Optional, Tuple

from datasets import Dataset, load_dataset


class PileLoader:
    """Loader for The Pile dataset."""

    def __init__(
        self,
        tokenizer: Any,
        max_tokens: int = 512,
        dataset_name: str = "NeelNanda/pile-10k",
    ):
        """Initialize the loader."""
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.dataset_name = dataset_name
        self._eos_id: Optional[int] = None
        self._dataset: Optional[Dataset] = None

    def _get_eos_id(self) -> int:
        """Get or compute EOS token ID."""
        if self._eos_id is None:
            eos_id = self.tokenizer.eos_token_id
            if eos_id is None:
                eos_id = self.tokenizer.encode(
                    "<|endoftext|>", add_special_tokens=False
                )[0]

            self._eos_id = int(eos_id)
        return self._eos_id

    def _load_dataset(self) -> Dataset:
        """Load dataset once to hugging face default storage location"""
        if self._dataset is None:
            self._dataset = load_dataset(self.dataset_name, split="train")
        return self._dataset

    def _tokenize_doc(self, text: str) -> Optional[list[int]]:
        """Tokenize a single document, returns None if invalid."""
        if not text or len(text.strip()) <= 20:
            return None

        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
        except Exception:
            return None

        # Only keep docs that fit entirely (respect boundaries)
        # +1 for EOS token
        if len(tokens) + 1 > self.max_tokens:
            return None

        tokens.append(self._get_eos_id())
        return tokens

    def load_n_docs(self, n_docs: int = 100) -> Tuple[list[list[int]], list[int]]:
        """Load exactly n_docs documents that fit within max_tokens.

        Returns:
            Tuple of (token_lists, source_doc_indices) where source_doc_indices
            maps each loaded doc to its original index in the dataset.
        """
        dataset = self._load_dataset()
        tokens_list: list[list[int]] = []
        source_indices: list[int] = []

        for dataset_idx, example in enumerate(dataset):
            if len(tokens_list) >= n_docs:
                break

            ex: dict = example  # type: ignore
            tokens = self._tokenize_doc(ex.get("text", ""))
            if tokens is not None:
                tokens_list.append(tokens)
                source_indices.append(dataset_idx)

        return tokens_list, source_indices


def load_pile_docs(
    tokenizer: Any,
    n_docs: int = 100,
    max_tokens: int = 512,
    dataset_name: str = "NeelNanda/pile-10k",
) -> Tuple[list[list[int]], list[int]]:
    """Load documents from The Pile that fit within max_tokens.

    Returns:
        Tuple of (token_lists, source_doc_indices) where source_doc_indices
        maps each loaded doc to its original index in the dataset.
    """
    loader = PileLoader(tokenizer, max_tokens, dataset_name)
    return loader.load_n_docs(n_docs)
