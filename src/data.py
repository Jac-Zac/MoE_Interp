"""Data loading utilities for MoE interpretability."""

from typing import Any, Optional

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
        """
        Get or compute EOS token ID. Supporting even when not available
        """
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

        # The default storage location is set to be scratch with env variables
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

    def load_n_docs(self, n_docs: int = 100) -> list[list[int]]:
        """Load exactly n_docs documents that fit within max_tokens."""
        dataset = self._load_dataset()
        docs: list[list[int]] = []

        for example in dataset:
            if len(docs) >= n_docs:
                break

            ex: dict[str, Any] = example  # type: ignore
            tokens = self._tokenize_doc(ex.get("text", ""))
            if tokens is not None:
                docs.append(tokens)

        return docs


def load_pile_docs(
    tokenizer: Any,
    n_docs: int = 100,
    max_tokens: int = 512,
    dataset_name: str = "NeelNanda/pile-10k",
) -> list[list[int]]:
    """Load documents from The Pile that fit within max_tokens."""
    loader = PileLoader(tokenizer, max_tokens, dataset_name)
    return loader.load_n_docs(n_docs)
