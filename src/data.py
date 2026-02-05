from typing import List

from datasets import load_dataset


def load_pile_docs(
    tokenizer,
    n_docs: int = 100,
    max_tokens: int = 512,
) -> List[List[int]]:
    """
    Load documents from The Pile that fit within max_tokens.

    Respects document boundaries by only keeping complete documents.
    Returns list of token sequences for nnsight to handle batching/padding.
    Documents are loaded sequentially without shuffling for efficiency.

    Args:
        tokenizer: HuggingFace tokenizer
        n_docs: Number of documents to load
        max_tokens: Maximum tokens per document (context window)

    Returns:
        List of token ID sequences, each with length <= max_tokens
    """
    dataset = load_dataset(
        "NeelNanda/pile-10k",
        split="train",
        streaming=True,
    )

    docs = []

    # Use the model eos token else use the custom one defined here
    eos_id = (
        tokenizer.eos_token_id
        or tokenizer.encode("<|endoftext|>", add_special_tokens=False)[0]
    )

    for example in dataset:
        if len(docs) >= n_docs:
            break

        text = example.get("text", "")
        if not text or len(text.strip()) <= 50:
            continue

        tokens = tokenizer.encode(text, add_special_tokens=False)

        # Only keep docs that fit entirely (respect boundaries)
        # +1 for EOS token
        if len(tokens) + 1 <= max_tokens:
            tokens = tokens + [eos_id]
            docs.append(tokens)

    return docs
