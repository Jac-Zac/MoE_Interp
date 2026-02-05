import torch
from datasets import load_dataset


def load_pile_docs(
    tokenizer,
    n_docs: int = 100,
    max_tokens: int = 512,
    seed: int = 1337,
) -> torch.Tensor:
    """
    Load documents from The Pile that fit within max_tokens.

    Respects document boundaries by only keeping complete documents
    and padding shorter ones to enable nnsight batching.

    Args:
        tokenizer: HuggingFace tokenizer
        n_docs: Number of documents to load
        max_tokens: Maximum tokens per document (context window)
        seed: Random seed for shuffling

    Returns:
        Tensor of shape [n_docs, max_tokens] with padded token IDs
    """
    dataset = load_dataset(
        "NeelNanda/pile-10k",
        split="train",
        streaming=True,
    ).shuffle(seed=seed, buffer_size=10000)

    docs = []
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
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
            # Pad to max_tokens
            tokens = tokens + [pad_id] * (max_tokens - len(tokens))
            docs.append(tokens)

    return torch.tensor(docs, dtype=torch.long)
