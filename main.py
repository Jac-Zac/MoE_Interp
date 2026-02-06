#!/usr/bin/env python

import torch
from nnsight import LanguageModel

from src.capture import capture_moe_activations
from src.data import load_pile_docs
from src.environment import set_seed

# Configuration
SEED = 1337
N_DOCS = 10
BATCH_SIZE = 8
MAX_TOKENS = 100


def main():
    set_seed(SEED)

    model = LanguageModel(
        "allenai/OLMoE-1B-7B-0924-Instruct",
        device_map="auto",
        dtype=torch.float16,
        dispatch=True,
    )

    docs = load_pile_docs(
        tokenizer=model.tokenizer,
        n_docs=N_DOCS,
        max_tokens=MAX_TOKENS,
        dataset_name="NeelNanda/pile-10k",
    )

    print(f"Loaded {len(docs)} documents\n")

    traces = capture_moe_activations(model, docs, batch_size=BATCH_SIZE)

    first_trace = traces[0]
    print(f"\nTotal batches: {len(traces)}")
    print(f"Example from batch 0:")
    print(f"  Total tokens: {len(first_trace.token_ids)}")
    print(
        f"  Shape: [tokens={first_trace.token_ids.shape[0]}, layers={first_trace.n_layers}, k={first_trace.k}]"
    )

    expert_ids, weights = first_trace.experts_for_token(0, 0)
    print("\nLayer 0, Token 0:")
    print(f"  Expert ids: {expert_ids.tolist()}")
    print(f"  Weights: {weights.tolist()}")


if __name__ == "__main__":
    main()
