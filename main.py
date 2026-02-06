#!/usr/bin/env python

import torch
from nnsight import LanguageModel
from pathlib import Path

from src.capture import capture_moe_activations
from src.checkpoint import load_batch, list_batch_info, get_data_dir
from src.data import load_pile_docs
from src.environment import set_seed

# Configuration
SEED = 1337
N_DOCS = 10
BATCH_SIZE = 4
MAX_TOKENS = 100


def main():
    set_seed(SEED)

    model = LanguageModel(
        "allenai/OLMoE-1B-7B-0924-Instruct",
        device_map="auto",
        dtype=torch.float16,
        dispatch=True,
    )

    # Load docs and their source indices
    docs, doc_source_ids = load_pile_docs(
        tokenizer=model.tokenizer,
        n_docs=N_DOCS,
        max_tokens=MAX_TOKENS,
        dataset_name="NeelNanda/pile-10k",
    )

    print(f"Loaded {len(docs)} documents")
    print(f"Source doc indices (first 5): {doc_source_ids[:5]}")
    print()

    # Capture and save to disk
    data_dir = get_data_dir()
    print(f"Saving checkpoints to: {data_dir}")
    print()

    batch_info = capture_moe_activations(
        model,
        docs=docs,
        doc_source_ids=doc_source_ids,
        batch_size=BATCH_SIZE,
        save_dir=data_dir,
    )

    print(f"\n\nCaptured {len(batch_info)} batches")
    print("\nAvailable checkpoints:")
    for info in list_batch_info(data_dir):
        print(
            f"  Batch {info['batch_idx']}: docs {info['doc_range']} -> {info['filename']}"
        )

    # Load batch 0 and show source tracking
    print("\n\nLoading batch 0 from disk...")
    trace, doc_range = load_batch(0, data_dir)

    print(f"\nBatch 0 info:")
    print(f"  Docs in batch: {doc_range[0]} to {doc_range[1] - 1}")
    print(f"  Total tokens: {len(trace.token_ids)}")
    print(
        f"  Shape: [layers={trace.n_layers}, tokens={len(trace.token_ids)}, k={trace.k}]"
    )

    # Show source tracking
    print(f"\nSource document tracking:")
    for i in range(trace.n_docs):
        source_idx = trace.doc_source_ids[i].item()
        doc_slice = trace.doc_slice(i)
        n_tokens = doc_slice.stop - doc_slice.start
        print(f"  Doc {i} -> Original Pile index {source_idx} ({n_tokens} tokens)")

    # Show expert data for first token of first doc
    first_doc_slice = trace.doc_slice(0)
    expert_ids, weights = trace.experts_for_token(0, first_doc_slice.start)
    print(
        f"\nLayer 0, First token of doc 0 (Pile index {trace.doc_source_ids[0].item()}):"
    )
    print(f"  Expert ids: {expert_ids.tolist()}")
    print(f"  Weights: {weights.tolist()}")


if __name__ == "__main__":
    main()
