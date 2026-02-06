#!/usr/bin/env python

import torch
from nnsight import LanguageModel

from src.capture import capture_moe_activations
from src.checkpoint import get_data_dir, list_documents, load_document
from src.data import load_pile_docs
from src.environment import set_seed

# Configuration
SEED = 1337
N_DOCS = 10
STORE_FREQ = 5  # Save every 5 documents
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
    print(f"Saving traces to: {data_dir}")
    print()

    saved_files = capture_moe_activations(
        model,
        docs=docs,
        doc_ids=doc_source_ids,
        store_freq=STORE_FREQ,
        output_dir=data_dir,
    )

    print(f"\n\nCaptured {len(saved_files)} documents total")
    print("\nAvailable documents:")
    for doc_id in list_documents(data_dir):
        print(f"  Doc {doc_id}")

    # Load first document and show structure
    if saved_files:
        first_doc_id = list_documents(data_dir)[0]
        print(f"\n\nLoading document {first_doc_id} from disk...")
        trace = load_document(first_doc_id, data_dir)

        print(f"\nDocument {first_doc_id} info:")
        print(f"  Shape: [layers={trace.n_layers}, seq={trace.seq_len}, k={trace.k}]")

        # Show expert data for first token
        expert_ids, weights = trace.get_token(0, 0)
        print(f"\nLayer 0, First token:")
        print(f"  Expert ids: {expert_ids.tolist()}")
        print(f"  Weights: {weights.tolist()}")


if __name__ == "__main__":
    main()
