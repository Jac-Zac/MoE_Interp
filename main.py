#!/usr/bin/env python

from pathlib import Path

import torch
from nnsight import LanguageModel

from src.cache import DocumentTrace, list_all
from src.capture import capture_documents
from src.data import load_pile_docs
from src.display import print_trace, print_trace_summary
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
    data_dir = Path("./data")
    print(f"Saving traces to: {data_dir}")
    print()

    saved_files = capture_documents(
        model,
        docs=docs,
        doc_ids=doc_source_ids,
        store_freq=STORE_FREQ,
        output_dir=data_dir,
    )

    print(f"\n\nCaptured {len(saved_files)} documents total")
    print_trace_summary(data_dir)

    # Load first document and show structure
    if saved_files:
        first_doc_id = list_all(data_dir)[0]
        print(f"\nLoading document {first_doc_id}...")
        trace = DocumentTrace.load(first_doc_id, data_dir)

        print(trace)
        print()
        print_trace(trace)

        # Show first expert activation from layer 0
        layer_0_acts = trace.expert_traces[0]
        if layer_0_acts:
            first_expert_id = sorted(layer_0_acts.keys())[0]
            expert_trace = layer_0_acts[first_expert_id]
            print(
                f"\nLayer 0, Expert {first_expert_id}: "
                f"tokens={len(expert_trace.token_indices)}, down_proj_shape={tuple(expert_trace.raw_outputs.shape)}"
            )


if __name__ == "__main__":
    main()
