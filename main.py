#!/usr/bin/env python
"""Expert Pursuit pipeline entry point.

Usage:
    python main.py encode --n_docs 100 --max_tokens 512
    python main.py pursuit --concept countries --k 50
"""

import argparse

from src.data import load_pile_docs
from src.environment import get_data_dir, load_model, set_seed

SEED = 1337
DATA_DIR = get_data_dir()
MODEL_NAME = "allenai/OLMoE-1B-7B-0924-Instruct"


def cmd_encode(args: argparse.Namespace) -> None:
    """Encode documents: capture gated expert outputs to HDF5."""
    from src.capture import encode_dataset
    from src.dictionary import extract_unembedding

    model = load_model(MODEL_NAME)

    docs, doc_ids = load_pile_docs(
        tokenizer=model.tokenizer,
        n_docs=args.n_docs,
        max_tokens=args.max_tokens,
        truncate=args.truncate,
    )

    print(f"Loaded {len(docs)} documents")

    output_dir = DATA_DIR / "encodings"

    # Extract and save unembedding matrix (needed for SOMP later)
    extract_unembedding(model, save_dir=output_dir)
    print("Saved unembedding matrix")

    # Encode dataset to HDF5
    encode_dataset(
        model,
        docs=docs,
        doc_ids=doc_ids,
        output_dir=output_dir / "activations",
    )
    print(f"\nEncoding complete: {output_dir}")


def cmd_pursuit(args: argparse.Namespace) -> None:
    """Run SOMP analysis on encoded expert activations."""
    from src.pursuit import expert_pursuit, print_top_experts, save_pursuit_results

    model = load_model(MODEL_NAME)

    data_dir = DATA_DIR / "encodings" / "activations"
    unembed_path = DATA_DIR / "encodings" / "unembedding.h5"

    results = expert_pursuit(
        data_dir=data_dir,
        concept=args.concept,
        tokenizer=model.tokenizer,
        unembed_path=unembed_path,
        k=args.k,
    )

    output_dir = DATA_DIR / "somp" / args.concept
    save_pursuit_results(results, model.tokenizer, output_dir)

    print(f"\nTop experts for concept '{args.concept}':")
    print_top_experts(results["evr"])
    print(f"\nResults saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Expert Pursuit pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Encode command
    enc = subparsers.add_parser("encode", help="Encode dataset to HDF5")
    enc.add_argument("--n_docs", type=int, default=100)
    enc.add_argument("--max_tokens", type=int, default=512)

    enc.add_argument(
        "--truncate", action="store_true", help="Truncate long docs instead of skipping"
    )

    # Pursuit command
    pur = subparsers.add_parser("pursuit", help="Run SOMP analysis")
    pur.add_argument(
        "--concept",
        type=str,
        required=True,
        help="Concept name (e.g., countries, colors) or 'full'",
    )
    pur.add_argument("--k", type=int, default=50, help="Number of SOMP iterations")

    args = parser.parse_args()
    set_seed(SEED)

    if args.command == "encode":
        cmd_encode(args)
    elif args.command == "pursuit":
        cmd_pursuit(args)


if __name__ == "__main__":
    main()
