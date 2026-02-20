#!/usr/bin/env python
"""Expert Pursuit pipeline: encode TriviaQA expert activations to HDF5."""

import argparse

import torch

from src.data import load_triviaqa
from src.environment import get_data_dir, load_model, set_seed

SEED = 1337
DATA_DIR = get_data_dir()
MODEL_NAME = "allenai/OLMoE-1B-7B-0924-Instruct"


def cmd_encode(args: argparse.Namespace) -> None:
    """Encode documents: capture gated expert outputs to HDF5."""
    from src.capture import encode_dataset

    model = load_model(MODEL_NAME)

    questions = load_triviaqa(
        tokenizer=model.tokenizer,
        n_docs=args.n_docs,
        split=args.split,
    )

    print(f"Loaded {len(questions)} questions")

    output_dir = DATA_DIR / "encodings"

    # Save unembedding matrix for later SOMP analysis
    unembed_path = output_dir / "unembedding.pt"
    unembed_path.parent.mkdir(parents=True, exist_ok=True)
    unembed = model.lm_head.weight.detach().float().cpu()
    torch.save(unembed, unembed_path)
    print(f"Saved unembedding matrix to {unembed_path}")

    # Encode dataset to HDF5
    encode_dataset(
        model,
        questions=questions,
        output_dir=output_dir / "activations",
    )
    print(f"\nEncoding complete: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Expert Pursuit pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Encode command
    enc = subparsers.add_parser("encode", help="Encode dataset to HDF5")
    enc.add_argument("--n_docs", type=int, default=5000)
    enc.add_argument("--split", type=str, default="train")

    args = parser.parse_args()
    set_seed(SEED)

    if args.command == "encode":
        cmd_encode(args)


if __name__ == "__main__":
    main()
