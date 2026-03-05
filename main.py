#!/usr/bin/env python
"""CLI for Expert Pursuit encoding and pursuit."""

import argparse

import torch

from src.capture import capture_expert_activations
from src.data import load_triviaqa
from src.environment import get_data_dir, load_env, set_seed
from src.pursuit import run_pursuit


def main():
    load_env()
    set_seed(1337)

    parser = argparse.ArgumentParser(description="Expert Pursuit CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    encode_parser = subparsers.add_parser("encode", help="Encode expert activations")
    encode_parser.add_argument(
        "--model",
        type=str,
        default="allenai/OLMoE-1B-7B-0924-Instruct",
        help="Model name or path",
    )
    encode_parser.add_argument(
        "--n_docs", type=int, default=5000, help="Number of TriviaQA documents"
    )
    encode_parser.add_argument("--batch_size", type=int, default=8, help="Batch size")

    pursuit_parser = subparsers.add_parser(
        "pursuit", help="Run projection pursuit analysis"
    )
    pursuit_parser.add_argument(
        "--k", type=int, default=50, help="Top-k tokens per expert"
    )
    pursuit_parser.add_argument(
        "--min_activations",
        type=int,
        default=5,
        help="Minimum activations to analyze expert",
    )

    args = parser.parse_args()

    if args.command == "encode":
        from nnsight import LanguageModel

        model = LanguageModel(
            args.model,
            device_map="auto",
            dtype=torch.float16,
            dispatch=True,
        )
        tokenizer = model.tokenizer

        prompts = load_triviaqa(tokenizer, n_docs=args.n_docs)
        print(f"Loaded {len(prompts)} TriviaQA prompts")

        data_dir = get_data_dir()
        output_dir = data_dir / "encodings"
        capture_expert_activations(
            model, prompts, args.batch_size, output_dir, data_dir, args.model
        )

    elif args.command == "pursuit":
        data_dir = get_data_dir()
        encodings_dir = data_dir / "encodings"
        output_dir = data_dir / "pursuit"

        run_pursuit(
            encodings_dir,
            min_activations=args.min_activations,
            k=args.k,
            output_dir=output_dir,
            data_dir=data_dir,
        )


if __name__ == "__main__":
    main()
