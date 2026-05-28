"""CLI argument parser for Expert Pursuit."""

import argparse

from moe_interp.config import get_default_model
from moe_interp.io.data import DATASET_SPECS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Expert Pursuit CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract", help="Extract expert activations")
    extract_parser.add_argument(
        "--model",
        type=str,
        default=get_default_model(),
        help="Model name or path",
    )
    extract_parser.add_argument(
        "--n_docs",
        type=int,
        default=None,
        help="Number of TriviaQA documents (default: all docs)",
    )
    extract_parser.add_argument(
        "--dataset",
        type=str,
        default="triviaqa",
        choices=sorted(DATASET_SPECS),
        required=False,
        help="Dataset to extract from (default: triviaqa)",
    )
    extract_parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for capture"
    )
    extract_parser.add_argument(
        "--token_selection",
        type=str,
        default="last",
        choices=["last", "all"],
        help="Tokens to store per prompt: last real token or all real tokens",
    )

    pursuit_parser = subparsers.add_parser(
        "pursuit", help="Run projection pursuit analysis"
    )
    pursuit_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (if not specified, reads from metadata.json)",
    )
    pursuit_parser.add_argument(
        "--dataset",
        type=str,
        default="triviaqa",
        choices=sorted(DATASET_SPECS),
        required=False,
        help="Dataset used for the extractions (default: triviaqa)",
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
    concept_group = pursuit_parser.add_mutually_exclusive_group()
    concept_group.add_argument(
        "--concept",
        type=str,
        default=None,
        help="Optional concept restriction: offensive, countries, numbers",
    )
    concept_group.add_argument(
        "--word_top_k",
        type=int,
        default=None,
        nargs="?",
        const=10000,
        help="Use augmented dictionary with top-k common words. "
        "Pass no value for default (10000), or a number. "
        "Mutually exclusive with --concept.",
    )

    return parser
