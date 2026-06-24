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
        help="Number of documents (default: all docs)",
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
    extract_parser.add_argument(
        "--max_rows_per_expert",
        type=int,
        default=None,
        help="Cap rows kept per (layer, expert); extra rows are dropped once full "
        "(recommended with --token_selection all to bound disk; default: unbounded)",
    )
    extract_parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Max prompt token length (default: model max_position_embeddings). "
        "Lower values (e.g. 256) fit far more documents per GB for all-token capture.",
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
    pursuit_parser.add_argument(
        "--concept",
        type=str,
        default=None,
        help="Optional concept restriction: offensive, countries, numbers",
    )

    # analysis: logit-lens baseline vs SOMP (no model needed)
    analysis_parser = subparsers.add_parser(
        "analysis",
        help="Post-hoc analysis on stored activations: logit-lens baseline vs SOMP",
    )
    analysis_parser.add_argument("--model", type=str, default=None)
    analysis_parser.add_argument(
        "--dataset", type=str, default="pile10k", choices=sorted(DATASET_SPECS)
    )
    analysis_parser.add_argument(
        "--min_activations", type=int, default=50, help="Min rows to analyze an expert"
    )
    analysis_parser.add_argument(
        "--max_rows",
        type=int,
        default=1500,
        help="Per-expert row cap (subsample) for speed",
    )
    analysis_parser.add_argument(
        "--extractions_dir",
        type=str,
        default=None,
        help="Override the activations dir (default: data/<model>/extractions/<dataset>)",
    )
    analysis_parser.add_argument(
        "--pursuit_dir",
        type=str,
        default=None,
        help="Override the SOMP results dir (default: data/<model>/pursuit/<dataset>)",
    )

    # The causal toxic-expert circuit study (DLA classifier, activation patching, gate-AtP,
    # faithfulness, generation-time knockout / project-out, and the HTML report) lives in the
    # `# %%` walkthroughs under notebooks/circuits/, driving moe_interp.circuit directly.

    return parser
