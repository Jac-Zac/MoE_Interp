"""CLI argument parser for Expert Pursuit."""

import argparse

from moe_interp.config import get_default_model
from moe_interp.io.data import DATASET_SPECS
from moe_interp.pursuit.concepts import CONCEPT_WORDS


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

    # toxic-dla: gradient-free toxic-expert classifier from stored activations (no model)
    dla_parser = subparsers.add_parser(
        "toxic-dla",
        help="Direct Logit Attribution: which experts write toward toxic vocab (no model)",
    )
    dla_parser.add_argument("--model", type=str, default=None)
    dla_parser.add_argument(
        "--dataset", type=str, default="pile10k", choices=sorted(DATASET_SPECS),
        help="All-token extraction to score over (default: pile10k; rtp is too sparse)",
    )
    dla_parser.add_argument(
        "--min_activations", type=int, default=50, help="Min rows to score an expert"
    )
    dla_parser.add_argument(
        "--max_rows", type=int, default=2000, help="Per-expert row cap (subsample)"
    )

    # circuit: causal expert activation-patching grid (loads the model via nnsight)
    circuit_parser = subparsers.add_parser(
        "circuit",
        help="Causal toxic-expert study: per-(layer,expert) ablation-patching effect grid",
    )
    circuit_parser.add_argument("--model", type=str, default=None)
    circuit_parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Restrict the grid to these layers (default: all)",
    )
    circuit_parser.add_argument("--batch_size", type=int, default=6)
    circuit_parser.add_argument(
        "--n_prompts",
        type=int,
        default=None,
        help="Cap the number of toxic seed prompts (default: all built-in seeds)",
    )

    # circuit-compare: faithfulness of cheap attributors vs the causal patching grid
    cmp_parser = subparsers.add_parser(
        "circuit-compare",
        help="Score gate-AtP (+ DLA) against the causal patching grid (needs `circuit` first)",
    )
    cmp_parser.add_argument("--model", type=str, default=None)
    cmp_parser.add_argument("--batch_size", type=int, default=8)

    # circuit-steer: generation-time interventions (knockout / steer) vs baseline
    steer_parser = subparsers.add_parser(
        "circuit-steer",
        help="Suppress toxic generation by knocking out top gate-AtP experts / steering",
    )
    steer_parser.add_argument("--model", type=str, default=None)
    steer_parser.add_argument("--batch_size", type=int, default=8)
    steer_parser.add_argument(
        "--concept", type=str, default="offensive", choices=sorted(CONCEPT_WORDS),
        help="Concept to suppress (default: offensive). Non-toxicity concepts use the "
        "unembedding direction + project-out; the expert-knockout comparison runs only for "
        "'offensive' (the seed prompts only elicit toxicity).",
    )
    steer_parser.add_argument(
        "--knockout_k", type=int, default=15, help="How many top gate-AtP experts to knock out"
    )
    steer_parser.add_argument("--steer_layer", type=int, default=12)
    steer_parser.add_argument("--max_new_tokens", type=int, default=24)

    # circuit-report: assemble all circuit artifacts into one HTML report (no model)
    report_parser = subparsers.add_parser(
        "circuit-report", help="Build the self-contained toxic-circuit HTML report"
    )
    report_parser.add_argument("--model", type=str, default=None)

    return parser
