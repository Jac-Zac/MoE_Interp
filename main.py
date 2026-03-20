#!/usr/bin/env python
"""CLI for Expert Pursuit extraction and pursuit."""

import argparse

from src.capture import capture_expert_activations
from src.data import DATASET_SPECS, load_dataset_prompts
from src.environment import (
    get_data_dir,
    get_extractions_dir,
    get_pursuit_dir,
    get_unembedding_dir,
    load_env,
    set_seed,
)
from src.plots import plot_count_heatmap, plot_evr_heatmap
from src.pursuit import run_pursuit


def main():
    load_env()
    set_seed(1337)

    parser = argparse.ArgumentParser(description="Expert Pursuit CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract", help="Extract expert activations")
    extract_parser.add_argument(
        "--model",
        type=str,
        default="allenai/OLMoE-1B-7B-0924-Instruct",
        help="Model name or path",
    )
    extract_parser.add_argument(
        "--n_docs", type=int, default=5000, help="Number of TriviaQA documents"
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
    pursuit_parser.add_argument(
        "--word_top_k",
        type=int,
        default=None,
        nargs="?",
        const=10000,
        help="Use augmented dictionary with top-k common words. "
        "Pass no value for default (10000), or a number. "
        "Mutually exclusive with --concept.",
    )

    args = parser.parse_args()

    if args.command == "extract":
        import torch.distributed as dist
        from nnsight import LanguageModel

        model_kwargs = dict(dtype="auto", dispatch=True)
        if dist.is_initialized() and dist.get_world_size() > 1:
            model_kwargs["tp_plan"] = "auto"
        else:
            model_kwargs["device_map"] = "auto"

        model_name = args.model
        model = LanguageModel(model_name, **model_kwargs)  # type: ignore[arg-type, call-arg]
        tokenizer = model.tokenizer

        prompts = load_dataset_prompts(args.dataset, tokenizer, n_docs=args.n_docs)
        print(f"Loaded {len(prompts)} {args.dataset} prompts")

        output_dir = get_extractions_dir(model_name, args.dataset)
        capture_expert_activations(
            model,
            prompts,
            output_dir,
            model_name=model_name,
            dataset_name=args.dataset,
            batch_size=args.batch_size,
        )

    elif args.command == "pursuit":
        if args.concept and args.word_top_k:
            parser.error("--concept and --word_top_k are mutually exclusive")

        from transformers import AutoTokenizer

        from src.cache import load_metadata, load_unembedding
        from src.word_dictionary import build_word_dictionary

        data_dir = get_data_dir()

        model_name = args.model or "allenai/OLMoE-1B-7B-0924-Instruct"
        dataset_name = args.dataset

        extractions_dir = get_extractions_dir(model_name, dataset_name)

        word_dictionary = None
        if args.word_top_k:
            output_dir = get_pursuit_dir(model_name, dataset_name, "words")
            metadata = load_metadata(extractions_dir / "metadata.json")
            tokenizer = AutoTokenizer.from_pretrained(metadata["model_name"])
            base_dictionary = load_unembedding(
                get_unembedding_dir(model_name) / "dictionary.h5"
            ).float()
            word_dictionary = build_word_dictionary(
                tokenizer, base_dictionary, top_k=args.word_top_k
            )
        else:
            output_dir = get_pursuit_dir(model_name, dataset_name, args.concept)

        results, evr_matrix, count_matrix = run_pursuit(
            extractions_dir,
            min_activations=args.min_activations,
            k=args.k,
            output_dir=output_dir,
            data_dir=data_dir,
            concept=args.concept,
            word_dictionary=word_dictionary,
        )
        plot_evr_heatmap(
            evr_matrix,
            output_path=output_dir / "evr_heatmap.html",
        )
        plot_count_heatmap(
            count_matrix,
            output_path=output_dir / "count_heatmap.html",
        )
        print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
