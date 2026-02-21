#!/usr/bin/env python
"""Expert Pursuit pipeline: encode, pursuit, plot."""

import argparse

from src.environment import get_data_dir, set_seed

SEED = 1337
DATA_DIR = get_data_dir()
MODEL_NAME = "allenai/OLMoE-1B-7B-0924-Instruct"


def cmd_encode(args: argparse.Namespace) -> None:
    """Encode documents: capture last-token expert outputs to per-layer HDF5."""
    from src.cache import save_unembedding
    from src.capture import encode_dataset
    from src.data import load_triviaqa
    from src.environment import load_model

    model = load_model(MODEL_NAME)

    prompts = load_triviaqa(
        tokenizer=model.tokenizer,
        n_docs=args.n_docs,
        split=args.split,
    )
    print(f"Loaded {len(prompts)} prompts")

    output_dir = DATA_DIR / "encodings"

    # Save unembedding matrix for later SOMP
    unembed = model.lm_head.weight.detach().float().cpu()
    unembed_path = save_unembedding(output_dir, unembed)
    print(f"Saved unembedding to {unembed_path}")

    # Encode dataset to per-layer HDF5
    encode_dataset(
        model,
        prompts=prompts,
        output_dir=output_dir / "activations",
        batch_size=args.batch_size,
    )
    print(f"\nEncoding complete: {output_dir}")


def cmd_pursuit(args: argparse.Namespace) -> None:
    """Run SOMP on cached activations to find expert specializations."""
    from transformers import AutoTokenizer

    from src.cache import load_unembedding
    from src.pursuit import build_filtered_dictionary, run_expert_pursuit

    encodings_dir = DATA_DIR / "encodings"
    activations_dir = encodings_dir / "activations"

    # Load unembedding + tokenizer (no model needed)
    unembed = load_unembedding(encodings_dir)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"Loaded unembedding {tuple(unembed.shape)} and tokenizer")

    # Build filtered dictionary
    dictionary, tokens_data = build_filtered_dictionary(
        unembed,
        tokenizer,
        args.property,
    )
    print(f"Dictionary: {args.property} -> {len(tokens_data)} tokens")

    # Run SOMP on per-layer HDF5
    result = run_expert_pursuit(
        activations_dir=activations_dir,
        dictionary=dictionary,
        tokens_data=tokens_data,
        tokenizer=tokenizer,
        k=args.k,
        property_name=args.property,
    )

    # Save results
    out_dir = DATA_DIR / "pursuit" / args.property
    result.save(out_dir)
    print(f"\nPursuit complete: {out_dir}")
    print(f"  {len(result.experts)} active experts analyzed")

    # Print top concepts
    freq = result.concept_frequency(top_n=5)
    print(f"\nTop 10 concepts ({args.property}):")
    for word, count in freq.most_common(10):
        print(f"  {word}: {count}")


def cmd_plot(args: argparse.Namespace) -> None:
    """Generate plots from saved pursuit results."""
    from src.plot import generate_all_plots
    from src.pursuit import PursuitResult

    results_dir = DATA_DIR / "pursuit" / args.property
    if not (results_dir / "pursuit_results.json").exists():
        print(f"No results found at {results_dir}")
        print("Run 'pursuit' first: python main.py pursuit --property " + args.property)
        return

    result = PursuitResult.load(results_dir)
    print(
        f"Loaded results: {len(result.experts)} experts, property={result.property_name}"
    )

    out_dir = DATA_DIR / "plots" / args.property
    paths = generate_all_plots(result, out_dir)

    print(f"\nPlots saved to {out_dir}:")
    for p in paths:
        print(f"  {p.name}")


def main():
    parser = argparse.ArgumentParser(description="Expert Pursuit pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # encode
    enc = subparsers.add_parser("encode", help="Encode dataset to per-layer HDF5")
    enc.add_argument("--n_docs", type=int, default=5000)
    enc.add_argument("--split", type=str, default="train")
    enc.add_argument("--batch_size", type=int, default=8)

    # pursuit
    pur = subparsers.add_parser("pursuit", help="Run SOMP on cached activations")
    pur.add_argument(
        "--property",
        type=str,
        default="countries",
        help="Word list: countries, colors, quantity, or 'all'",
    )
    pur.add_argument("--k", type=int, default=50, help="SOMP atoms per expert")

    # plot
    plo = subparsers.add_parser("plot", help="Generate plots from pursuit results")
    plo.add_argument(
        "--property",
        type=str,
        default="countries",
        help="Which pursuit results to plot",
    )

    args = parser.parse_args()
    set_seed(SEED)

    commands = {"encode": cmd_encode, "pursuit": cmd_pursuit, "plot": cmd_plot}
    commands[args.command](args)


if __name__ == "__main__":
    main()
