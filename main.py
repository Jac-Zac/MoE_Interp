#!/usr/bin/env python
"""Expert Pursuit pipeline: encode, pursuit (with plots)."""

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

    unembed = model.lm_head.weight.detach().float().cpu()
    unembed_path = save_unembedding(output_dir, unembed)
    print(f"Saved unembedding to {unembed_path}")

    encode_dataset(
        model,
        prompts=prompts,
        output_dir=output_dir / "activations",
        batch_size=args.batch_size,
    )
    print(f"\nEncoding complete: {output_dir}")


def cmd_pursuit(args: argparse.Namespace) -> None:
    """Run pursuit on cached activations, save results + plots."""
    from transformers import AutoTokenizer

    from src.cache import load_unembedding
    from src.plot import generate_all_plots
    from src.pursuit import run_expert_pursuit

    encodings_dir = DATA_DIR / "encodings"
    activations_dir = encodings_dir / "activations"

    unembed = load_unembedding(encodings_dir)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"Loaded unembedding {tuple(unembed.shape)}")

    result = run_expert_pursuit(
        activations_dir=activations_dir,
        unembed=unembed,
        tokenizer=tokenizer,
        k=args.k,
    )

    out_dir = DATA_DIR / "pursuit"
    result.save(out_dir)
    print(f"\nPursuit complete: {out_dir}")
    print(f"  {len(result.experts)} experts analyzed")

    # Print top 5 concepts per expert for first few
    print("\nSample expert concepts:")
    for e in result.experts[:5]:
        print(f"  L{e.layer}E{e.expert_id}: {e.tokens[:5]}")

    # Generate plots
    plots_dir = DATA_DIR / "plots"
    paths = generate_all_plots(result, plots_dir)
    print(f"\nPlots saved to {plots_dir}:")
    for p in paths:
        print(f"  {p.name}")


def main():
    parser = argparse.ArgumentParser(description="Expert Pursuit pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    enc = subparsers.add_parser("encode", help="Encode dataset to per-layer HDF5")
    enc.add_argument("--n_docs", type=int, default=5000)
    enc.add_argument("--split", type=str, default="train")
    enc.add_argument("--batch_size", type=int, default=8)

    pur = subparsers.add_parser("pursuit", help="Run pursuit and generate plots")
    pur.add_argument("--k", type=int, default=50, help="Top atoms per expert")

    args = parser.parse_args()
    set_seed(SEED)

    commands = {"encode": cmd_encode, "pursuit": cmd_pursuit}
    commands[args.command](args)


if __name__ == "__main__":
    main()
