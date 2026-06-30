#!/usr/bin/env python
"""CLI for Expert Pursuit extraction and pursuit."""

from dotenv import load_dotenv

from moe_interp.capture import capture_expert_activations
from moe_interp.config import (
    get_default_model,
    get_device,
    get_extractions_dir,
    get_pursuit_dir,
    set_seed,
)
from moe_interp.io.data import load_dataset_prompts
from moe_interp.io.plots import plot_count_heatmap, plot_evr_heatmap
from moe_interp.parser import build_parser
from moe_interp.pursuit import run_pursuit


def main():
    load_dotenv()
    set_seed(1337)

    parser = build_parser()
    args = parser.parse_args()

    if args.command == "extract":
        from nnsight import LanguageModel

        # Default device_map: single best device. Pass --device_map auto for
        # pipeline parallelism across multiple GPUs (see --help for caveats).
        model_name = args.model
        device_map = args.device_map or str(get_device())
        model = LanguageModel(
            model_name, device_map=device_map, dtype="auto", dispatch=True
        )  # type: ignore

        max_length = args.max_length or model.config.max_position_embeddings
        prompts = load_dataset_prompts(
            args.dataset,
            model.tokenizer,
            n_docs=args.n_docs,
            max_length=max_length,
        )
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
        model_name = args.model or get_default_model()
        dataset_name = args.dataset

        extractions_dir = get_extractions_dir(model_name, dataset_name)
        output_dir = get_pursuit_dir(model_name, dataset_name, args.concept)

        results, evr_matrix, count_matrix = run_pursuit(
            extractions_dir,
            min_activations=args.min_activations,
            k=args.k,
            output_dir=output_dir,
            concept=args.concept,
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

    elif args.command == "analysis":
        from moe_interp.analysis import run_logit_lens_comparison
        from moe_interp.config import get_model_dir

        model_name = args.model or get_default_model()
        out_dir = get_model_dir(model_name) / "analysis" / args.dataset

        lens = run_logit_lens_comparison(
            model_name,
            args.dataset,
            min_activations=args.min_activations,
            max_rows=args.max_rows,
            extractions_dir=args.extractions_dir,
            pursuit_dir=args.pursuit_dir,
            output_dir=out_dir,
        )["summary"]
        print(
            f"logit-lens vs SOMP: Jaccard@{lens['k']}={lens['mean_jaccard_topk']:.3f}  "
            f"EVR lens(1 dir)={lens['mean_lens_evr']:.4f} somp@10={lens['mean_somp_evr_10']:.4f}"
        )
        print(f"Saved analysis to {out_dir}")


if __name__ == "__main__":
    main()
