#!/usr/bin/env python
"""CLI for Expert Pursuit extraction and pursuit."""

from dotenv import load_dotenv

from moe_interp.capture import capture_expert_activations
from moe_interp.config import (
    get_default_model,
    get_device,
    get_extractions_dir,
    get_pursuit_dir,
    get_unembedding_dir,
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
        import torch.distributed as dist
        from nnsight import LanguageModel

        model_kwargs = dict(dtype="auto", dispatch=True)
        if dist.is_initialized() and dist.get_world_size() > 1:
            model_kwargs["tp_plan"] = "auto"
        else:
            # Pin the whole model to the single best device. `device_map="auto"`
            # probes free memory at load time and can flakily offload a sliver of an
            # MoE model to disk (which then fails: MoE weights can't be re-saved
            # without an offload_folder). Forcing the device avoids that when it fits.
            model_kwargs["device_map"] = str(get_device())

        model_name = args.model
        model = LanguageModel(model_name, **model_kwargs)  # type: ignore

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
            token_selection=args.token_selection,
            max_rows_per_expert=args.max_rows_per_expert,
        )

    elif args.command == "pursuit":
        from transformers import AutoTokenizer

        from moe_interp.capture.cache import load_metadata, load_unembedding
        from moe_interp.pursuit.dictionary import build_word_dictionary

        model_name = args.model or get_default_model()
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

    elif args.command == "analysis":
        from moe_interp.analysis.pipeline import run_analysis
        from moe_interp.analysis.report import write_report
        from moe_interp.config import get_analysis_dir

        model_name = args.model or get_default_model()
        dataset_name = args.dataset

        extractions_dir = get_extractions_dir(model_name, dataset_name)
        output_dir = get_analysis_dir(model_name, dataset_name)

        run_analysis(
            extractions_dir,
            output_dir,
            model_name=model_name,
            dataset=dataset_name,
            min_activations=args.min_activations,
            methods=tuple(m.strip() for m in args.methods.split(",") if m.strip()),
            pursuit_dir=args.pursuit_dir,
            top_k=args.top_k,
            logit_lens=not args.skip_logit_lens,
            adp=args.adp,
            adp_min_rows=args.adp_min_rows,
            n_bootstrap=args.n_bootstrap,
        )
        if args.report:
            report_path = write_report(output_dir, model_name, dataset_name)
            print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
