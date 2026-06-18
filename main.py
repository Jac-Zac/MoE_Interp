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
            f"EVR@10 lens={lens['mean_lens_evr_10']:.4f} somp={lens['mean_somp_evr_10']:.4f}"
        )
        print(f"Saved analysis to {out_dir}")

    elif args.command == "circuit":
        import json

        import numpy as np
        from nnsight import LanguageModel

        from moe_interp.circuit.patching import (
            expert_patching_grid,
            plot_expert_effect_grid,
            top_grid_experts,
        )
        from moe_interp.circuit.pipeline import default_prompts
        from moe_interp.circuit.toxicity import build_toxic_token_ids
        from moe_interp.config import get_model_dir

        model_name = args.model or get_default_model()
        model = LanguageModel(
            model_name, device_map=str(get_device()), dtype="auto", dispatch=True
        )
        toxic_prompts, _ = default_prompts(model.tokenizer)
        if args.n_prompts:
            toxic_prompts = toxic_prompts[: args.n_prompts]
        toxic_ids = build_toxic_token_ids(model.tokenizer)

        grid = expert_patching_grid(
            model, toxic_prompts, toxic_ids,
            batch_size=args.batch_size, layers=args.layers,
        )
        out_dir = get_model_dir(model_name) / "circuit" / "patching"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "patching_grid.npy", grid.numpy())
        plot_expert_effect_grid(
            grid, out_dir / "patching_grid.html",
            title=f"Expert ablation effect on toxic-logit — {model_name}",
        )
        top = top_grid_experts(grid)
        (out_dir / "top_experts.json").write_text(json.dumps(top, indent=2))
        print("top causal toxic experts (by |ablation effect|):")
        for r in top[:10]:
            print(f"  L{r['layer']}E{r['expert']}  effect={r['effect']:+.4f}")
        print(f"Saved patching grid + heatmap to {out_dir}")

    elif args.command == "circuit-compare":
        import json

        import numpy as np
        import torch
        from nnsight import LanguageModel

        from moe_interp.capture.cache import load_unembedding
        from moe_interp.circuit.compare import (
            faithfulness,
            method_grids,
            plot_faithfulness,
        )
        from moe_interp.circuit.pipeline import default_prompts
        from moe_interp.circuit.toxicity import build_toxic_token_ids
        from moe_interp.config import get_model_dir, get_unembedding_dir

        model_name = args.model or get_default_model()
        md = get_model_dir(model_name)
        grid_path = md / "circuit" / "patching" / "patching_grid.npy"
        if not grid_path.exists():
            raise FileNotFoundError(
                f"No patching grid at {grid_path}. Run `python main.py circuit` first."
            )
        patching = torch.from_numpy(np.load(grid_path)).float()
        unembedding = load_unembedding(
            get_unembedding_dir(model_name) / "dictionary.h5"
        ).float()

        model = LanguageModel(
            model_name, device_map=str(get_device()), dtype="auto", dispatch=True
        )
        toxic, neutral = default_prompts(model.tokenizer)
        toxic_ids = build_toxic_token_ids(model.tokenizer)
        grids = method_grids(
            model, toxic, neutral, toxic_ids, unembedding,
            batch_size=args.batch_size, layers=args.layers,
        )
        scores = faithfulness(grids, patching)

        out_dir = md / "circuit" / "compare"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "faithfulness.json").write_text(json.dumps(scores, indent=2))
        plot_faithfulness(
            scores, out_dir / "faithfulness.html",
            title=f"Attributor faithfulness vs causal patching — {model_name}",
        )
        print("faithfulness vs causal patching grid (pooled r):")
        for name, s in scores.items():
            print(f"  {name:18s} r = {s['pooled_r']:+.3f}")
        print(f"Saved comparison to {out_dir}")


if __name__ == "__main__":
    main()
