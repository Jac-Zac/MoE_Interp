#!/usr/bin/env python
"""Run the full causal circuit pipeline: patching, gate-AtP, comparison, steer, report.

Builds a disjoint RealToxicityPrompts train/test split and threads it through every
model-dependent step so all methods share the same identification prompts and are scored
on the same held-out test set. All artifacts land under ``data/<model>/circuit/``.

Usage:
    python scripts/cineca/circuit_runner.py [--model MODEL] [--batch-size N]
                                            [--atp-batch-size N] [--knockout-k N]
                                            [--steer-layer L] [--max-new-tokens N]
                                            [--n-prompts N] [--n-test N]
"""

import argparse
import json
import os

import numpy as np
import torch
from dotenv import load_dotenv
from nnsight import LanguageModel

from moe_interp.circuit.attribution import gate_attribution
from moe_interp.circuit.compare import faithfulness, plot_faithfulness
from moe_interp.circuit.patching import (
    expert_patching_grid,
    plot_expert_effect_grid,
    top_grid_experts,
)
from moe_interp.circuit.prompts import rtp_split
from moe_interp.circuit.report import build_report
from moe_interp.circuit.steer import run_steer
from moe_interp.config import get_default_model, get_device, get_model_dir, set_seed
from moe_interp.pursuit.concepts import build_toxic_token_ids


def main():
    load_dotenv()
    set_seed(1337)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=get_default_model())
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument(
        "--atp-batch-size",
        type=int,
        default=None,
        help="Batch size for the AtP backward pass. Defaults to --batch-size. "
        "Use a smaller value (e.g. 2) if the backward pass OOMs after the grid sweep.",
    )
    parser.add_argument("--knockout-k", type=int, default=15)
    parser.add_argument("--steer-layer", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--n-prompts", type=int, default=48,
                        help="Train-set size (used for grid identification and steer training).")
    parser.add_argument("--n-test", type=int, default=24,
                        help="Held-out test set size (scored by steer experiment).")
    args = parser.parse_args()
    atp_batch = args.atp_batch_size if args.atp_batch_size is not None else args.batch_size

    model_name = args.model
    cdir = get_model_dir(model_name) / "circuit"
    cdir.mkdir(parents=True, exist_ok=True)
    print(f"Circuit artifacts -> {cdir}")

    device_map = os.environ.get("DEVICE_MAP", str(get_device()))
    model = LanguageModel(
        model_name, device_map=device_map, dtype="auto", dispatch=True
    )

    # Disjoint train/test split — all methods share elic_tr for identification,
    # all are scored on the disjoint elic_te so comparisons are out-of-sample.
    elic_tr, elic_te, neut_tr, neut_te = rtp_split(
        model.tokenizer, n_train=args.n_prompts, n_test=args.n_test
    )
    toxic_ids = build_toxic_token_ids(model.tokenizer)
    print(
        f"Model loaded: {len(elic_tr)} train + {len(elic_te)} test eliciting prompts",
        flush=True,
    )

    # 1. Causal patching grid (one forward per routed expert) — the ground truth.
    patch_dir = cdir / "patching"
    patch_dir.mkdir(parents=True, exist_ok=True)
    grid_path = patch_dir / "patching_grid.npy"
    if not grid_path.exists():
        print("[1/4] Causal patching grid (this will take a while) ...", flush=True)
        grid = expert_patching_grid(
            model, elic_tr, toxic_ids, batch_size=args.batch_size
        )
        np.save(grid_path, grid.numpy())
        top = top_grid_experts(grid)
        (patch_dir / "top_experts.json").write_text(json.dumps(top, indent=2))
        plot_expert_effect_grid(
            grid,
            patch_dir / "patching_grid.html",
            title=f"Expert ablation effect — {model_name}",
        )
        print(f"  patching grid -> {patch_dir}", flush=True)
    else:
        print("[1/4] Patching grid already exists, skipping", flush=True)

    # 2. gate-AtP (one backward pass) — the cheap estimate of the patching grid.
    # Keyed by train-set size so step 4's offensive knockout reuses this exact grid
    # (same prompts, same toxic ids) instead of paying for a second backward pass.
    atp_path = cdir / "attribution" / f"atp_grid_n{len(elic_tr)}.npy"
    if not atp_path.exists():
        print("[2/4] gate-AtP ...", flush=True)
        # Flush VRAM fragmentation from the patching sweep before the backward pass.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        atp = gate_attribution(model, elic_tr, toxic_ids, batch_size=atp_batch)
        atp_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(atp_path, atp.numpy())
    else:
        print("[2/4] gate-AtP already exists, skipping", flush=True)

    # 3. Faithfulness comparison (gate-AtP vs the causal patching grid; model-free).
    cmp_dir = cdir / "compare"
    if not (cmp_dir / "faithfulness.json").exists():
        print("[3/4] Faithfulness ...", flush=True)
        patching = torch.from_numpy(np.load(grid_path)).float()
        grids = {"gate-AtP": torch.from_numpy(np.load(atp_path)).float()}
        scores = faithfulness(grids, patching)
        cmp_dir.mkdir(parents=True, exist_ok=True)
        (cmp_dir / "faithfulness.json").write_text(json.dumps(scores, indent=2))
        plot_faithfulness(
            scores,
            cmp_dir / "faithfulness.html",
            title=f"Attributor faithfulness — {model_name}",
        )
    else:
        print("[3/4] Faithfulness already exists, skipping", flush=True)

    # 4. Knockout / project-out / steer intervention experiment.
    steer_out = cdir / "steer" / "offensive"
    if not (steer_out / "intervention.json").exists():
        print("[4/4] Intervention experiment ...", flush=True)
        res = run_steer(
            model,
            model_name,
            concept="offensive",
            knockout_k=args.knockout_k,
            steer_layer=args.steer_layer,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            train=(elic_tr, neut_tr),
            test=(elic_te, neut_te),
        )
        steer_out.mkdir(parents=True, exist_ok=True)
        (steer_out / "intervention.json").write_text(json.dumps(res, indent=2))
    else:
        print("[4/4] Intervention already exists, skipping", flush=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Building report ...", flush=True)
    report_path = build_report(model_name)
    print(f"Report -> {report_path}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
