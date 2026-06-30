#!/usr/bin/env python
"""Run the gate-AtP localization step + render the localization report.

Builds the disjoint RealToxicityPrompts train/test split and computes the gate-AtP causal-effect
grid (one backward pass) over the eliciting *train* prompts, then renders the HTML report
(gate-AtP heatmap + activation-patching faithfulness). Artifacts land under ``data/<model>/circuit/``.

The grid is keyed by train-set size (``atp_grid_n<N>.npy``) and is *shared* with the
knockout/downweighting sweep (``scripts/cineca/downweight_runner.py``), which is where the
intervention results are produced. See ``moe_interp.circuit.attribution`` for the AtP method and
its one-off validation against exhaustive activation patching.

Usage:
    python scripts/cineca/circuit_runner.py [--model MODEL] [--batch-size N]
                                            [--atp-batch-size N] [--n-prompts N] [--n-test N]
"""

import argparse
import os

import numpy as np
import torch
from dotenv import load_dotenv
from nnsight import LanguageModel

from moe_interp.circuit.attribution import gate_attribution
from moe_interp.circuit.prompts import rtp_split
from moe_interp.circuit.report import build_report
from moe_interp.config import get_default_model, get_device, get_model_dir, set_seed
from moe_interp.pursuit.concepts import build_concept_token_ids


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
        "Use a smaller value (e.g. 2) if the backward pass OOMs.",
    )
    parser.add_argument(
        "--n-prompts",
        type=int,
        default=100,
        help="Train-set size (used for grid identification; keys the grid file).",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=64,
        help="Held-out test-set size (kept for a split shared with the downweight sweep).",
    )
    parser.add_argument(
        "--hi",
        type=float,
        default=0.5,
        help="Min RealToxicityPrompts toxicity for the eliciting set.",
    )
    parser.add_argument(
        "--challenging",
        action="store_true",
        help="Restrict the eliciting set to RTP's curated 'challenging' subset (more toxic).",
    )
    args = parser.parse_args()
    atp_batch = (
        args.atp_batch_size if args.atp_batch_size is not None else args.batch_size
    )

    model_name = args.model
    cdir = get_model_dir(model_name) / "circuit"
    cdir.mkdir(parents=True, exist_ok=True)
    print(f"Circuit artifacts -> {cdir}")

    device_map = os.environ.get("DEVICE_MAP", str(get_device()))
    model = LanguageModel(
        model_name, device_map=device_map, dtype="auto", dispatch=True
    )

    # Train split identifies the experts; keyed by train size so the downweight sweep reuses this
    # exact grid (same prompts, same toxic ids). The test split is computed for a shared split but
    # not used here — the intervention scoring lives in the downweight sweep.
    elic_tr, _, _, _ = rtp_split(
        model.tokenizer,
        n_train=args.n_prompts,
        n_test=args.n_test,
        hi=args.hi,
        challenging=args.challenging,
    )
    regime = (
        ""
        if (args.hi == 0.5 and not args.challenging)
        else (f"_hi{args.hi:g}" + ("_chal" if args.challenging else ""))
    )
    toxic_ids = build_concept_token_ids(model.tokenizer)
    print(f"Model loaded: {len(elic_tr)} train eliciting prompts", flush=True)

    # gate-AtP localization grid (one backward pass) — the causal localizer.
    atp_path = cdir / "attribution" / f"atp_grid_n{len(elic_tr)}{regime}.npy"
    if not atp_path.exists():
        print("gate-AtP localization grid ...", flush=True)
        atp = gate_attribution(model, elic_tr, toxic_ids, batch_size=atp_batch)
        atp_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(atp_path, atp.numpy())
    else:
        print("gate-AtP already exists, skipping", flush=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Building report ...", flush=True)
    report_path = build_report(model_name)
    print(f"Report -> {report_path}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
