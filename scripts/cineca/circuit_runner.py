#!/usr/bin/env python
"""Run the causal circuit pipeline: gate-AtP localization, expert interventions, report.

Builds a disjoint RealToxicityPrompts train/test split and threads it through every
model-dependent step so all methods share the same identification prompts and are scored
on the same held-out test set. All artifacts land under ``data/<model>/circuit/``. Every
intervention is *expert-level* (gate knockout / expert-output steering) — no residual-stream edits.

The causal localizer is gate-AtP (one backward pass). Exhaustive activation patching was used
once to validate it (the two grids agreed closely, pooled r≈0.69) and is no longer run here.

Usage:
    python scripts/cineca/circuit_runner.py [--model MODEL] [--batch-size N]
                                            [--atp-batch-size N] [--knockout-k N]
                                            [--max-new-tokens N]
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
from moe_interp.circuit.prompts import rtp_split
from moe_interp.circuit.report import build_report
from moe_interp.circuit.steer import run_dose_response, run_expert_steer
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
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument(
        "--n-prompts",
        type=int,
        default=48,
        help="Train-set size (used for grid identification and steer training).",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=24,
        help="Held-out test set size (scored by steer experiment).",
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

    # 1. gate-AtP localization grid (one backward pass) — the causal localizer.
    # Keyed by train-set size so step 2's offensive knockout reuses this exact grid
    # (same prompts, same toxic ids) instead of paying for a second backward pass.
    atp_path = cdir / "attribution" / f"atp_grid_n{len(elic_tr)}.npy"
    if not atp_path.exists():
        print("[1/3] gate-AtP localization grid ...", flush=True)
        atp = gate_attribution(model, elic_tr, toxic_ids, batch_size=atp_batch)
        atp_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(atp_path, atp.numpy())
    else:
        print("[1/3] gate-AtP already exists, skipping", flush=True)

    # 2. Expert-level causal interventions on the SOMP and gate-AtP experts (rtp/offensive, ranked
    # by EVR@k / |AtP|) vs a matched-random control: knockout (necessity) and α-expert-OUTPUT DoM
    # steering (influence), done per expert. Every intervention is expert-level — no residual-stream
    # edits. The key checks are (a) does the causal (AtP) set beat SOMP and random, and (b) does the
    # output stay coherent (distinct1)?
    steer_out = cdir / "steer" / "offensive"
    exp_path = steer_out / "expert_intervention.json"
    if not exp_path.exists():
        print("[2/3] Expert-level interventions (SOMP / AtP vs random) ...", flush=True)
        exp = run_expert_steer(
            model,
            model_name,
            concept="offensive",
            dataset="rtp",
            k=args.knockout_k,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            train=(elic_tr, neut_tr),
            test=(elic_te, neut_te),
        )
        steer_out.mkdir(parents=True, exist_ok=True)
        exp_path.write_text(json.dumps(exp, indent=2))
    else:
        print("[2/3] Expert interventions already exist, skipping", flush=True)

    # 3. Dose-response: cumulative top-1..k SOMP / AtP vs random, for α expert steering, on
    # eliciting prompts only. Shows whether toxicity falls monotonically as more causal experts
    # are hit (and stays flat for random) — the signature of a localized causal set.
    dose_path = steer_out / "dose_response.json"
    if not dose_path.exists():
        print("[3/3] Dose-response curve ...", flush=True)
        dose = run_dose_response(
            model,
            model_name,
            concept="offensive",
            dataset="rtp",
            k=args.knockout_k,
            max_new_tokens=args.max_new_tokens,
            train=(elic_tr, neut_tr),
            test=(elic_te, neut_te),
        )
        dose_path.write_text(json.dumps(dose, indent=2))
    else:
        print("[3/3] Dose-response already exists, skipping", flush=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Building report ...", flush=True)
    report_path = build_report(model_name)
    print(f"Report -> {report_path}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
