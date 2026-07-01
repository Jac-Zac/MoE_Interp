#!/usr/bin/env python
"""Run the knockout/downweight sweep (no steering) with per-prompt bootstrap error bars.

Reuses the circuit study's disjoint RealToxicityPrompts train/test split and the gate-AtP
localizer, then sweeps gate downweighting (``s=0`` knockout ... ``s=0.9`` 10% downweight) on the
SOMP / AtP / matched-random expert sets at two budgets (fractions of all experts). Writes
``data/<model>/circuit/downweight/sweep_<concept>.json`` with per-prompt arrays + 95% CIs.

The AtP grid is shared with ``circuit_runner.py`` (keyed by train-set size); if it is missing this
script computes it once. The SOMP concept pursuit must already exist (run the concept-restricted
pursuit first) — :func:`expert_intervention_sets` raises otherwise.

Usage:
    python scripts/cineca/downweight_runner.py [--model M] [--n-prompts N] [--n-test N]
        [--max-new-tokens N] [--budgets 0.01 0.05] [--scales 0.9 0.75 0.5 0.25 0.0]
"""

import argparse
import os

import numpy as np
from dotenv import load_dotenv
from nnsight import LanguageModel

from moe_interp.circuit.attribution import gate_attribution
from moe_interp.circuit.downweight import run_downweight_sweep
from moe_interp.circuit.prompts import rtp_split
from moe_interp.config import get_default_model, get_device, get_model_dir, set_seed
from moe_interp.pursuit.concepts import build_concept_token_ids


def main():
    load_dotenv()
    set_seed(1337)

    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default=get_default_model())
    p.add_argument("--concept", type=str, default="offensive")
    p.add_argument("--dataset", type=str, default="rtp")
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--max-new-tokens", type=int, default=24)
    p.add_argument("--n-prompts", type=int, default=100)
    p.add_argument("--n-test", type=int, default=64)
    p.add_argument("--hi", type=float, default=0.5)
    p.add_argument("--challenging", action="store_true")
    p.add_argument(
        "--budgets",
        type=float,
        nargs="+",
        default=[0.01, 0.05],
        help="Expert budgets as a fraction of n_layers*n_experts (top-k per selector).",
    )
    p.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=[0.9, 0.5, 0.25, 0.0],
        help="Gate multipliers to sweep (0.0 = knockout, 0.9 = 10%% downweight).",
    )
    p.add_argument("--n-boot", type=int, default=10000)
    args = p.parse_args()

    model_name = args.model
    cdir = get_model_dir(model_name) / "circuit"
    device_map = os.environ.get("DEVICE_MAP", str(get_device()))
    model = LanguageModel(
        model_name, device_map=device_map, dtype="auto", dispatch=True
    )

    elic_tr, elic_te, neut_tr, neut_te = rtp_split(
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
    print(f"{len(elic_tr)} train + {len(elic_te)} test eliciting prompts", flush=True)

    # gate-AtP grid (shared with circuit_runner, keyed by train size) — build if absent.
    atp_path = cdir / "attribution" / f"atp_grid_n{len(elic_tr)}{regime}.npy"
    if not atp_path.exists():
        print("gate-AtP grid missing; computing ...", flush=True)
        toxic_ids = build_concept_token_ids(model.tokenizer)
        atp = gate_attribution(model, elic_tr, toxic_ids, batch_size=args.batch_size)
        atp_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(atp_path, atp.numpy())

    out_path = cdir / "downweight" / f"sweep_{args.concept}{regime}.json"
    run_downweight_sweep(
        model,
        model_name,
        concept=args.concept,
        dataset=args.dataset,
        train=(elic_tr, neut_tr),
        test=(elic_te, neut_te),
        out_path=out_path,
        budgets_frac=tuple(args.budgets),
        scales=tuple(args.scales),
        max_new_tokens=args.max_new_tokens,
        atp_grid_path=atp_path,
        n_boot=args.n_boot,
    )
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
