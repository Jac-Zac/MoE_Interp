#!/usr/bin/env python
"""Sufficiency curve: concept removal vs number of experts knocked out (AtP / SOMP / random).

This *measures* the top-k-redundancy claim instead of asserting it. For a sweep of set sizes
``k`` (a prefix of each selector's ranking) we zero the gates of the top-k experts during greedy
generation on the held-out eliciting prompts and record how far the concept drops. Three lines:

  * AtP    — top-k promoters from the cached gate-AtP grid (the causal selector)
  * SOMP   — top-k by concept EVR@10 from the concept-restricted pursuit (correlational)
  * random — layer-matched to the AtP prefix at each k (specificity control)

If even k = 103 (10% of all 1024 experts) fails to remove the concept, redundancy is proven, not
claimed. Per-prompt arrays are dumped so ``bootstrap.py`` can put a CI on each point.

Reproducible for ``--concept offensive --dataset rtp`` (uses ``rtp_split``); other concepts need
their own eliciting-prompt source.

    DATA_DIR=./data HF_HUB_OFFLINE=1 .venv/bin/python scripts/rigor/sufficiency_curve.py
"""

import argparse
import json
import os

import numpy as np
import torch
from dotenv import load_dotenv
from nnsight import LanguageModel

from moe_interp.capture.model_adapter import model_num_experts
from moe_interp.circuit.intervene import (
    concept_propensity,
    concept_regex,
    generate,
    knockout_intervention,
)
from moe_interp.circuit.prompts import rtp_split
from moe_interp.circuit.steer import (
    _causal_grid_set,
    _matched_random_set,
    somp_concept_experts_evr,
)
from moe_interp.config import get_default_model, get_device, get_model_dir, set_seed
from moe_interp.pursuit.concepts import CONCEPT_WORDS, build_concept_token_ids


def _score(model, prompts, concept_ids, pattern, experts, max_new_tokens):
    """Knock out ``experts`` and return per-prompt (propensity, word-hit 0/1, distinct-1) arrays."""
    fn = knockout_intervention(experts) if experts else None
    props, hits, distinct = [], [], []
    for ids in prompts:
        cont = generate(model, ids, max_new_tokens, fn)
        props.append(concept_propensity(model, ids, cont, concept_ids, fn))
        hits.append(1.0 if pattern.findall(model.tokenizer.decode(cont)) else 0.0)
        distinct.append(len(set(cont)) / max(len(cont), 1))
    return np.array(props), np.array(hits), np.array(distinct)


def main():
    load_dotenv()
    set_seed(1337)
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=get_default_model())
    p.add_argument("--concept", default="offensive")
    p.add_argument("--dataset", default="rtp")
    p.add_argument(
        "--ks", default="1,2,5,10,25,50,103", help="comma-separated set sizes"
    )
    p.add_argument(
        "--n-prompts",
        type=int,
        default=100,
        help="train size (selects the AtP grid file)",
    )
    p.add_argument("--n-test", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=24)
    args = p.parse_args()
    ks = [int(x) for x in args.ks.split(",")]
    kmax = max(ks)

    model = LanguageModel(
        args.model,
        device_map=os.environ.get("DEVICE_MAP", str(get_device())),
        dtype="auto",
        dispatch=True,
    )
    ne = model_num_experts(model)
    elic_tr, elic_te, _, _ = rtp_split(
        model.tokenizer, n_train=args.n_prompts, n_test=args.n_test
    )
    concept_ids = build_concept_token_ids(model.tokenizer, CONCEPT_WORDS[args.concept])
    pattern = concept_regex(CONCEPT_WORDS[args.concept])

    # Full ranked selector lists (prefixes give every k).
    md = get_model_dir(args.model)
    atp_path = md / "circuit" / "attribution" / f"atp_grid_n{len(elic_tr)}.npy"
    atp_full = _causal_grid_set(atp_path, kmax) or []
    somp_full = somp_concept_experts_evr(args.model, args.dataset, args.concept, kmax)
    if not atp_full:
        raise RuntimeError(
            f"No gate-AtP grid at {atp_path}; run the circuit pipeline (localize) first."
        )
    rand_full = _matched_random_set(atp_full, ne)
    selectors = {"AtP": atp_full, "SOMP": somp_full, "random": rand_full}

    out = {
        "meta": {
            "model": args.model,
            "concept": args.concept,
            "ks": ks,
            "ne": ne,
            "n_test": len(elic_te),
        }
    }
    base_p, base_h, base_d = _score(
        model, elic_te, concept_ids, pattern, [], args.max_new_tokens
    )
    out["baseline"] = {
        "propensity": float(base_p.mean()),
        "word_frac": float(base_h.mean()),
        "distinct1": float(base_d.mean()),
        "propensity_per_prompt": base_p.tolist(),
        "word_hit_per_prompt": base_h.tolist(),
    }
    for name, full in selectors.items():
        out[name] = {}
        for k in ks:
            if k > len(full):
                continue
            pr, hi, di = _score(
                model, elic_te, concept_ids, pattern, full[:k], args.max_new_tokens
            )
            out[name][str(k)] = {
                "propensity": float(pr.mean()),
                "word_frac": float(hi.mean()),
                "distinct1": float(di.mean()),
                "propensity_per_prompt": pr.tolist(),
                "word_hit_per_prompt": hi.tolist(),
            }
            print(
                f"[{name} k={k}] word_frac={hi.mean():.3f} prop={pr.mean():+.3f} d1={di.mean():.2f}",
                flush=True,
            )

    out_dir = md / "circuit" / "rigor"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"sufficiency_{args.concept}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"sufficiency curve -> {path}", flush=True)


if __name__ == "__main__":
    main()
