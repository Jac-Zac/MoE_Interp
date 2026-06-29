#!/usr/bin/env python
"""Co-firing group ablation — does removing the redundant group break what the sparse set can't?

Single-expert knockout fails because, with 8-of-64 routing, the other experts on the same token
absorb the gate mass. This script targets that mechanism directly: for each causal (gate-AtP)
expert it finds the experts that most often appear in the *same token's top-k* (its in-layer
co-firing neighborhood — the ones that route around it), then knocks out the whole group.

Compares, on the held-out eliciting prompts:
  * AtP-only        — the bare causal set (the redundancy-bound baseline)
  * AtP + co-firing — causal set plus its co-firing neighbors (the redundant group)
  * random group    — a size-matched random set on the same layers (specificity control)

If "AtP + co-firing" removes the concept where "AtP-only" does not, redundancy — not absence of a
causal locus — is what defeated sparse knockout, and the fix is to ablate groups/paths.

    DATA_DIR=./data HF_HUB_OFFLINE=1 .venv/bin/python scripts/rigor/group_ablation.py
"""

import argparse
import json
import os
from collections import Counter

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
from moe_interp.circuit.steer import _causal_grid_set, _matched_random_set
from moe_interp.config import get_default_model, get_device, get_model_dir, set_seed
from moe_interp.pursuit.concepts import CONCEPT_WORDS, build_toxic_token_ids


def cofiring_neighbors(model, prompts, targets, m):
    """For each ``(layer, e*)`` target, the ``m`` experts most often in the same token's top-k."""
    by_layer = {}
    for layer, e in targets:
        by_layer.setdefault(layer, []).append(e)
    counts = {le: Counter() for le in targets}
    for ids in prompts:
        saved = {}
        with torch.no_grad(), model.trace([ids]):
            for layer in by_layer:
                saved[layer] = model.model.layers[layer].mlp.experts.inputs.save()
        for layer, estar_list in by_layer.items():
            _, idx, _ = saved[layer][0]  # idx: (n_tok, top_k)
            idx = idx.cpu()
            for estar in estar_list:
                routed = (idx == estar).any(dim=-1)
                if not routed.any():
                    continue
                others = idx[routed].flatten().tolist()
                for o in others:
                    if o != estar:
                        counts[(layer, estar)][o] += 1
    neighbors = []
    for le, c in counts.items():
        layer = le[0]
        neighbors += [(layer, e) for e, _ in c.most_common(m)]
    return list(dict.fromkeys(neighbors))  # dedup, keep order


def _score(model, prompts, concept_ids, pattern, experts, max_new_tokens):
    """Knock out ``experts`` and return summary + per-prompt word-hit array (for bootstrap.py)."""
    fn = knockout_intervention(experts) if experts else None
    props, hits, distinct = [], [], []
    for ids in prompts:
        cont = generate(model, ids, max_new_tokens, fn)
        props.append(concept_propensity(model, ids, cont, concept_ids, fn))
        hits.append(1.0 if pattern.findall(model.tokenizer.decode(cont)) else 0.0)
        distinct.append(len(set(cont)) / max(len(cont), 1))
    return {
        "propensity": float(np.mean(props)),
        "word_frac": float(np.mean(hits)),
        "distinct1": float(np.mean(distinct)),
        "word_hit_per_prompt": hits,
    }


def main():
    load_dotenv()
    set_seed(1337)
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=get_default_model())
    p.add_argument("--concept", default="offensive")
    p.add_argument("--k", type=int, default=15, help="number of causal (AtP) experts")
    p.add_argument(
        "--m", type=int, default=4, help="co-firing neighbors per causal expert"
    )
    p.add_argument("--n-prompts", type=int, default=100)
    p.add_argument("--n-test", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=24)
    args = p.parse_args()

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
    concept_ids = build_toxic_token_ids(model.tokenizer, CONCEPT_WORDS[args.concept])
    pattern = concept_regex(CONCEPT_WORDS[args.concept])

    md = get_model_dir(args.model)
    atp_path = md / "circuit" / "attribution" / f"atp_grid_n{len(elic_tr)}.npy"
    atp = _causal_grid_set(atp_path, args.k)
    if not atp:
        raise RuntimeError(
            f"No gate-AtP grid at {atp_path}; run the circuit pipeline first."
        )

    neighbors = cofiring_neighbors(model, elic_tr, atp, args.m)
    group = list(dict.fromkeys(atp + neighbors))
    rand_group = _matched_random_set(group, ne)

    sets = {"AtP-only": atp, "AtP+cofiring": group, "random-group": rand_group}
    out = {
        "meta": {
            "model": args.model,
            "concept": args.concept,
            "k": args.k,
            "m": args.m,
            "n_test": len(elic_te),
            "group_size": len(group),
        }
    }
    out["baseline"] = _score(
        model, elic_te, concept_ids, pattern, [], args.max_new_tokens
    )
    print(f"[baseline] word_frac={out['baseline']['word_frac']:.3f}", flush=True)
    for name, s in sets.items():
        block = _score(model, elic_te, concept_ids, pattern, s, args.max_new_tokens)
        block["n_experts"] = len(s)
        block["experts"] = [list(le) for le in s]
        out[name] = block
        print(
            f"[{name}] n={len(s)} word_frac={block['word_frac']:.3f} "
            f"prop={block['propensity']:+.3f} d1={block['distinct1']:.2f}",
            flush=True,
        )

    out_dir = md / "circuit" / "rigor"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"group_ablation_{args.concept}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"group ablation -> {path}", flush=True)


if __name__ == "__main__":
    main()
