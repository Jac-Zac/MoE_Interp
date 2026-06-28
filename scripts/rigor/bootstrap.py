#!/usr/bin/env python
"""Bootstrap 95% CIs on the rigor deltas — pure post-processing, no model needed.

The held-out prompt sets are small (n ≈ 50–64), so a point estimate like word_frac 0.60→0.33
could be noise. We resample the *paired* per-prompt arrays (baseline and method scored on the same
prompts) with replacement ``--n-boot`` times and report the 2.5/97.5 percentile interval on the
paired delta (baseline − method; positive = removal). A CI that straddles 0 means the effect is not
resolved at this sample size.

Reads any rigor JSON that stores a ``word_hit_per_prompt`` array under ``baseline`` and under each
method block (one level of nesting is walked, so the sufficiency selector→k tree works too).

    .venv/bin/python scripts/rigor/bootstrap.py data/<model>/circuit/rigor/sufficiency_offensive.json
"""

import argparse
import json

import numpy as np


def boot_delta(base, meth, n_boot, rng):
    """95% CI on mean(base) − mean(meth) under paired prompt resampling."""
    n = min(len(base), len(meth))
    base, meth = np.asarray(base[:n]), np.asarray(meth[:n])
    idx = rng.integers(0, n, size=(n_boot, n))
    deltas = base[idx].mean(1) - meth[idx].mean(1)
    return float(base.mean() - meth.mean()), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+")
    p.add_argument("--field", default="word_hit_per_prompt")
    p.add_argument("--n-boot", type=int, default=10000)
    args = p.parse_args()
    rng = np.random.default_rng(0)

    for path in args.paths:
        d = json.load(open(path))
        base = d["baseline"][args.field]
        print(f"\n=== {path} ===\nbaseline = {np.mean(base):.3f}  (n={len(base)}, {args.n_boot} resamples)")
        for name, block in d.items():
            if name in ("meta", "baseline") or not isinstance(block, dict):
                continue
            # flat method block, or a nested selector→k tree
            children = {name: block} if args.field in block else {
                f"{name} k={k}": v for k, v in block.items() if isinstance(v, dict) and args.field in v
            }
            for label, b in children.items():
                delta, lo, hi = boot_delta(base, b[args.field], args.n_boot, rng)
                flag = "" if (lo > 0 or hi < 0) else "  <-- CI straddles 0"
                print(f"  {label:18s} Δ = {delta:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]{flag}")


if __name__ == "__main__":
    main()
