# Paper-rigor experiments

Small add-ons that turn two *asserted* claims into *measured* ones, plus error bars. All run on the
toxicity circuit and reuse the cached gate-AtP grid (`atp_grid_n<N>.npy`) from the main pipeline —
so run `scripts/orfeo/circuit.sh` first with the same `--n-prompts`.

| script | turns this assertion → into a measurement | output |
|---|---|---|
| `sufficiency_curve.py` | "top-k knockout is redundant" → curve of removal vs #experts (AtP/SOMP/random) | `circuit/rigor/sufficiency_<concept>.json` |
| `group_ablation.py` | "the model routes around the sparse set" → ablate the co-firing group, not singletons | `circuit/rigor/group_ablation_<concept>.json` |
| `bootstrap.py` | point estimates → 95% CIs (CPU, no model) | prints to stdout |

**Run (Orfeo):** `sbatch scripts/orfeo/rigor.sh` then bootstrap on a login node:
```bash
python scripts/rigor/bootstrap.py data/*/circuit/rigor/sufficiency_offensive.json
python scripts/rigor/bootstrap.py data/*/circuit/rigor/group_ablation_offensive.json
```

## What each one is for

- **Sufficiency curve.** The redundancy claim currently rests on one point (10% knockout barely
  moves it). This sweeps `k = 1..103` and plots removal vs `k` for the causal (AtP), correlational
  (SOMP) and random selectors on the *same* held-out prompts. If even `k=103` doesn't remove the
  concept, redundancy is proven, and the AtP-vs-random gap shows whether the causal ranking matters
  at all. This is the headline figure for "influence ≠ necessity".

- **Co-firing group ablation.** Single-expert knockout fails because, with 8-of-64 routing, the
  other experts on the same token absorb the gate mass. For each causal expert we find the experts
  most often in the *same token's top-k* (its in-layer co-firing neighbourhood) and knock out the
  whole group. If "AtP + co-firing" removes the concept where "AtP-only" can't, redundancy — not
  absence of a causal locus — is what defeated sparse knockout, and the fix is groups/paths. This is
  the constructive follow-up to the negative knockout result.

- **Bootstrap CIs.** n ≈ 64, so resample the paired per-prompt arrays 10k× and report the 2.5/97.5
  interval on each delta. A CI that straddles 0 = the effect is noise at this n.

## Still a spec (implement with repo access — needs the per-expert activation matrices)

**EVR floor & ceiling.** "SOMP gets 1.46% EVR@10" has no reference scale. Add two baselines on the
same centered activation matrix `E` (n_docs × d) that SOMP already decomposes per expert:
- *Floor (random-dictionary):* select `k` atoms from **random** unembedding rows instead of the best
  ones. If SOMP barely beats this, "2× over the lens" is unimpressive.
- *Ceiling (PCA):* the top-`k` principal components of `E` are the best possible `k`-direction linear
  fit — the most variance *any* `k` directions can explain. SOMP is restricted to vocabulary
  directions, so PCA upper-bounds it. Report **"SOMP captures X% of the PCA ceiling"** = the cost of
  demanding interpretable (vocab) directions. ~20 lines once `E` is in hand; persist `E` for a sample
  of experts during the pursuit run, then compute floor/ceiling offline.
