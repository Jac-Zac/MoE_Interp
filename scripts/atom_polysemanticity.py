"""Quantify expert polysemanticity from the *content* of an expert's top SOMP atoms.

Low EVR alone does not establish polysemanticity (it is consistent with superposition, but
also with an expert whose output simply lies off the vocabulary manifold). The direct test the
slides need is: do an expert's top atoms span *many unrelated token directions*? We answer it in
the model's own readout geometry — each atom is a row of the (L2-normalized) unembedding, so two
atoms are "related" when their unembedding rows are aligned (high cosine) and "unrelated" when
near-orthogonal.

For each expert we take its top-N atoms and report two threshold-questions:

* participation ratio  PR = N^2 / ||A Aᵀ||_F^2  of the N unit atom-directions — the *effective
  number of independent directions* the expert mixes (PR≈1: one family; PR≈N: N orthogonal
  directions). Threshold-free.
* number of semantic families = agglomerative clusters of the atom rows at cosine ≥ 0.4.

Both are anchored by two controls: a *monosemantic* set (one concept's words → low PR / 1 family)
and a *random-vocab* set (orthogonal ceiling → PR≈N). If experts sit near the random ceiling,
their "specialization" is a thin layer over a polysemantic core.

Runs locally off the cached ``dictionary.h5`` if present; otherwise lazy-loads ``lm_head.weight``
from the HF safetensors shards (no full-model load). On a cluster (Orfeo/Cineca) just point
``--model`` / ``--results`` at the local paths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def load_unembedding_rows(model_name: str, cache_dict: Path | None) -> torch.Tensor:
    """L2-normalized unembedding (vocab × d). Prefer the cached dictionary; else pull just
    ``lm_head.weight`` from the safetensors shards (lazy — no full model materialization).
    """
    if cache_dict is not None and cache_dict.exists():
        from moe_interp.capture.cache import load_unembedding

        return torch.nn.functional.normalize(
            load_unembedding(cache_dict).float(), dim=1
        )

    from huggingface_hub import try_to_load_from_cache
    from safetensors import safe_open
    from transformers import AutoConfig

    AutoConfig.from_pretrained(model_name)  # ensures the snapshot is present
    index = try_to_load_from_cache(model_name, "model.safetensors.index.json")
    weight_map = json.load(open(index))["weight_map"]
    shard = Path(index).parent / weight_map["lm_head.weight"]
    with safe_open(shard, framework="pt") as f:
        W = f.get_tensor("lm_head.weight").float()
    return torch.nn.functional.normalize(W, dim=1)


def reverse_decode_map(tokenizer, vocab_size: int) -> dict[str, int]:
    """``decode([id]) -> id`` so the stored atom *strings* map back to unembedding rows."""
    return {tokenizer.decode([i]): i for i in range(vocab_size)}


def participation_ratio(rows: torch.Tensor) -> float:
    """Effective number of independent directions among unit-norm ``rows`` (N × d)."""
    if rows.shape[0] < 2:
        return float(rows.shape[0])
    g = rows @ rows.T  # Gram of unit vectors; trace = N
    return float(g.shape[0] ** 2 / (g**2).sum())


def _components(rows: torch.Tensor, cos_thr: float) -> list[int]:
    """Union-find root per atom; atoms share a root iff connected at cosine ≥ ``cos_thr``."""
    n = rows.shape[0]
    sim = (rows @ rows.T).numpy()
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= cos_thr:
                parent[find(i)] = find(j)
    return [find(i) for i in range(n)]


def largest_family_frac(rows: torch.Tensor, cos_thr: float = 0.4) -> float:
    """Share of atoms in the biggest cosine-family — robust polysemanticity score.

    ≈1 for a monosemantic expert (one dominant topic); ≈1/N when the top atoms are a grab-bag of
    mutually-unrelated directions. Unlike the raw family *count* it is not saturated by SOMP's
    atom decorrelation, so it cleanly separates the few clean specialists from the polysemantic
    majority."""
    n = rows.shape[0]
    if n < 2:
        return 1.0
    from collections import Counter

    return max(Counter(_components(rows, cos_thr)).values()) / n


def n_families(rows: torch.Tensor, cos_thr: float = 0.4) -> int:
    """# connected components of the atom graph (edge when cosine ≥ ``cos_thr``):
    single-linkage clusters = the number of mutually-unrelated token families. Pure union-find,
    no scipy/sklearn so it runs on a bare cluster node."""
    if rows.shape[0] < 2:
        return int(rows.shape[0])
    return len(set(_components(rows, cos_thr)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0924-Instruct")
    ap.add_argument(
        "--results",
        default="data/allenai_OLMoE_1B_7B_0924_Instruct/pursuit/pile10k/results.jsonl",
    )
    ap.add_argument("--cache-dict", default=None, help="path to cached dictionary.h5")
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--cos-thr", type=float, default=0.4)
    ap.add_argument("--min-acts", type=int, default=50)
    ap.add_argument("--out", default="presentation/assets/atom_polysemanticity.png")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    D = load_unembedding_rows(
        args.model, Path(args.cache_dict) if args.cache_dict else None
    )
    rev = reverse_decode_map(tok, D.shape[0])

    rows = [json.loads(l) for l in open(args.results)]
    rows = [
        r for r in rows if r.get("evr") and r.get("n_activations", 0) >= args.min_acts
    ]

    prs, fams, lffs, finals = [], [], [], []
    for r in rows:
        ids = [rev[t] for t in r["tokens"][: args.top_n] if t in rev]
        if len(ids) < 5:
            continue
        a = D[ids]
        prs.append(participation_ratio(a))
        fams.append(n_families(a, args.cos_thr))
        lffs.append(largest_family_frac(a, args.cos_thr))
        finals.append(r["evr"][-1])
    prs, fams, lffs, finals = map(np.array, (prs, fams, lffs, finals))

    # Controls: monosemantic (one concept's single-token words) and random-vocab ceiling.
    from moe_interp.pursuit.concepts import CONCEPT_WORDS

    def concept_rows(words):
        ids = [
            tok(w, add_special_tokens=False).input_ids[0]
            for w in words
            if len(tok(w, add_special_tokens=False).input_ids) == 1
        ]
        return D[ids[: args.top_n]]

    mono = concept_rows(CONCEPT_WORDS["numbers"])
    rng = np.random.default_rng(0)
    rand = D[rng.integers(0, D.shape[0], args.top_n)]

    summary = {
        "n_experts": int(len(prs)),
        "top_n": args.top_n,
        "cos_thr": args.cos_thr,
        "largest_family_share_median": float(np.median(lffs)),
        "largest_family_share_q1_q3": [
            float(np.percentile(lffs, 25)),
            float(np.percentile(lffs, 75)),
        ],
        "pct_experts_with_core_ge4_atoms": float(np.mean(lffs * args.top_n >= 4) * 100),
        "PR_median": float(np.median(prs)),
        "families_median": float(np.median(fams)),
        "control_monosemantic_numbers": {
            "largest_family_share": largest_family_frac(mono, args.cos_thr),
            "PR": participation_ratio(mono),
            "families": n_families(mono, args.cos_thr),
        },
        "control_random_vocab": {
            "largest_family_share": largest_family_frac(rand, args.cos_thr),
            "PR": participation_ratio(rand),
            "families": n_families(rand, args.cos_thr),
        },
        "sharpest_evr_experts": [
            {
                "le": f"L{r['layer']:02d}E{r['expert']:02d}",
                "final_evr": r["evr"][-1],
                "largest_family_share": largest_family_frac(
                    D[[rev[t] for t in r["tokens"][: args.top_n] if t in rev]],
                    args.cos_thr,
                ),
                "top12": [t.strip() for t in r["tokens"][:12]],
            }
            for r in sorted(rows, key=lambda r: r["evr"][-1], reverse=True)[:5]
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(out.with_suffix(".json")).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        lff_mono = summary["control_monosemantic_numbers"]["largest_family_share"]
        # Deck palette: pink = data, blue = expert median, grey = control reference.
        PINK, BLUE, GREY = "#cf1c77", "#3462e7", "#9aa0a6"
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
        ax1.hist(lffs, bins=np.linspace(0, 1, 31), color=PINK, alpha=0.85)
        ax1.axvline(
            lff_mono,
            color=GREY,
            lw=2,
            ls="--",
            label=f"monosemantic (numbers) = {lff_mono:.2f}",
        )
        ax1.axvline(
            np.median(lffs),
            color=BLUE,
            lw=2,
            label=f"expert median = {np.median(lffs):.2f}",
        )
        ax1.set_xlabel(
            f"largest token-family share of top-{args.top_n} atoms (cos ≥ {args.cos_thr})"
        )
        ax1.set_ylabel("# experts")
        ax1.set_title("Does one topic dominate the readout?")
        ax1.legend(fontsize=8)
        ax2.hist(prs, bins=30, color=PINK, alpha=0.85)
        ax2.axvline(
            np.median(prs),
            color=BLUE,
            lw=2,
            label=f"expert median = {np.median(prs):.1f} / {args.top_n}",
        )
        ax2.set_xlabel(f"participation ratio of top-{args.top_n} atom directions")
        ax2.set_ylabel("# experts")
        ax2.set_title("Effective # independent directions")
        ax2.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out, dpi=130)
        print("wrote", out)
    except Exception as e:  # noqa: BLE001
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
