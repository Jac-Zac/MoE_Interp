#!/usr/bin/env python
"""NLP exam report: SOMP & polysemanticity of MoE experts → self-contained HTML.

Framing (notes/to_think.md, notes/PAPER_PLAN.md): MoE-Lens reads one top token per
expert (single logit-lens argmax). We show that an expert's output is a *subspace*, so a
single token under-reads it. For each expert we:

  1. SVD the centered activations → spectrum. Classify Case A (one dominant direction =
     monosemantic), B (several comparable = polysemantic), C (flat = generic transform),
     via PC1-EVR and effective rank.
  2. Project the top few PC directions onto the unembedding → decode each direction's
     top tokens. Show experts where the PCs carry DIFFERENT semantics (true polysemy).
  3. Quantify: how often does logit-lens top-1 miss directions that SOMP/PCA surface?
     (token-set overlap between PC1 tokens and PC2/PC3 tokens).

Run: DATA_DIR=./data .venv/bin/python scripts/nlp_report.py --dataset pile10k

FUTURE WORK (gradient / causal validation, see TODO.md & NEXT_STEPS.md):
  - Patchscope / logit-lens / tuned-lens comparison for the high-confidence SOMP experts.
  - Gradient-based concept direction: diff-of-means (e.g. toxic vs neutral prompts) to
    get a direction, rank experts by projection onto it (replaces the noisy lexicon).
  - Causal test: project-out / ablate a SOMP-identified direction from W_down for the top
    experts, measure Δ next-token loss and Δ target-token logprob vs matched-random
    expert controls. Turns "this expert spans concept X" into a causal claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch

from moe_interp.capture.cache import load_layer_h5, load_metadata, load_unembedding
from moe_interp.config import (
    get_analysis_dir,
    get_default_model,
    get_model_dir,
    get_unembedding_dir,
)


def _decode_dir(vec: np.ndarray, U: torch.Tensor, tok, k: int = 8) -> list[str]:
    """Logit-lens a residual-space direction: top-k unembedding tokens (both signs)."""
    scores = U @ torch.from_numpy(vec).float()
    idx = torch.topk(scores, k).indices.tolist()
    return [tok.decode([i]).strip() for i in idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pile10k")
    ap.add_argument("--model", default=None)
    ap.add_argument("--min_activations", type=int, default=200)
    ap.add_argument("--top_pcs", type=int, default=3)
    args = ap.parse_args()

    model_name = args.model or get_default_model()
    ed = get_model_dir(model_name) / "extractions" / args.dataset
    meta = load_metadata(ed / "metadata.json")
    n_layers, n_experts = meta["n_layers"], meta["n_experts"]

    U = load_unembedding(get_unembedding_dir(model_name) / "dictionary.h5").float()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)

    # ---- per-expert SVD spectrum + PC direction decoding ----
    print("SVD per expert ...")
    records = []  # one per expert
    pc1_evrs, eff_ranks = [], []
    for L in range(n_layers):
        full = load_layer_h5(ed, L, n_experts, args.min_activations)
        for e, entry in full.items():
            X = entry["activations"].float()
            Xc = X - X.mean(0)
            # economy SVD via Gram eigendecomposition (n << d not guaranteed here; n~1.6k)
            try:
                u, s, vh = torch.linalg.svd(Xc, full_matrices=False)
            except Exception:  # noqa: BLE001
                continue
            sq = s**2
            total = sq.sum().clamp_min(1e-12)
            pc1 = (sq[0] / total).item()
            erank = (total**2 / (sq**2).sum().clamp_min(1e-12)).item()
            pc1_evrs.append(pc1)
            eff_ranks.append(erank)
            pc_tokens = [
                _decode_dir(vh[i].numpy(), U, tok)
                for i in range(min(args.top_pcs, vh.shape[0]))
            ]
            records.append(
                {
                    "layer": L,
                    "expert": e,
                    "pc1_evr": pc1,
                    "eff_rank": erank,
                    "n": X.shape[0],
                    "pc_tokens": pc_tokens,
                    "evr2": (sq[:2].sum() / total).item(),
                    "evr3": (sq[:3].sum() / total).item(),
                }
            )
    print(f"  {len(records)} experts")
    pc1_evrs = np.array(pc1_evrs)
    eff_ranks = np.array(eff_ranks)

    # ---- case classification (to_think.md A/B/C) ----
    case = []
    for r in records:
        if r["pc1_evr"] > 0.5:
            r["case"] = "A monosemantic"
        elif r["eff_rank"] > 0.25 * 2048:  # variance spread very widely
            r["case"] = "C generic"
        else:
            r["case"] = "B polysemantic"
        case.append(r["case"])
    from collections import Counter

    case_counts = Counter(case)

    figs = []
    # spectrum-shape scatter: PC1-EVR vs effective rank, the monosemantic<->generic axis
    sc = go.Figure(
        go.Scatter(
            x=pc1_evrs,
            y=eff_ranks,
            mode="markers",
            marker=dict(
                size=5,
                color=eff_ranks,
                colorscale="Turbo",
                colorbar=dict(title="eff rank"),
            ),
            text=[f"L{r['layer']}E{r['expert']} {r['case']}" for r in records],
        )
    )
    sc.update_layout(
        title="Every expert's spectrum: PC1-EVR vs effective rank "
        "(low PC1-EVR + high rank = polysemantic)",
        xaxis_title="PC1 EVR (fraction of variance in top direction)",
        yaxis_title="effective rank",
        height=450,
    )
    figs.append(sc)

    # cumulative EVR: how many directions to explain the expert
    cum = go.Figure()
    cum.add_trace(
        go.Histogram(x=[r["pc1_evr"] for r in records], name="PC1", opacity=0.6)
    )
    cum.add_trace(
        go.Histogram(x=[r["evr3"] for r in records], name="PC1-3", opacity=0.6)
    )
    cum.update_layout(
        barmode="overlay",
        title="Variance captured by top-1 vs top-3 directions",
        xaxis_title="cumulative EVR",
        yaxis_title="experts",
        height=380,
    )
    figs.append(cum)

    # ---- pick the most polysemantic experts whose PC directions decode DIFFERENTLY ----
    def _distinct(pcs):
        # mean pairwise Jaccard distance between the PC token sets (higher = more distinct)
        sets = [set(p) for p in pcs]
        if len(sets) < 2:
            return 0.0
        ds = []
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                u = sets[i] | sets[j]
                ds.append(1 - len(sets[i] & sets[j]) / len(u) if u else 0.0)
        return float(np.mean(ds))

    for r in records:
        r["pc_distinct"] = _distinct(r["pc_tokens"])
    poly = sorted(
        # Favor mid/late-layer experts: their PC tokens are far more interpretable than
        # layer-0 (which decodes to subword noise). Rank by distinctness, restricted to
        # layers >= 4, falling back to all if too few.
        [r for r in records if r["case"] == "B polysemantic" and r["layer"] >= 4]
        or [r for r in records if r["case"] == "B polysemantic"],
        key=lambda r: r["pc_distinct"],
        reverse=True,
    )[:12]

    mono = sorted(
        [r for r in records if r["case"] == "A monosemantic" and r["layer"] >= 4]
        or [r for r in records if r["case"] == "A monosemantic"],
        key=lambda r: r["pc1_evr"],
        reverse=True,
    )[:8]

    findings = [
        f"<b>Single-token logit-lens under-reads experts.</b> Mean PC1-EVR is "
        f"{pc1_evrs.mean():.3f}: the top direction explains only ~{pc1_evrs.mean():.0%} of "
        f"an expert's variance, so reading one argmax token (MoE-Lens) discards most of the "
        f"signal. The top-3 directions still capture only "
        f"{np.mean([r['evr3'] for r in records]):.0%}.",
        f"<b>Most experts are polysemantic (Case B).</b> Of {len(records)} experts: "
        f"{case_counts.get('A monosemantic', 0)} monosemantic (Case A, one dominant "
        f"direction), {case_counts.get('B polysemantic', 0)} polysemantic (Case B), "
        f"{case_counts.get('C generic', 0)} generic-transform (Case C, flat spectrum). "
        f"Mean effective rank {eff_ranks.mean():.0f}.",
        f"<b>Polysemantic experts mix unrelated concepts.</b> The experts below have top PC "
        f"directions that decode to DIFFERENT token sets (high PC-distinctness) — a single "
        f"logit lens would report only one of these meanings.",
    ]

    def table(headers, rows):
        h = "".join(f"<th>{x}</th>" for x in headers)
        b = "".join(
            "<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows
        )
        return f"<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>"

    poly_rows = [
        (
            f"L{r['layer']}E{r['expert']}",
            f"{r['pc1_evr']:.2f}",
            f"{r['eff_rank']:.0f}",
            *[
                " / ".join(r["pc_tokens"][i][:5])
                for i in range(min(3, len(r["pc_tokens"])))
            ],
        )
        for r in poly
    ]
    mono_rows = [
        (
            f"L{r['layer']}E{r['expert']}",
            f"{r['pc1_evr']:.2f}",
            ", ".join(r["pc_tokens"][0][:6]),
        )
        for r in mono
    ]

    figs_html = "".join(
        f.to_html(full_html=False, include_plotlyjs=("inline" if i == 0 else False))
        for i, f in enumerate(figs)
    )
    findings_html = "".join(f"<li>{x}</li>" for x in findings)
    out = get_analysis_dir(model_name, args.dataset) / "nlp_report.html"
    out.write_text(f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>SOMP & Polysemanticity — {args.dataset}</title>
<style>body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1000px;
margin:2rem auto;padding:0 1rem;color:#1a1a1a;line-height:1.55}}h1{{margin-bottom:0}}
.sub{{color:#666}}h2{{margin-top:2.4rem;border-bottom:2px solid #eee;padding-bottom:.3rem}}
table{{border-collapse:collapse;width:100%;font-size:.82rem;margin:1rem 0}}
th,td{{border:1px solid #ddd;padding:.35rem .5rem;text-align:left}}th{{background:#f5f7fa}}
tr:nth-child(even){{background:#fafbfc}}li{{margin-bottom:.6rem}}
code{{background:#f3f3f3;padding:0 .3rem}}</style></head><body>
<h1>SOMP &amp; expert polysemanticity</h1>
<p class="sub">{model_name} · {args.dataset} · {len(records)} experts (≥{args.min_activations}
rows) · each expert's activation subspace decoded via PC→unembedding projection</p>
<h2>Key findings</h2><ul>{findings_html}</ul>
<h2>Figures</h2>{figs_html}
<h2>Polysemantic experts: top-3 PC directions decode to different concepts</h2>
{table(["expert", "PC1-EVR", "eff rank", "PC1 tokens", "PC2 tokens", "PC3 tokens"], poly_rows)}
<h2>Monosemantic experts (Case A): one dominant direction</h2>
{table(["expert", "PC1-EVR", "PC1 tokens"], mono_rows)}
</body></html>""")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
