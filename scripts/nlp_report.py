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
from collections import Counter
from html import escape

import numpy as np
import plotly.graph_objects as go
import torch

from moe_interp.analysis.decode import top_tokens_for_vector
from moe_interp.analysis.report_html import (
    figs_to_html,
    html_page,
    load_pursuit_map,
    table,
)
from moe_interp.analysis.summaries import compute_expert_summary
from moe_interp.capture.cache import load_layer_h5, load_metadata, load_unembedding
from moe_interp.config import (
    get_analysis_dir,
    get_default_model,
    get_model_dir,
    get_unembedding_dir,
)
from moe_interp.pursuit import projection_pursuit


def _decode_dir(vec: np.ndarray, U: torch.Tensor, tok, k: int = 8) -> list[str]:
    """Logit-lens a residual-space direction: top-k unembedding tokens (stripped)."""
    return [
        t.strip() for t in top_tokens_for_vector(torch.from_numpy(vec), U, tok, k=k)
    ]


def _top_input_tokens(tokens: torch.Tensor, tok, k: int = 8) -> list[str]:
    counts = Counter(int(t) for t in tokens.tolist())
    return [tok.decode([tid]).strip() for tid, _ in counts.most_common(k)]


def _join_tokens(tokens: list[str], limit: int = 6) -> str:
    return ", ".join(escape(t) for t in tokens[:limit])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pile10k")
    ap.add_argument("--model", default=None)
    ap.add_argument("--min_activations", type=int, default=200)
    ap.add_argument("--top_pcs", type=int, default=3)
    ap.add_argument("--pc1_pursuit_k", type=int, default=10)
    args = ap.parse_args()

    model_name = args.model or get_default_model()
    ed = get_model_dir(model_name) / "extractions" / args.dataset
    meta = load_metadata(ed / "metadata.json")
    n_layers, n_experts, d_model = meta["n_layers"], meta["n_experts"], meta["d_model"]

    U = load_unembedding(get_unembedding_dir(model_name) / "dictionary.h5").float()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    pursuit = load_pursuit_map(model_name, args.dataset)

    # ---- per-expert centered spectrum (shared robust Gram-SVD) + PC direction decoding ----
    print("Spectrum per expert ...")
    records = []  # one per expert
    pc1_evrs, eff_ranks = [], []
    for L in range(n_layers):
        full = load_layer_h5(ed, L, n_experts, args.min_activations)
        for e, entry in full.items():
            summ = compute_expert_summary(
                entry["activations"], L, e, top_pcs=args.top_pcs
            )
            sq = summ.singular_values**2
            total = sq.sum().clamp_min(1e-12)
            pc1_evrs.append(summ.pc1_evr)
            eff_ranks.append(summ.effective_rank)
            pc_tokens = [
                _decode_dir(summ.top_pc_directions[i].numpy(), U, tok)
                for i in range(summ.top_pc_directions.shape[0])
            ]
            records.append(
                {
                    "layer": L,
                    "expert": e,
                    "pc1_evr": summ.pc1_evr,
                    "eff_rank": summ.effective_rank,
                    "n": summ.count,
                    "pc_tokens": pc_tokens,
                    "input_tokens": _top_input_tokens(entry["tokens"], tok),
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
        elif r["eff_rank"] > 0.25 * d_model:  # variance spread very widely
            r["case"] = "C generic"
        else:
            r["case"] = "B polysemantic"
        case.append(r["case"])
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

    device = "cpu"
    dictionary = U.to(device)
    for r in mono:
        source = load_layer_h5(ed, r["layer"], n_experts, args.min_activations)
        X = source[r["expert"]]["activations"].float()
        tokens, evr = projection_pursuit(
            X,
            dictionary,
            tok,
            device=device,
            k=args.pc1_pursuit_k,
            pc=1,
        )
        full = pursuit.get((r["layer"], r["expert"]), {})
        r["pursuit_tokens"] = full.get("tokens", [])
        r["pursuit_evr"] = (full.get("evr") or [None])[-1]
        r["pc1_pursuit_tokens"] = tokens
        r["pc1_pursuit_evr"] = evr[-1] if evr else None

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

    poly_rows = [
        (
            f"L{r['layer']}E{r['expert']}",
            f"{r['pc1_evr']:.2f}",
            f"{r['eff_rank']:.0f}",
            *[
                " / ".join(escape(t) for t in r["pc_tokens"][i][:5])
                for i in range(min(3, len(r["pc_tokens"])))
            ],
        )
        for r in poly
    ]
    mono_rows = [
        (
            f"L{r['layer']}E{r['expert']}",
            f"{r['pc1_evr']:.2f}",
            _join_tokens(r["pc_tokens"][0]),
            _join_tokens(r.get("pursuit_tokens", [])),
            "" if r.get("pursuit_evr") is None else f"{r['pursuit_evr']:.3f}",
            _join_tokens(r.get("pc1_pursuit_tokens", [])),
            "" if r.get("pc1_pursuit_evr") is None else f"{r['pc1_pursuit_evr']:.3f}",
            _join_tokens(r["input_tokens"]),
        )
        for r in mono
    ]

    findings_html = "".join(f"<li>{x}</li>" for x in findings)
    body = (
        f"<h2>Key findings</h2><ul>{findings_html}</ul>"
        f"<h2>Figures</h2>{figs_to_html(figs)}"
        "<h2>Polysemantic experts: top-3 PC directions decode to different concepts</h2>"
        + table(
            ["expert", "PC1-EVR", "eff rank", "PC1 tokens", "PC2 tokens", "PC3 tokens"],
            poly_rows,
        )
        + "<h2>Monosemantic experts (Case A): one dominant direction</h2>"
        + table(
            [
                "expert",
                "PC1-EVR",
                "PC1 tokens",
                "pursuit tokens",
                "pursuit EVR",
                "PC1-pursuit tokens",
                "PC1-pursuit EVR",
                "input tokens",
            ],
            mono_rows,
        )
    )
    out = get_analysis_dir(model_name, args.dataset) / "nlp_report.html"
    out.write_text(
        html_page(
            title=f"SOMP & Polysemanticity — {args.dataset}",
            heading="SOMP &amp; expert polysemanticity",
            subtitle=(
                f"{model_name} · {args.dataset} · {len(records)} experts "
                f"(≥{args.min_activations} rows) · each expert's activation subspace "
                "decoded via PC→unembedding projection"
            ),
            body=body,
        )
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
