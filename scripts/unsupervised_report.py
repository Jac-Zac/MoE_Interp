#!/usr/bin/env python
"""Unsupervised exploration of MoE experts (ACTIVATIONS only) → self-contained HTML.

Exam framing (notes/unsupervised.md, notes/to_think.md): treat each (layer, expert) as a
point whose feature is built ONLY from its captured activations, and ask what
unsupervised structure lives among experts. SOMP tokens are used purely as human-readable
LABELS for clusters, never as clustering inputs.

Per-expert activation feature = layer-centered + L2 mean-activation direction, optionally
concatenated with spectrum-shape features (PC1-EVR, effective rank, norm stats). Analyses:

  1. Intrinsic dimension of the expert population (TwoNN) vs ambient d_model.
  2. Hierarchical clustering per layer (cosine) — dendrogram heatmap + "twin" experts.
  3. HDBSCAN density clustering per layer — dense families vs outliers, noise vs depth.
  4. Per-expert ADP density peaks (already computed) as a multimodality map.

Run: DATA_DIR=./data .venv/bin/python scripts/unsupervised_report.py --dataset pile10k
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from moe_interp.analysis.report_html import (
    figs_to_html,
    html_page,
    load_pursuit_map,
    table,
)
from moe_interp.analysis.summaries import compute_expert_summary
from moe_interp.capture.cache import load_layer_h5, load_metadata
from moe_interp.config import get_analysis_dir, get_default_model, get_model_dir


def _label(members, pursuit, layer, top=4):
    """Human-readable cluster label = most common SOMP tokens across members (display only)."""
    toks = []
    for e in members:
        toks += [t.strip() for t in pursuit.get((layer, e), [])[:8]]
    common = [t for t, _ in Counter(toks).most_common(top) if t]
    return ", ".join(common) if common else "—"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pile10k")
    ap.add_argument("--model", default=None)
    ap.add_argument("--min_activations", type=int, default=50)
    args = ap.parse_args()

    model_name = args.model or get_default_model()
    ed = get_model_dir(model_name) / "extractions" / args.dataset
    meta = load_metadata(ed / "metadata.json")
    n_layers, n_experts = meta["n_layers"], meta["n_experts"]
    # SOMP tokens are display LABELS only — never clustering inputs.
    pursuit = {
        ke: r.get("tokens", [])
        for ke, r in load_pursuit_map(model_name, args.dataset).items()
    }
    analysis_dir = get_analysis_dir(model_name, args.dataset)

    # ---- per-layer activation features ----
    # dirs: layer-centered + L2 mean direction (for cosine clustering / TwoNN).
    # feats: [mean direction (scaled) | standardized spectrum scalars] (for HDBSCAN).
    print("Loading expert activation features per layer ...")
    layer_dirs: dict[int, tuple[list[int], np.ndarray]] = {}
    layer_feats: dict[int, np.ndarray] = {}
    for L in range(n_layers):
        full = load_layer_h5(ed, L, n_experts, args.min_activations)
        if len(full) < 4:
            continue
        ids = sorted(full)
        # Activation-only feature per expert (shared summary): mean direction +
        # spectrum-shape scalars [pc1_evr, effective_rank, mean_norm, row_norm_std].
        means, shapes = [], []
        for e in ids:
            s = compute_expert_summary(full[e]["activations"], L, e, top_pcs=1)
            means.append(s.mean.numpy())
            shapes.append(
                np.array([s.pc1_evr, s.effective_rank, s.mean_norm, s.row_norm_std])
            )
        means = np.stack(means)
        means = means - means.mean(0, keepdims=True)  # layer-center
        dirs = means / np.clip(np.linalg.norm(means, axis=1, keepdims=True), 1e-8, None)
        layer_dirs[L] = (ids, dirs.astype(np.float64))
        # combined feature: unit mean direction + standardized spectrum scalars
        shapes_z = StandardScaler().fit_transform(np.stack(shapes))
        layer_feats[L] = np.hstack([dirs, shapes_z]).astype(np.float64)
    print(f"  {len(layer_dirs)} layers with >=4 active experts")

    figs: list[go.Figure] = []
    findings: list[str] = []

    # ============ 1. Intrinsic dimension (TwoNN) of the expert mean-direction cloud =======
    alldirs = np.concatenate([m for _, m in layer_dirs.values()])
    id_val = None
    try:
        from dadapy.data import Data

        d = Data(alldirs, verbose=False)
        d.compute_distances(maxk=min(100, alldirs.shape[0] - 1))
        id_twonn, _, _ = d.compute_id_2NN()
        id_val = float(np.mean(np.atleast_1d(id_twonn)))
        findings.append(
            f"<b>Experts live on a low-dimensional manifold.</b> The {alldirs.shape[0]} "
            f"expert mean-activation vectors sit in {alldirs.shape[1]}-D residual space, but "
            f"their <b>TwoNN intrinsic dimension is {id_val:.1f}</b> — the expert population "
            f"is far simpler than the ambient space."
        )
    except Exception as exc:  # noqa: BLE001
        print("TwoNN failed:", exc)

    # ============ 2. Hierarchical clustering per layer (cosine) + twins ==================
    # Representative layer = best cosine silhouette (where activation structure is real).
    def _layer_sil(L):
        lids, lM = layer_dirs[L]
        if len(lids) < 8:
            return -1.0
        Zl = linkage(pdist(lM, metric="cosine"), method="average")
        cut = fcluster(Zl, t=max(2, len(lids) // 8), criterion="maxclust")
        return silhouette_score(lM, cut, metric="cosine") if len(set(cut)) > 1 else -1.0

    rep_layer = max(layer_dirs, key=_layer_sil)
    ids, M = layer_dirs[rep_layer]
    dist = pdist(M, metric="cosine")
    Z = linkage(dist, method="average")
    labels_h = [
        f"E{e}:{(pursuit.get((rep_layer, e), ['?'])[:1] or ['?'])[0][:8]}" for e in ids
    ]
    order = dendrogram(Z, labels=labels_h, no_plot=True)["leaves"]
    D = squareform(dist)
    hm = go.Figure(
        go.Heatmap(
            z=D[np.ix_(order, order)],
            x=[labels_h[i] for i in order],
            y=[labels_h[i] for i in order],
            colorscale="Viridis_r",
            colorbar=dict(title="cosine dist"),
        )
    )
    hm.update_layout(
        title=f"Layer {rep_layer}: experts reordered by hierarchical clustering of "
        f"mean activation (dark blocks = expert families)",
        height=700,
        width=750,
    )
    figs.append(hm)
    cut = fcluster(Z, t=max(2, len(ids) // 8), criterion="maxclust")
    sil = silhouette_score(M, cut, metric="cosine") if len(set(cut)) > 1 else None
    findings.append(
        f"<b>Hierarchical clustering of mean activations finds weak-but-present groups.</b> "
        f"At the most-structured layer ({rep_layer}) the cosine-silhouette cut reaches "
        f"{f'{sil:.2f}' if sil else '—'}; the dendrogram heatmap shows a few tight expert "
        f"blocks against a diffuse background. Cluster labels (SOMP tokens) are shown for "
        f"interpretation only — clustering itself uses activations."
    )

    # nearest-neighbour "twin" experts (closest by mean-activation direction)
    np.fill_diagonal(D, np.inf)
    nn = D.argmin(1)
    twin_rows, seen = [], set()
    for dval, a, b in sorted(
        (D[i, nn[i]], ids[i], ids[nn[i]]) for i in range(len(ids))
    ):
        if (b, a) in seen:
            continue
        seen.add((a, b))
        twin_rows.append(
            (
                a,
                b,
                f"{dval:.3f}",
                ", ".join(pursuit.get((rep_layer, a), [])[:4]),
                ", ".join(pursuit.get((rep_layer, b), [])[:4]),
            )
        )
        if len(twin_rows) >= 8:
            break

    # ============ 3. HDBSCAN per layer on the combined activation feature ===============
    hdb_layers, hdb_noise, hdb_nclust, hdb_by_layer = [], [], [], {}
    for L in sorted(layer_feats):
        lids = layer_dirs[L][0]
        lab = HDBSCAN(min_cluster_size=3, min_samples=1).fit_predict(layer_feats[L])
        hdb_layers.append(L)
        hdb_noise.append(int((lab == -1).sum()) / len(lab))
        hdb_nclust.append(len(set(lab) - {-1}))
        hdb_by_layer[L] = (lids, lab)
    rep_hdb_layer = max(hdb_by_layer, key=lambda L: len(set(hdb_by_layer[L][1]) - {-1}))
    lids, lab = hdb_by_layer[rep_hdb_layer]

    noise_fig = go.Figure()
    noise_fig.add_trace(go.Bar(x=hdb_layers, y=hdb_nclust, name="# HDBSCAN families"))
    noise_fig.add_trace(
        go.Scatter(
            x=hdb_layers,
            y=[f * max(hdb_nclust or [1]) for f in hdb_noise],
            name="noise fraction (scaled)",
            yaxis="y2",
            mode="lines+markers",
        )
    )
    noise_fig.update_layout(
        title="HDBSCAN expert families & outlier fraction by layer "
        "(features: mean activation + spectrum shape)",
        xaxis_title="layer",
        yaxis=dict(title="# families"),
        yaxis2=dict(title="noise fraction", overlaying="y", side="right", range=[0, 1]),
        height=400,
    )
    figs.append(noise_fig)

    hdb_rows = []
    for c in sorted(set(lab) - {-1}):
        members = [lids[i] for i in range(len(lids)) if lab[i] == c]
        hdb_rows.append(
            (
                c,
                len(members),
                ", ".join(str(m) for m in members[:10]),
                _label(members, pursuit, rep_hdb_layer),
            )
        )
    n_out = int((lab == -1).sum())
    findings.append(
        f"<b>Density clustering recovers a handful of expert families per layer.</b> "
        f"HDBSCAN on the activation feature (mean direction + spectrum shape) finds, at its "
        f"densest layer ({rep_hdb_layer}), {len(hdb_rows)} families with {n_out}/{len(lids)} "
        f"experts left as density outliers — most experts are generalists; a minority form "
        f"tight activation-defined groups."
    )

    # ============ 4. ADP multimodality map (already computed by the pipeline) ===========
    npk_path = analysis_dir / "adp_n_peaks_matrix.npy"
    if npk_path.exists():
        npk = np.load(npk_path)
        heat = go.Figure(
            go.Heatmap(z=npk, colorscale="Inferno", colorbar=dict(title="ADP peaks"))
        )
        heat.update_layout(
            title="Per-expert ADP density peaks (layer × expert) — multimodality map",
            xaxis_title="expert",
            yaxis_title="layer",
            height=450,
        )
        figs.append(heat)
        mm = (npk >= 2).mean()
        findings.append(
            f"<b>Most experts are multimodal.</b> ADP density-peak clustering on the raw "
            f"activation clouds finds ≥2 peaks in {mm:.0%} of experts (mean "
            f"{npk[npk > 0].mean():.1f}) — an expert's activations are not a single blob but "
            f"several density modes."
        )

    # ---------------- assemble HTML ----------------
    findings_html = "".join(f"<li>{x}</li>" for x in findings)
    body = (
        f"<h2>Key findings</h2><ul>{findings_html}</ul>"
        f"<h2>Figures</h2>{figs_to_html(figs)}"
        f"<h2>“Twin” experts (nearest neighbours by mean activation, layer {rep_layer})</h2>"
        "<p>The two closest experts by mean-activation direction; SOMP tokens shown to "
        "read what each pair represents.</p>"
        + table(
            [
                "expert A",
                "expert B",
                "cosine dist",
                "A tokens (label)",
                "B tokens (label)",
            ],
            twin_rows,
        )
        + f"<h2>HDBSCAN activation families (layer {rep_hdb_layer})</h2>"
        + table(["family", "size", "members", "label (SOMP tokens)"], hdb_rows)
    )
    out = analysis_dir / "unsupervised_report.html"
    out.write_text(
        html_page(
            title=f"Unsupervised Expert Structure — {args.dataset}",
            heading="Unsupervised structure among MoE experts",
            subtitle=(
                f"{model_name} · {args.dataset} · {n_layers}×{n_experts} experts · "
                "clustered on ACTIVATIONS (SOMP tokens are labels only) · "
                f"intrinsic dim (TwoNN) = {f'{id_val:.1f}' if id_val else '—'} · "
                f"direction silhouette (L{rep_layer}) = {f'{sil:.2f}' if sil else '—'}"
            ),
            body=body,
        )
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
