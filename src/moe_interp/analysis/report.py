"""Build a self-contained report.html from analysis artifacts.

Reads the JSON / npy outputs written by ``pipeline.run_analysis`` and assembles a single
HTML page: auto-generated findings, embedded Plotly figures, and tables. No model or
network access required.
"""

import json
from datetime import date
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from moe_interp.analysis.report_html import fig_html as _fig_html
from moe_interp.analysis.report_html import table as _table
from moe_interp.io.plots import plot_label_grid, plot_scatter_grid


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _fmt(x, nd: int = 3) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def write_report(output_dir: Path, model_name: str, dataset: str) -> Path:
    output_dir = Path(output_dir)
    metrics = json.loads((output_dir / "cluster_metrics.json").read_text())
    recovery = json.loads((output_dir / "activation_recovery.json").read_text())
    labels = json.loads((output_dir / "cluster_labels.json").read_text())
    semantics = _load_jsonl(output_dir / "cluster_semantics.jsonl")
    summaries = _load_jsonl(output_dir / "expert_summaries.jsonl")
    tox_path = output_dir / "toxicity_candidates.json"
    toxicity = json.loads(tox_path.read_text()) if tox_path.exists() else None
    tbc_path = output_dir / "toxicity_by_cluster.json"
    tox_by_cluster = json.loads(tbc_path.read_text()) if tbc_path.exists() else {}
    adp_records = _load_jsonl(output_dir / "adp_results.jsonl")
    cfg_path = output_dir / "run_config.json"
    run_config = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    token_selection = run_config.get("token_selection", "last")

    pc1 = np.load(output_dir / "pc1_evr_matrix.npy")
    erank = np.load(output_dir / "effective_rank_matrix.npy")
    n_layers, n_experts = pc1.shape
    layers = list(range(n_layers))

    # ---- derive findings ----
    ari = [recovery.get(str(L), {}).get("ari") for L in layers]
    nmi = [recovery.get(str(L), {}).get("nmi") for L in layers]
    purity = [recovery.get(str(L), {}).get("purity") for L in layers]
    matched = [recovery.get(str(L), {}).get("matched_accuracy") for L in layers]
    n_active = [metrics.get(str(L), {}).get("n_experts", 0) for L in layers]

    def _primary(L):
        return metrics.get(str(L), {}).get("primary_method")

    sil = [
        (
            metrics.get(str(L), {}).get(_primary(L), {}).get("silhouette")
            if _primary(L)
            else None
        )
        for L in layers
    ]

    ari_valid = [a for a in ari if a is not None]
    pc1_vals = (
        np.array([r["pc1_evr"] for r in summaries]) if summaries else np.array([])
    )
    erank_vals = (
        np.array([r["effective_rank"] for r in summaries])
        if summaries
        else np.array([])
    )

    findings = []
    if ari_valid:
        mean_ari = float(np.mean(ari_valid))
        best_ari_layer = max(
            (L for L in layers if ari[L] is not None), key=lambda L: ari[L]
        )
        if mean_ari > 0.5:
            findings.append(
                f"<b>Activation identity is highly recoverable.</b> Mean ARI across "
                f"layers is {mean_ari:.3f} (1.0 = perfect): routed activations are "
                f"geometrically separable by expert — strongest at layer "
                f"{best_ari_layer} (ARI {ari[best_ari_layer]:.3f}). This validates the "
                f"captures but is a diagnostic, not evidence of semantic specialization."
            )
        else:
            findings.append(
                f"<b>Activation identity is not recoverable at this capture density.</b> "
                f"Mean ARI is only {mean_ari:.3f} — too few rows per expert for "
                f"activation-level structure (this dataset is last-token / sparse). "
                f"Treat its clustering as unreliable and rely on the toxicity pursuit "
                f"signal below."
            )
    findings.append(
        f"<b>The captured token sample touches more experts with depth.</b> The number "
        f"of experts receiving ≥{run_config.get('min_activations', '?')} captured tokens "
        f"goes from {n_active[0]} (layer 0) to {n_active[-1]} (layer {n_layers - 1}). "
        f"This is a property of <i>this</i> token sample, not a routing-entropy measure; "
        f"expert-level clustering is only meaningful where enough experts are active."
    )
    stab = [
        metrics.get(str(L), {}).get("mean_l2_stability", {}).get("mean_ari")
        for L in layers
    ]
    stab_valid = [s for s in stab if s is not None]

    def _rep_best_sil(prefix):
        vals = [
            m["silhouette"]
            for L in layers
            for key, m in metrics.get(str(L), {}).items()
            if isinstance(m, dict)
            and key.startswith(prefix)
            and m.get("silhouette") is not None
        ]
        return max(vals) if vals else None

    mean_sil = _rep_best_sil("mean_l2")
    stats_sil = _rep_best_sil("stats")
    if mean_sil is not None:
        stab_note = (
            f" Bootstrap stability of the direction clustering is mean ARI "
            f"{np.mean(stab_valid):.2f}, so groupings are hypotheses, not stable structure."
            if stab_valid
            else ""
        )
        findings.append(
            f"<b>Direction-based expert clusters are weak; the gains come from magnitude.</b> "
            f"Clustering the activation <i>direction</i> (centered, L2-normalized means) "
            f"tops out at silhouette {_fmt(mean_sil, 2)}. The distributional <i>stats</i> "
            f"representation (count, norm, EVR, effective rank) separates far more cleanly "
            f"(silhouette {_fmt(stats_sil, 2)}), but that mostly sorts experts by activation "
            f"magnitude/frequency, not by what they mean.{stab_note}"
        )
    if pc1_vals.size:
        mono_frac = float((pc1_vals > 0.5).mean())
        findings.append(
            f"<b>Most experts are polysemantic.</b> Mean PC1-EVR is "
            f"{pc1_vals.mean():.3f}; only {mono_frac:.0%} of experts concentrate &gt;50% "
            f"of their variance in a single direction. Mean effective rank "
            f"{erank_vals.mean():.1f}."
        )
    if adp_records:
        idims = [r["intrinsic_dim"] for r in adp_records if "intrinsic_dim" in r]
        npks = [r.get("n_peaks", 0) for r in adp_records]
        uniq = [r["n_unique_tokens"] for r in adp_records if r.get("n_unique_tokens")]
        findings.append(
            f"<b>ADP over-segments on the last-token clouds.</b> Across "
            f"{len(adp_records)} experts (≥{run_config.get('adp_min_rows', '?')} rows) "
            f"ADP finds a mean of {np.mean(npks):.0f} density peaks on only ~242 points, "
            f"and each expert sees few unique source tokens (median "
            f"{int(np.median(uniq)) if uniq else 0}) — so peaks are labelled by "
            f"logit-lens, not source token. Mean intrinsic dimension {np.mean(idims):.1f}. "
            f"Treat ADP here as a wiring check: it needs the denser all-token capture "
            f"(2k–10k rows) to give credible multimodality."
        )
    coherent = sorted(
        [s for s in semantics if s.get("mean_pairwise_jaccard") is not None],
        key=lambda s: s["mean_pairwise_jaccard"],
        reverse=True,
    )
    if coherent:
        top = coherent[0]
        findings.append(
            f"<b>Cluster token-overlap is weak and only semi-supervised.</b> Even the "
            f"most consistent cluster (layer {top['layer']}, {top['n_members']} experts) "
            f"has exact-match top-token Jaccard {top['mean_pairwise_jaccard']:.2f} "
            f"[{', '.join(top['aggregated_top_tokens'][:6])}]. These tokens come from "
            f"SOMP, an annotator, so this is not unsupervised semantic discovery on its "
            f"own."
        )
    by_tok = (toxicity or {}).get("by_offensive_tokens")
    by_evr = (toxicity or {}).get("by_offensive_evr")
    overlap = (toxicity or {}).get("overlap_evr_and_tokens") or []
    if by_tok and by_tok.get("candidates"):
        c0 = by_tok["candidates"][0]
        n_with = by_tok.get("n_experts_with_offensive_token")
        n_scored = by_tok.get("n_experts_scored")
        findings.append(
            f"<b>Toxicity is distributed, not localized.</b> {n_with} of {n_scored} "
            f"experts surface ≥1 offensive word in their top tokens (expected — the RTP "
            f"corpus is all toxic); the strongest, L{c0['layer']}·E{c0['expert']} "
            f"({', '.join(c0['offensive_hits'])}), still reaches only "
            f"{c0['offensive_token_fraction']:.0%}. No single 'toxicity expert'."
        )
    if by_evr and by_evr.get("candidates"):
        e0 = by_evr["candidates"][0]
        findings.append(
            f"<b>Restricted-vocab EVR flags a different set.</b> The expert most explained "
            f"by toxic-word directions is L{e0['layer']}·E{e0['expert']} "
            f"(offensive EVR {_fmt(e0['offensive_evr'])}, z={_fmt(e0.get('evr_zscore'), 1)} "
            f"above baseline). {len(overlap)} expert(s) rank highly on <i>both</i> EVR and "
            f"literal toxic tokens — those are the strongest causal-test candidates."
        )

    # ---- figures ----
    figs = []

    rec_fig = go.Figure()
    for name, series in [
        ("ARI", ari),
        ("NMI", nmi),
        ("purity", purity),
        ("matched acc", matched),
    ]:
        rec_fig.add_trace(
            go.Scatter(x=layers, y=series, mode="lines+markers", name=name)
        )
    rec_fig.update_layout(
        title="Activation-level expert recovery by layer",
        xaxis_title="layer",
        yaxis_title="score",
        plot_bgcolor="white",
        height=400,
    )
    figs.append(rec_fig)

    sil_fig = go.Figure()
    sil_fig.add_trace(
        go.Bar(x=layers, y=n_active, name="active experts", yaxis="y2", opacity=0.3)
    )
    sil_fig.add_trace(
        go.Scatter(x=layers, y=sil, mode="lines+markers", name="silhouette")
    )
    sil_fig.update_layout(
        title="Expert-cluster silhouette &amp; active-expert count by layer",
        xaxis_title="layer",
        yaxis=dict(title="silhouette"),
        yaxis2=dict(title="active experts", overlaying="y", side="right"),
        plot_bgcolor="white",
        height=400,
    )
    figs.append(sil_fig)

    if pc1_vals.size:
        hist = go.Figure(go.Histogram(x=pc1_vals, nbinsx=20))
        hist.update_layout(
            title="Distribution of per-expert PC1-EVR (monosemanticity)",
            xaxis_title="PC1 EVR",
            yaxis_title="experts",
            plot_bgcolor="white",
            height=350,
        )
        figs.append(hist)

    figs.append(
        plot_scatter_grid(pc1, "Per-expert PC1-EVR (layer × expert)", "pc1_evr")
    )
    figs.append(
        plot_scatter_grid(
            erank, "Per-expert effective rank (layer × expert)", "eff_rank"
        )
    )

    grid_records = []
    for L in layers:
        primary = _primary(L)
        lbls = labels.get(str(L), {}).get(primary, {}) if primary else {}
        for e_str, cid in lbls.items():
            grid_records.append(
                {
                    "layer": L,
                    "expert": int(e_str),
                    "labels": [f"c{cid}"],
                    "evr": [float(pc1[L, int(e_str)])],
                    "tokens": [],
                    "n_activations": 0,
                }
            )
    if grid_records:
        figs.append(
            plot_label_grid(grid_records, n_layers=n_layers, n_experts=n_experts)
        )

    npeaks_path = output_dir / "adp_n_peaks_matrix.npy"
    if adp_records and npeaks_path.exists():
        figs.append(
            plot_scatter_grid(
                np.load(npeaks_path),
                "ADP density peaks per expert (layer × expert)",
                "n_peaks",
            )
        )

    figs_html = "".join(_fig_html(f, i == 0) for i, f in enumerate(figs))

    # ---- tables ----
    per_layer_rows = []
    for L in layers:
        p = _primary(L)
        mp = metrics.get(str(L), {}).get(p, {}) if p else {}
        per_layer_rows.append(
            [
                L,
                n_active[L],
                p or "—",
                _fmt(mp.get("k")),
                _fmt(sil[L]),
                _fmt(stab[L], 2),
                _fmt(ari[L]),
                _fmt(matched[L]),
                _fmt(purity[L]),
            ]
        )
    per_layer_table = _table(
        [
            "layer",
            "active experts",
            "best (rep.method)",
            "k",
            "silhouette",
            "bootstrap ARI",
            "act. ARI",
            "matched acc",
            "purity",
        ],
        per_layer_rows,
    )

    coherent_rows = [
        [
            s["layer"],
            s.get("cluster_id"),
            s["n_members"],
            _fmt(s["mean_pairwise_jaccard"], 2),
            _fmt(s.get("mean_final_evr"), 3),
            ", ".join(s["aggregated_top_tokens"][:8]),
        ]
        for s in coherent[:12]
    ]
    coherent_table = _table(
        ["layer", "cluster", "members", "Jaccard", "mean EVR", "shared top tokens"],
        coherent_rows,
    )

    tox_section = ""
    if toxicity:
        sections = []

        if by_tok and by_tok.get("candidates"):
            rows = [
                [
                    i + 1,
                    c["layer"],
                    c["expert"],
                    _fmt(c["offensive_token_fraction"], 2),
                    ", ".join(c.get("offensive_hits") or []),
                    _fmt(c.get("offensive_evr"), 3),
                    ", ".join((c.get("top_tokens") or [])[:8]),
                ]
                for i, c in enumerate(by_tok["candidates"])
            ]
            sections.append(
                f"<h3>Ranked by offensive token presence (full vocab)</h3>"
                f"<p>{by_tok['n_experts_with_offensive_token']} of "
                f"{by_tok['n_experts_scored']} experts have ≥1 offensive top token; "
                f"baseline mean fraction {_fmt(by_tok['baseline_mean_fraction'])}.</p>"
                + _table(
                    [
                        "rank",
                        "layer",
                        "expert",
                        "off. frac",
                        "hits",
                        "off. EVR",
                        "top tokens",
                    ],
                    rows,
                )
            )

        if by_evr and by_evr.get("candidates"):
            rows = [
                [
                    i + 1,
                    c["layer"],
                    c["expert"],
                    _fmt(c.get("offensive_evr"), 3),
                    _fmt(c.get("evr_zscore"), 1),
                    ", ".join(c.get("offensive_hits") or []) or "—",
                    ", ".join((c.get("top_tokens") or [])[:8]),
                ]
                for i, c in enumerate(by_evr["candidates"])
            ]
            sections.append(
                f"<h3>Ranked by restricted-vocab offensive EVR</h3>"
                f"<p>How much each expert is explained by toxic-word directions. Baseline "
                f"mean {_fmt(by_evr['baseline_mean'])} ± {_fmt(by_evr['baseline_std'])}. "
                f"High EVR with no literal toxic tokens usually means a generic subword "
                f"expert — read alongside the table above.</p>"
                + _table(
                    [
                        "rank",
                        "layer",
                        "expert",
                        "off. EVR",
                        "z",
                        "literal hits",
                        "top tokens",
                    ],
                    rows,
                )
            )

        if overlap:
            rows = [
                [
                    c["layer"],
                    c["expert"],
                    _fmt(c.get("offensive_evr"), 3),
                    _fmt(c["offensive_token_fraction"], 2),
                    ", ".join(c.get("offensive_hits") or []),
                ]
                for c in overlap
            ]
            sections.append(
                "<h3>Strongest candidates (high on both signals)</h3>"
                + _table(["layer", "expert", "off. EVR", "off. frac", "hits"], rows)
            )

        # Toxicity × clustering: where do the literal-toxic experts land?
        tox_experts = (by_tok or {}).get("candidates", [])[:12]
        link_rows = []
        for c in tox_experts:
            Lk = str(c["layer"])
            p = metrics.get(Lk, {}).get("primary_method")
            cid = labels.get(Lk, {}).get(p, {}).get(str(c["expert"])) if p else None
            cl = tox_by_cluster.get(Lk, {}).get("clusters", {}).get(str(cid), {})
            link_rows.append(
                [
                    c["layer"],
                    c["expert"],
                    ", ".join(c.get("offensive_hits") or []),
                    "—" if cid is None else f"c{cid}",
                    cl.get("n_experts", "—"),
                    _fmt(cl.get("mean_offensive_fraction"), 3),
                    ", ".join(str(e) for e in cl.get("offensive_experts", [])),
                ]
            )
        link_table = _table(
            [
                "layer",
                "expert",
                "hits",
                "cluster",
                "cluster size",
                "cluster mean off. frac",
                "other toxic in cluster",
            ],
            link_rows,
        )
        sections.append(
            "<h3>Toxicity × clustering</h3>"
            "<p>Where each top toxic expert lands in its layer's primary clustering — do "
            "toxic experts co-cluster, or scatter? (Clustering on this corpus is weak, so "
            "read as a starting point for the correlation, not a result.)</p>"
            + link_table
        )

        tox_section = "<h2>Toxicity candidate experts</h2>" + "".join(sections)

    adp_section = ""
    if adp_records:
        multimodal = sorted(
            [r for r in adp_records if r.get("n_peaks", 0) >= 2],
            key=lambda r: r["n_peaks"],
            reverse=True,
        )

        def _peak_label(p):
            toks = p.get("lens_tokens") or p.get("top_tokens") or []
            return f"{p['size']}×[{', '.join(t for t in toks[:5])}]"

        rows = []
        for r in multimodal[:15]:
            peak_desc = " &nbsp;|&nbsp; ".join(
                _peak_label(p) for p in r.get("peaks", [])[:4]
            )
            rows.append(
                [
                    r["layer"],
                    r["expert"],
                    r["n_rows"],
                    r.get("n_unique_tokens", "—"),
                    r["n_peaks"],
                    _fmt(r.get("intrinsic_dim"), 1),
                    peak_desc,
                ]
            )
        idims = [r["intrinsic_dim"] for r in adp_records if "intrinsic_dim" in r]
        npks = [r.get("n_peaks", 0) for r in adp_records]
        body = (
            _table(
                [
                    "layer",
                    "expert",
                    "rows",
                    "uniq tokens",
                    "peaks",
                    "intr. dim",
                    "peak size × logit-lens label",
                ],
                rows,
            )
            if rows
            else "<p>No expert showed ≥2 significant density peaks.</p>"
        )
        adp_section = (
            f"<h2>ADP manifold analysis (per-expert multimodality)</h2>"
            f"<p>DADApy Advanced Density Peak clustering (kstarNN density) on each "
            f"well-populated expert's activation rows ({len(adp_records)} experts with "
            f"≥{run_config.get('adp_min_rows', '?')} rows; mean {np.mean(npks):.0f} peaks, "
            f"mean intrinsic dim {np.mean(idims):.1f}). <b>Caveat:</b> with only ~242 "
            f"points per expert ADP over-segments, and each expert's rows share very few "
            f"source tokens (this is a last-token capture), so peaks are labelled by "
            f"<i>logit-lens of the peak centroid</i> rather than source token. This is a "
            f"validated wiring of the method (it cleanly separates synthetic modes in the "
            f"tests); credible multimodality needs the denser all-token capture.</p>"
            + body
        )

    findings_html = "".join(f"<li>{f}</li>" for f in findings)
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>MoE Unsupervised Analysis — {dataset}</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1100px;
margin:2rem auto;padding:0 1rem;color:#1a1a1a;line-height:1.5}}
h1{{margin-bottom:0}} .sub{{color:#666;margin-top:.25rem}}
h2{{margin-top:2.5rem;border-bottom:2px solid #eee;padding-bottom:.3rem}}
table{{border-collapse:collapse;width:100%;margin:1rem 0;font-size:.86rem}}
th,td{{border:1px solid #ddd;padding:.35rem .5rem;text-align:left}}
th{{background:#f5f7fa}} tr:nth-child(even){{background:#fafbfc}}
li{{margin-bottom:.6rem}} .caveat{{background:#fff8e6;border-left:4px solid #f0c000;
padding:.6rem 1rem;border-radius:4px}}
</style></head><body>
<h1>Unsupervised Expert Analysis</h1>
<p class="sub">{model_name} · dataset: <b>{dataset}</b> · {n_layers} layers ×
{n_experts} experts · generated {date.today().isoformat()}</p>
<h2>Key findings</h2><ul>{findings_html}</ul>
<p class="caveat">⚠ Small-corpus, <b>{token_selection}-token</b> exploratory results.
Clusters, ADP peaks, and toxicity hits are candidates for stability + causal validation,
not conclusions. The semantic layer (SOMP tokens, offensive lexicon) is a
<b>semi-supervised annotator</b> — the toxic word list is noisy and conflates toxicity
with topic and generic negativity. Last-token captures describe answer-position
behavior, not broad specialization.</p>
<h2>Figures</h2>{figs_html}
<h2>Per-layer metrics</h2>{per_layer_table}
{adp_section}
<h2>Most coherent clusters</h2>{coherent_table}
{tox_section}
</body></html>"""

    report_path = output_dir / "report.html"
    report_path.write_text(html)
    return report_path
