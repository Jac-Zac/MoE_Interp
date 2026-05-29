"""Orchestration for the unsupervised expert-analysis pipeline.

Runs on existing last-token HDF5 captures (no model forward passes). For each layer it
computes per-expert summaries, clusters the experts, runs an activation-level recovery
diagnostic, and interprets the resulting clusters semantically by reusing the
precomputed SOMP pursuit results. For toxicity-oriented datasets it additionally ranks
experts by offensive-concept EVR against controls.

These corpora are small, so every output here is candidate-generation, not proof of
specialization — clusters still need stability and causal validation downstream.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import torch

# Spectral clustering on small/sparse cosine affinities legitimately hits this; we
# handle disconnected components by silhouette-selecting across methods.
warnings.filterwarnings("ignore", message="Graph is not fully connected")
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from moe_interp.analysis.adp import adp_expert
from moe_interp.analysis.clustering import (
    DEFAULT_METHODS,
    bootstrap_mean_clustering_stability,
    cluster_activations,
    cluster_layer_experts,
)
from moe_interp.analysis.decode import (
    cluster_semantic_coherence,
    load_pursuit_results,
    top_tokens_for_vector,
)
from moe_interp.analysis.normalize import normalize_features
from moe_interp.analysis.summaries import compute_expert_summary
from moe_interp.capture.cache import load_layer_h5, load_metadata, load_unembedding
from moe_interp.config import get_data_dir, get_model_dir, get_unembedding_dir
from moe_interp.io.plots import plot_label_grid, plot_scatter_grid
from moe_interp.pursuit.concepts import CONCEPT_WORDS

FEATURE_NORM = "layer_centered+l2"


def _resolve_pursuit_dir(
    model_name: str, dataset: str, explicit: Path | None
) -> Path | None:
    """Find precomputed pursuit results, preferring an explicit path, then the standard
    location, then the synced Orfeo-cluster results."""
    safe = get_model_dir(model_name).name
    candidates = []
    if explicit is not None:
        candidates.append(Path(explicit))
    candidates.append(get_model_dir(model_name) / "pursuit" / dataset)
    candidates.append(get_data_dir() / "orfeo" / safe / "pursuit" / dataset)
    for c in candidates:
        if (c / "results.jsonl").exists():
            return c
    return None


def _feature_sets(summaries: dict) -> tuple[list[int], dict[str, torch.Tensor]]:
    """Build the per-expert feature representations to cluster.

    Two complementary views (critique: don't cluster the mean alone):
    * ``mean_l2``: layer-centered, L2-normalized mean vectors — the activation
      *direction* (cosine geometry).
    * ``stats``: standardized [pc1_evr, effective_rank, mean_norm, row_norm_std,
      log10 count] — the distributional *shape*, independent of direction.
    """
    expert_ids = sorted(summaries)
    if not expert_ids:
        return [], {}
    means = torch.stack([summaries[ei].mean for ei in expert_ids])
    stats = torch.tensor(
        [
            [
                summaries[ei].pc1_evr,
                summaries[ei].effective_rank,
                summaries[ei].mean_norm,
                summaries[ei].row_norm_std,
                float(np.log10(max(summaries[ei].count, 1))),
            ]
            for ei in expert_ids
        ],
        dtype=torch.float32,
    )
    return expert_ids, {
        "mean_l2": normalize_features(means, FEATURE_NORM),
        "stats": normalize_features(stats, "standardized"),
    }


def _best_primary(clusters_by_rep: dict, methods) -> str | None:
    """Pick the (representation.method) with the highest silhouette for this layer."""
    best, best_sil = None, -np.inf
    for rep, lc in clusters_by_rep.items():
        for m in methods:
            sil = lc.get(m, {}).get("silhouette")
            if sil is not None and sil > best_sil:
                best, best_sil = f"{rep}.{m}", sil
    return best


def _try_load_tokenizer(model_name: str):
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_name)
    except Exception as exc:  # noqa: BLE001 - decoding is optional
        print(f"Tokenizer unavailable (token decoding disabled): {exc}")
        return None


def _try_load_unembedding(model_name: str):
    dict_path = get_unembedding_dir(model_name) / "dictionary.h5"
    if not dict_path.exists():
        return None
    try:
        return load_unembedding(dict_path).float()
    except Exception as exc:  # noqa: BLE001 - logit lens is optional
        print(f"Logit-lens disabled (could not load unembedding): {exc}")
        return None


def run_analysis(
    extractions_dir: Path,
    output_dir: Path,
    model_name: str,
    dataset: str,
    min_activations: int = 20,
    methods=DEFAULT_METHODS,
    pursuit_dir: Path | None = None,
    seed: int = 1337,
    top_k: int = 20,
    logit_lens: bool = True,
    adp: bool = False,
    adp_min_rows: int = 100,
    n_bootstrap: int = 20,
) -> dict:
    """Run the full unsupervised analysis and write artifacts to output_dir."""
    extractions_dir = Path(extractions_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = list(methods)

    metadata = load_metadata(extractions_dir / "metadata.json")
    n_layers = metadata["n_layers"]
    n_experts = metadata["n_experts"]
    token_selection = metadata.get("token_selection", "last")

    resolved_pursuit = _resolve_pursuit_dir(model_name, dataset, pursuit_dir)
    pursuit = load_pursuit_results(resolved_pursuit) if resolved_pursuit else {}
    if resolved_pursuit:
        print(f"Loaded {len(pursuit)} pursuit records from {resolved_pursuit}")
    else:
        print("No precomputed pursuit found; semantic coherence will be limited.")

    tokenizer = _try_load_tokenizer(model_name) if (logit_lens or adp) else None
    unembedding = _try_load_unembedding(model_name) if (logit_lens or adp) else None

    pc1_matrix = np.zeros((n_layers, n_experts))
    erank_matrix = np.zeros((n_layers, n_experts))
    norm_matrix = np.zeros((n_layers, n_experts))
    count_matrix = np.zeros((n_layers, n_experts))
    npeaks_matrix = np.zeros((n_layers, n_experts))
    idim_matrix = np.zeros((n_layers, n_experts))

    summary_records: list[dict] = []
    cluster_labels: dict[str, dict] = {}
    cluster_metrics: dict[str, dict] = {}
    activation_recovery: dict[str, dict] = {}
    semantic_records: list[dict] = []
    label_grid_records: list[dict] = []
    adp_records: list[dict] = []
    adp_decode_cache: dict[int, str] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Analyzing layers", total=n_layers)

        for layer in range(n_layers):
            full = load_layer_h5(extractions_dir, layer, n_experts, min_activations)
            expert_acts = {ei: entry["activations"] for ei, entry in full.items()}
            summaries = {
                ei: compute_expert_summary(entry["activations"], layer, ei)
                for ei, entry in full.items()
            }
            for ei, s in summaries.items():
                summary_records.append(s.to_record())
                pc1_matrix[layer, ei] = s.pc1_evr
                erank_matrix[layer, ei] = s.effective_rank
                norm_matrix[layer, ei] = s.mean_norm
                count_matrix[layer, ei] = s.count

            expert_ids, feature_sets = _feature_sets(summaries)

            # Expert-level clustering under each feature representation.
            clusters_by_rep = {
                rep: cluster_layer_experts(feats, methods=methods, seed=seed)
                for rep, feats in feature_sets.items()
            }
            primary = _best_primary(clusters_by_rep, methods)

            labels_entry: dict[str, dict] = {}
            metrics_entry: dict = {
                "n_experts": len(expert_ids),
                "skipped": not bool(expert_ids),
                "primary_method": primary,
            }
            for rep, lc in clusters_by_rep.items():
                for m in methods:
                    if m not in lc:
                        continue
                    key = f"{rep}.{m}"
                    labels_entry[key] = {
                        str(expert_ids[i]): int(c)
                        for i, c in enumerate(lc[m].get("labels", []))
                    }
                    metrics_entry[key] = {
                        k: lc[m].get(k)
                        for k in (
                            "k",
                            "silhouette",
                            "davies_bouldin",
                            "calinski_harabasz",
                        )
                    }

            # Bootstrap stability of the geometric (mean-direction) clustering.
            mean_best = _best_primary(
                {"mean_l2": clusters_by_rep.get("mean_l2", {})}, methods
            )
            if mean_best and expert_acts:
                k = metrics_entry.get(mean_best, {}).get("k") or 2
                metrics_entry["mean_l2_stability"] = (
                    bootstrap_mean_clustering_stability(
                        expert_acts, k=k, n_bootstrap=n_bootstrap, seed=seed
                    )
                )

            cluster_labels[str(layer)] = labels_entry
            cluster_metrics[str(layer)] = metrics_entry

            # Activation-level recovery diagnostic.
            if expert_acts:
                rows = torch.cat([expert_acts[ei] for ei in sorted(expert_acts)])
                true_ids = torch.cat(
                    [
                        torch.full((expert_acts[ei].shape[0],), ei)
                        for ei in sorted(expert_acts)
                    ]
                )
                activation_recovery[str(layer)] = cluster_activations(
                    rows, true_ids, seed=seed
                )

            # Per-expert ADP manifold analysis (multimodality of activation clouds).
            if adp:
                for ei, entry in full.items():
                    rec = adp_expert(
                        entry["activations"],
                        token_ids=entry.get("tokens"),
                        tokenizer=tokenizer,
                        unembedding=unembedding,
                        layer=layer,
                        expert=ei,
                        min_rows=adp_min_rows,
                        seed=seed,
                        decode_cache=adp_decode_cache,
                    )
                    if rec.get("skipped") or rec.get("error"):
                        continue
                    adp_records.append(rec)
                    npeaks_matrix[layer, ei] = rec["n_peaks"]
                    idim_matrix[layer, ei] = rec["intrinsic_dim"]

            # Semantic interpretation + label grid for the primary clustering.
            primary_labels = None
            if primary:
                rep, m = primary.split(".", 1)
                primary_labels = clusters_by_rep[rep][m].get("labels")
            cluster_of = dict(zip(expert_ids, primary_labels)) if primary_labels else {}
            for ei in expert_ids:
                cid = cluster_of.get(ei)
                label_grid_records.append(
                    {
                        "layer": layer,
                        "expert": ei,
                        "labels": [f"c{cid}"] if cid is not None else ["other"],
                        "evr": [summaries[ei].pc1_evr],
                        "tokens": (pursuit.get((layer, ei), {}) or {}).get(
                            "tokens", []
                        ),
                        "n_activations": summaries[ei].count,
                    }
                )

            if primary_labels is not None:
                for cid in sorted(set(primary_labels)):
                    members = [ei for ei in expert_ids if cluster_of[ei] == cid]
                    member_keys = [(layer, ei) for ei in members]
                    coherence = cluster_semantic_coherence(
                        member_keys, pursuit, top_n=10
                    )
                    record = {
                        "layer": layer,
                        "method": primary,
                        "cluster_id": int(cid),
                        "members": members,
                        "n_members": len(members),
                        **coherence,
                    }
                    if unembedding is not None and tokenizer is not None:
                        centroid = torch.stack(
                            [summaries[ei].mean for ei in members]
                        ).mean(dim=0)
                        record["centroid_top_tokens"] = top_tokens_for_vector(
                            centroid, unembedding, tokenizer, k=top_k
                        )
                    semantic_records.append(record)

            progress.advance(task)

    # Write artifacts.
    _write_json(
        output_dir / "run_config.json",
        {
            "model_name": model_name,
            "dataset": dataset,
            "token_selection": token_selection,
            "min_activations": min_activations,
            "methods": methods,
            "adp": adp,
            "adp_min_rows": adp_min_rows,
            "n_bootstrap": n_bootstrap,
        },
    )
    _write_jsonl(output_dir / "expert_summaries.jsonl", summary_records)
    _write_json(output_dir / "cluster_labels.json", cluster_labels)
    _write_json(output_dir / "cluster_metrics.json", cluster_metrics)
    _write_json(output_dir / "activation_recovery.json", activation_recovery)
    _write_jsonl(output_dir / "cluster_semantics.jsonl", semantic_records)
    if adp:
        _write_jsonl(output_dir / "adp_results.jsonl", adp_records)
        np.save(output_dir / "adp_n_peaks_matrix.npy", npeaks_matrix)
        np.save(output_dir / "adp_intrinsic_dim_matrix.npy", idim_matrix)

    np.save(output_dir / "pc1_evr_matrix.npy", pc1_matrix)
    np.save(output_dir / "effective_rank_matrix.npy", erank_matrix)
    np.save(output_dir / "mean_norm_matrix.npy", norm_matrix)
    np.save(output_dir / "count_matrix.npy", count_matrix)

    plot_scatter_grid(
        pc1_matrix,
        "Per-expert PC1 EVR (monosemantic ↔ polysemantic)",
        "pc1_evr",
        output_path=output_dir / "pc1_evr_heatmap.html",
    )
    plot_scatter_grid(
        erank_matrix,
        "Per-expert effective rank (participation ratio)",
        "eff_rank",
        output_path=output_dir / "effective_rank_heatmap.html",
    )
    plot_label_grid(
        label_grid_records,
        n_layers=n_layers,
        n_experts=n_experts,
        output_path=output_dir / "cluster_grid.html",
    )

    toxicity = None
    if resolved_pursuit is not None:
        signals = _offensive_signals(resolved_pursuit, pursuit)
        toxicity = _toxicity_candidates(signals, top_n=20)
        if toxicity is not None:
            _write_json(output_dir / "toxicity_candidates.json", toxicity)
            tox_by_cluster = _toxicity_by_cluster(
                cluster_labels, cluster_metrics, signals
            )
            _write_json(output_dir / "toxicity_by_cluster.json", tox_by_cluster)

    print(f"Saved analysis to {output_dir}")
    return {
        "n_summaries": len(summary_records),
        "n_semantic_clusters": len(semantic_records),
        "n_adp_experts": len(adp_records),
        "activation_recovery": activation_recovery,
        "toxicity_candidates": toxicity,
        "output_dir": str(output_dir),
    }


def _offensive_signals(pursuit_dir: Path, general_pursuit: dict) -> dict:
    """Per-expert toxicity signals keyed by (layer, expert).

    * ``off_evr``: final EVR of the offensive-concept-restricted pursuit — "how much of
      this expert's activation is explained by toxic-word directions". The signal the
      user emphasized; meaningful where it is large *and* significant above baseline.
    * ``off_frac`` / ``off_hits``: fraction and identity of an expert's full-vocab SOMP
      top tokens that are offensive words — interpretable, but high baseline on RTP
      because the whole corpus is toxic, so it must be read against controls.
    """
    offensive = load_pursuit_results(pursuit_dir / "offensive")
    off_evr = (
        {k: float(r["evr"][-1]) for k, r in offensive.items() if r.get("evr")}
        if offensive
        else {}
    )
    concept_set = {w.strip().lower() for w in CONCEPT_WORDS["offensive"]}
    off_frac, off_hits, top_tokens = {}, {}, {}
    for key, r in general_pursuit.items():
        toks = [t.strip().lower() for t in (r.get("tokens") or [])[:20]]
        if not toks:
            continue
        off_frac[key] = sum(1 for t in toks if t in concept_set) / len(toks)
        off_hits[key] = sorted({t for t in toks if t in concept_set})
        top_tokens[key] = (r.get("tokens") or [])[:10]
    return {
        "off_evr": off_evr,
        "off_frac": off_frac,
        "off_hits": off_hits,
        "top_tokens": top_tokens,
    }


def _toxicity_candidates(signals: dict, top_n: int = 20) -> dict | None:
    """Rank toxic experts two ways and report the overlap.

    Ranking by restricted-vocab offensive EVR finds experts most explained by toxic
    directions; ranking by full-vocab offensive-token presence finds experts that
    literally decode to toxic words. Experts strong on *both* are the best candidates.
    """
    off_evr = signals["off_evr"]
    off_frac = signals["off_frac"]
    off_hits = signals["off_hits"]
    top_tokens = signals["top_tokens"]
    if not off_evr and not off_frac:
        return None

    def _row(key):
        layer, expert = key
        return {
            "layer": layer,
            "expert": expert,
            "offensive_evr": off_evr.get(key),
            "offensive_token_fraction": off_frac.get(key, 0.0),
            "offensive_hits": off_hits.get(key, []),
            "top_tokens": top_tokens.get(key, []),
        }

    result: dict = {}
    if off_evr:
        vals = np.array(list(off_evr.values()))
        mean, std = float(vals.mean()), float(vals.std())
        by_evr = sorted(off_evr, key=lambda k: off_evr[k], reverse=True)[:top_n]
        result["by_offensive_evr"] = {
            "baseline_mean": mean,
            "baseline_std": std,
            "candidates": [
                {
                    **_row(k),
                    "evr_zscore": ((off_evr[k] - mean) / std if std > 0 else None),
                }
                for k in by_evr
            ],
        }
    if off_frac:
        fr = np.array(list(off_frac.values()))
        by_frac = sorted(
            off_frac, key=lambda k: (off_frac[k], off_evr.get(k, 0.0)), reverse=True
        )[:top_n]
        result["by_offensive_tokens"] = {
            "baseline_mean_fraction": float(fr.mean()),
            "n_experts_with_offensive_token": int((fr > 0).sum()),
            "n_experts_scored": len(off_frac),
            "candidates": [_row(k) for k in by_frac],
        }
    if off_evr and off_frac:
        evr_top = set(sorted(off_evr, key=lambda k: off_evr[k], reverse=True)[:top_n])
        frac_top = {k for k in off_frac if off_frac[k] > 0}
        overlap = sorted(evr_top & frac_top, key=lambda k: off_evr[k], reverse=True)
        result["overlap_evr_and_tokens"] = [_row(k) for k in overlap]
    return result


def _toxicity_by_cluster(
    cluster_labels: dict, cluster_metrics: dict, signals: dict
) -> dict:
    """Join per-expert toxicity onto each layer's primary clustering.

    Lets us ask the synthesis question: do high-toxicity experts land in the same
    cluster? Reports per-cluster mean offensive EVR / token fraction and which members
    decode to offensive words.
    """
    off_evr = signals["off_evr"]
    off_frac = signals["off_frac"]
    off_hits = signals["off_hits"]
    out: dict = {}
    for layer_key, methods in cluster_labels.items():
        primary = cluster_metrics.get(layer_key, {}).get("primary_method")
        labels = methods.get(primary) if primary else None
        if not labels:
            continue
        layer = int(layer_key)
        buckets: dict[str, dict] = {}
        for e_str, cid in labels.items():
            key = (layer, int(e_str))
            b = buckets.setdefault(str(cid), {"experts": [], "evr": [], "frac": []})
            b["experts"].append(int(e_str))
            if key in off_evr:
                b["evr"].append(off_evr[key])
            if key in off_frac:
                b["frac"].append(off_frac[key])
        clusters = {
            cid: {
                "n_experts": len(b["experts"]),
                "experts": b["experts"],
                "mean_offensive_evr": float(np.mean(b["evr"])) if b["evr"] else None,
                "mean_offensive_fraction": (
                    float(np.mean(b["frac"])) if b["frac"] else None
                ),
                "offensive_experts": [
                    e for e in b["experts"] if off_hits.get((layer, e))
                ],
            }
            for cid, b in buckets.items()
        }
        out[layer_key] = {"primary_method": primary, "clusters": clusters}
    return out


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2))


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
