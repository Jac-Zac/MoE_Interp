"""Tests for the unsupervised analysis modules."""

import json

import h5py
import torch

from moe_interp.analysis.adp import adp_expert
from moe_interp.analysis.clustering import cluster_activations, cluster_layer_experts
from moe_interp.analysis.decode import (
    cluster_semantic_coherence,
    concept_scores,
    top_tokens_for_vector,
)
from moe_interp.analysis.normalize import normalize_features
from moe_interp.analysis.pipeline import run_analysis
from moe_interp.analysis.report import write_report
from moe_interp.analysis.summaries import compute_expert_summary
from moe_interp.capture.cache import append_to_file, save_metadata


class _DummyTokenizer:
    def decode(self, token_ids):
        return f"tok_{token_ids[0]}"


def _blobs(n_clusters=3, per=40, d=16, spread=0.05, seed=0):
    """Well-separated gaussian blobs with labels."""
    torch.manual_seed(seed)
    centers = torch.eye(n_clusters, d) * 5.0
    rows, labels = [], []
    for c in range(n_clusters):
        rows.append(centers[c] + spread * torch.randn(per, d))
        labels += [c] * per
    return torch.cat(rows), torch.tensor(labels)


def test_compute_expert_summary_shapes_and_bounds():
    torch.manual_seed(0)
    acts = torch.randn(50, 16)
    s = compute_expert_summary(acts, layer=2, expert=7, top_pcs=3)

    assert s.layer == 2 and s.expert == 7
    assert s.count == 50
    assert s.mean.shape == (16,)
    assert s.top_pc_directions.shape == (3, 16)
    assert 0.0 <= s.pc1_evr <= 1.0
    assert 1.0 <= s.effective_rank <= 16.0 + 1e-6
    assert set(s.to_record()) >= {
        "layer",
        "expert",
        "count",
        "pc1_evr",
        "effective_rank",
    }


def test_summary_single_row_is_degenerate():
    s = compute_expert_summary(torch.randn(1, 8), layer=0, expert=0)
    assert s.count == 1
    assert s.pc1_evr == 0.0
    assert s.row_norm_std == 0.0


def test_pc1_evr_high_for_rank_one_data():
    base = torch.randn(1, 16)
    acts = torch.randn(80, 1) @ base  # rank-1 (plus the mean) → one dominant direction
    s = compute_expert_summary(acts, layer=0, expert=0)
    assert s.pc1_evr > 0.9


def test_normalize_l2_unit_rows():
    X = torch.randn(10, 5)
    out = normalize_features(X, "l2")
    assert torch.allclose(out.norm(dim=1), torch.ones(10), atol=1e-5)


def test_normalize_composes_center_then_l2():
    X = torch.randn(8, 4)
    out = normalize_features(X, "layer_centered+l2")
    assert torch.allclose(out.norm(dim=1), torch.ones(8), atol=1e-5)


def test_cluster_layer_experts_separates_blobs():
    rows, _ = _blobs(n_clusters=3, per=4, d=16)  # 12 "experts" in 3 groups
    feats = normalize_features(rows, "layer_centered+l2")
    out = cluster_layer_experts(feats, methods=("kmeans",), seed=0)

    assert not out["skipped"]
    km = out["kmeans"]
    assert len(km["labels"]) == rows.shape[0]
    assert km["k"] == 3
    assert km["silhouette"] is not None and km["silhouette"] > 0.5


def test_cluster_layer_experts_skips_tiny_layers():
    out = cluster_layer_experts(torch.randn(3, 8), min_experts=4)
    assert out["skipped"] is True
    assert out["kmeans"]["labels"] == [0, 0, 0]


def test_cluster_activations_recovers_well_separated():
    rows, labels = _blobs(n_clusters=4, per=50, d=16, spread=0.02)
    out = cluster_activations(rows, labels, seed=0)

    assert not out["skipped"]
    assert out["n_experts"] == 4
    assert out["ari"] > 0.95
    assert out["matched_accuracy"] > 0.95


def test_top_tokens_for_vector_returns_k():
    unembedding = torch.eye(6)
    vec = torch.tensor([0.1, 0.9, 0.2, 0.05, 0.0, 0.3])
    toks = top_tokens_for_vector(vec, unembedding, _DummyTokenizer(), k=3)
    assert toks == ["tok_1", "tok_5", "tok_2"]


def test_cluster_semantic_coherence_overlap():
    pursuit = {
        (0, 1): {"tokens": ["war", "death", "blood"], "evr": [0.2, 0.4]},
        (0, 2): {"tokens": ["war", "death", "peace"], "evr": [0.3, 0.5]},
    }
    out = cluster_semantic_coherence([(0, 1), (0, 2)], pursuit, top_n=3)
    assert out["n_members_with_pursuit"] == 2
    assert (
        out["mean_pairwise_jaccard"] == 0.5
    )  # {war,death} shared of {war,death,blood,peace}
    assert out["mean_final_evr"] == 0.45


def test_concept_scores_fraction():
    pursuit = {(0, 0): {"tokens": ["war", "cat", "hate", "dog"]}}
    scores = concept_scores(pursuit, ["war", "hate"], top_n=4)
    assert scores[(0, 0)] == 0.5


def test_adp_expert_detects_multimodality():
    torch.manual_seed(0)
    X = torch.cat([torch.randn(80, 16) + 8.0, torch.randn(80, 16) - 8.0])
    toks = torch.cat(
        [torch.zeros(80, dtype=torch.long), torch.ones(80, dtype=torch.long)]
    )
    rec = adp_expert(X, token_ids=toks, layer=0, expert=0, min_rows=50, maxk=50)
    assert not rec.get("skipped") and not rec.get("error")
    assert rec["n_rows"] == 160
    assert rec["n_peaks"] >= 2
    assert all("top_tokens" in p for p in rec["peaks"])


def test_adp_expert_skips_sparse():
    rec = adp_expert(torch.randn(10, 16), min_rows=50)
    assert rec["skipped"] is True


def _make_extractions(tmp_path, n_layers=2, n_experts=6, d=16, per=40, seed=0):
    torch.manual_seed(seed)
    ed = tmp_path / "extractions"
    ed.mkdir(parents=True)
    save_metadata(
        ed,
        model_name="test/dummy",
        dataset_name="triviaqa",
        n_layers=n_layers,
        n_experts=n_experts,
        d_model=d,
        token_selection="last",
    )
    centers = torch.randn(n_experts, d) * 6.0
    for L in range(n_layers):
        with h5py.File(ed / f"layer_{L:02d}.h5", "w") as f:
            for e in range(n_experts):
                acts = (centers[e] + 0.1 * torch.randn(per, d)).half()
                toks = torch.randint(0, 100, (per,))
                append_to_file(f, e, acts, toks)
    return ed


def test_run_analysis_end_to_end_and_report(tmp_path):
    ed = _make_extractions(tmp_path)
    out = tmp_path / "out"
    result = run_analysis(
        ed,
        out,
        model_name="test/dummy",
        dataset="triviaqa",
        min_activations=5,
        n_bootstrap=3,
        logit_lens=False,
    )
    for fn in [
        "expert_summaries.jsonl",
        "cluster_labels.json",
        "cluster_metrics.json",
        "activation_recovery.json",
        "pc1_evr_matrix.npy",
        "run_config.json",
    ]:
        assert (out / fn).exists(), f"missing artifact {fn}"

    # Well-separated blobs → activation identity should be recoverable.
    recovery = json.loads((out / "activation_recovery.json").read_text())
    assert recovery["0"]["ari"] > 0.8
    assert result["n_summaries"] > 0

    # No pursuit available for a dummy model → toxicity is skipped gracefully.
    assert not (out / "toxicity_candidates.json").exists()

    report_path = write_report(out, "test/dummy", "triviaqa")
    assert report_path.exists()
    assert "Key findings" in report_path.read_text()
