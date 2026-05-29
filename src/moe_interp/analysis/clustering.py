"""Layerwise clustering of MoE experts.

Two complementary questions:

* **Expert-level** (``cluster_layer_experts``): do the per-expert summary vectors within
  one layer fall into geometric groups? This is the candidate-generation step.
* **Activation-level** (``cluster_activations``): are individual routed activations
  geometrically separable by expert id at all? This is a diagnostic — high recovery
  means the captured outputs are distinct, not that experts are semantically specialized.

Clustering happens per layer because representation geometry changes with depth, so
experts in different layers are not directly exchangeable.
"""

import warnings

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

DEFAULT_METHODS = ("kmeans", "agglomerative", "spectral")
_EPS = 1e-8


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    # Sparse captures can contain non-finite float16 values; scrub them so the
    # downstream sklearn estimators (which reject NaN/inf) stay happy.
    return np.nan_to_num(
        np.asarray(x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
    )


def _l2_rows(X: np.ndarray) -> np.ndarray:
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), _EPS, None)


def _fit_labels(
    method: str, X: np.ndarray, k: int, seed: int, affinity: np.ndarray
) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message="Graph is not fully connected")
        if method == "kmeans":
            return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
        if method == "agglomerative":
            return AgglomerativeClustering(
                n_clusters=k, metric="cosine", linkage="average"
            ).fit_predict(X)
        if method == "spectral":
            return SpectralClustering(
                n_clusters=k,
                affinity="precomputed",
                random_state=seed,
                assign_labels="kmeans",
            ).fit_predict(affinity)
    raise ValueError(f"Unknown clustering method '{method}'")


def _empty_result(n: int, methods) -> dict:
    skipped = {
        "labels": [0] * n,
        "k": 1,
        "silhouette": None,
        "davies_bouldin": None,
        "calinski_harabasz": None,
    }
    return {m: dict(skipped) for m in methods}


def cluster_layer_experts(
    features: torch.Tensor | np.ndarray,
    methods=DEFAULT_METHODS,
    k_range: list[int] | None = None,
    seed: int = 1337,
    min_experts: int = 4,
) -> dict:
    """Cluster the per-expert feature rows of a single layer.

    Args:
        features: (n_experts, n_features) — already normalized by the caller.
        methods: clustering algorithms to run.
        k_range: candidate cluster counts; defaults to [2 .. min(8, n_experts-1)].
        min_experts: layers with fewer active experts are skipped (clustering a
            handful of points is unstable and uninformative).

    Returns a dict keyed by method, each with the silhouette-selected ``labels``, ``k``,
    and internal metrics (silhouette/davies_bouldin/calinski_harabasz). Also includes
    ``n_experts`` and ``skipped``.
    """
    X = _to_numpy(features)
    n = X.shape[0]
    out: dict = {"n_experts": n}
    if n < min_experts:
        out["skipped"] = True
        out.update(_empty_result(n, methods))
        return out
    out["skipped"] = False

    kmax = min(8, n - 1)
    if k_range is None:
        ks = list(range(2, kmax + 1))
    else:
        ks = [k for k in k_range if 2 <= k <= n - 1]

    X_l2 = _l2_rows(X)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        affinity = np.clip(X_l2 @ X_l2.T, 0.0, None)
    affinity = np.nan_to_num(affinity, nan=0.0, posinf=0.0, neginf=0.0)

    for method in methods:
        best: dict | None = None
        for k in ks:
            try:
                labels = _fit_labels(method, X, k, seed, affinity)
            except Exception:
                continue
            if len(set(labels.tolist())) < 2:
                continue
            sil = float(silhouette_score(X, labels, metric="cosine"))
            if best is None or sil > best["silhouette"]:
                best = {"labels": labels.tolist(), "k": int(k), "silhouette": sil}
        if best is None:
            out[method] = _empty_result(n, [method])[method]
        else:
            labels = np.asarray(best["labels"])
            best["davies_bouldin"] = float(davies_bouldin_score(X, labels))
            best["calinski_harabasz"] = float(calinski_harabasz_score(X, labels))
            out[method] = best
    return out


def _kmeans_on_means(
    expert_acts: dict, ids: list[int], k: int, seed: int
) -> np.ndarray:
    means = np.stack([_to_numpy(expert_acts[ei]).mean(axis=0) for ei in ids])
    means = means - means.mean(axis=0, keepdims=True)
    means = means / np.clip(np.linalg.norm(means, axis=1, keepdims=True), _EPS, None)
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(means)


def bootstrap_mean_clustering_stability(
    expert_acts: dict,
    k: int,
    n_bootstrap: int = 20,
    seed: int = 1337,
) -> dict:
    """Stability of the mean-direction clustering under row resampling.

    For each bootstrap we resample each expert's activation rows with replacement,
    recompute its mean, recluster, and score agreement (ARI) with the full-data
    labels. Low mean ARI ⇒ the grouping is an artifact of sampling noise. Addresses
    the critique that clustering metrics should be treated as hypotheses until
    bootstrapped.
    """
    ids = sorted(expert_acts)
    if len(ids) < 4 or k < 2:
        return {"mean_ari": None, "std_ari": None, "n_bootstrap": 0}
    reference = _kmeans_on_means(expert_acts, ids, k, seed)
    rng = np.random.default_rng(seed)
    aris = []
    for b in range(n_bootstrap):
        resampled = {}
        for ei in ids:
            a = expert_acts[ei]
            n = a.shape[0]
            resampled[ei] = a[rng.integers(0, n, size=n)]
        labels = _kmeans_on_means(resampled, ids, k, seed + b + 1)
        aris.append(adjusted_rand_score(reference, labels))
    return {
        "mean_ari": float(np.mean(aris)),
        "std_ari": float(np.std(aris)),
        "n_bootstrap": n_bootstrap,
    }


def cluster_activations(
    rows: torch.Tensor | np.ndarray,
    true_expert_ids: torch.Tensor | np.ndarray,
    seed: int = 1337,
) -> dict:
    """Diagnostic: cluster individual activation rows and recover expert id.

    KMeans (k = number of distinct experts) on L2-normalized rows, then Hungarian-match
    clusters to experts. Reports adjusted Rand index, normalized mutual information,
    Hungarian-matched accuracy, and cluster purity.
    """
    X = _l2_rows(_to_numpy(rows))
    y = _to_numpy(true_expert_ids).astype(int)
    uniq = sorted(set(y.tolist()))
    k = len(uniq)

    result = {"n_rows": int(X.shape[0]), "n_experts": k}
    if k < 2 or X.shape[0] <= k:
        result.update(
            ari=None, nmi=None, matched_accuracy=None, purity=None, skipped=True
        )
        return result

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        pred = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)

    expert_to_col = {e: i for i, e in enumerate(uniq)}
    contingency = np.zeros((k, k), dtype=np.int64)
    for p, t in zip(pred.tolist(), y.tolist()):
        contingency[p, expert_to_col[t]] += 1

    row_ind, col_ind = linear_sum_assignment(-contingency)
    matched = contingency[row_ind, col_ind].sum()
    n_rows = len(y)

    result.update(
        skipped=False,
        ari=float(adjusted_rand_score(y, pred)),
        nmi=float(normalized_mutual_info_score(y, pred)),
        matched_accuracy=float(matched / n_rows),
        purity=float(contingency.max(axis=1).sum() / n_rows),
    )
    return result
