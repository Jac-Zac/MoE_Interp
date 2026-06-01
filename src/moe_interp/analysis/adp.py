"""Advanced Density Peak (ADP) manifold analysis of single experts.

Where the layerwise clustering asks "which experts group together", ADP asks a
different question of one expert at a time: does its cloud of routed activations form
several density peaks? Multiple peaks are evidence of *multimodality* — a necessary (not
sufficient) condition for polysemantic behavior. We attach the captured source tokens to
each peak so a mode can be read ("this expert has a dates mode and a names mode"), but a
peak is only a semantic hypothesis until token metadata and causal tests agree.

Runs on individual activation rows (not expert means), per the unsupervised plan. Needs
enough rows to estimate density, so sparse experts are skipped.
"""

import warnings

import numpy as np
import torch

# dadapy ships a couple of escape-sequence SyntaxWarnings in its plotting helpers.
warnings.filterwarnings("ignore", category=SyntaxWarning, module="dadapy")


def _decode_one(token_id: int, tokenizer, cache: dict | None) -> str:
    if tokenizer is None:
        return str(token_id)
    if cache is not None and token_id in cache:
        return cache[token_id]
    text = tokenizer.decode([token_id])
    if cache is not None:
        cache[token_id] = text
    return text


def _decode_top(
    token_ids: np.ndarray, mask: np.ndarray, tokenizer, k: int, cache: dict | None
) -> list[str]:
    ids, counts = np.unique(token_ids[mask], return_counts=True)
    order = np.argsort(-counts)[:k]
    return [_decode_one(int(ids[i]), tokenizer, cache) for i in order]


def _lens_tokens(
    centroid: np.ndarray, unembedding, tokenizer, k: int, cache: dict | None
):
    """Logit-lens label for a peak: project its centroid onto the unembedding.

    More informative than source tokens for last-token captures, where every row of an
    expert often shares one template source token but the activations still differ.
    """
    scores = unembedding @ torch.from_numpy(centroid).float()
    idx = torch.topk(scores, min(k, scores.shape[0])).indices.tolist()
    return [_decode_one(int(i), tokenizer, cache) for i in idx]


def adp_expert(
    acts: torch.Tensor,
    token_ids: torch.Tensor | np.ndarray | None = None,
    tokenizer=None,
    layer: int = -1,
    expert: int = -1,
    unembedding=None,
    min_rows: int = 100,
    maxk: int = 100,
    z: float = 3.0,
    seed: int = 1337,
    top_tokens: int = 6,
    max_decoded_peaks: int = 12,
    decode_cache: dict | None = None,
) -> dict:
    """Run ADP on one expert's activation matrix (n, d).

    Returns intrinsic dimension, the number of density peaks, and per-peak size +
    top source tokens. Marks ``skipped`` when there are too few rows, or ``error`` if
    the density estimation fails on degenerate input.
    """
    from dadapy.data import Data

    rng = np.random.default_rng(seed)
    X = np.nan_to_num(acts.detach().cpu().float().numpy()).astype(np.float64)
    n = X.shape[0]
    base = {"layer": layer, "expert": expert, "n_rows": int(n)}
    if n < min_rows:
        return {**base, "skipped": True}

    ids_arr = None
    if token_ids is not None:
        ids_arr = (
            token_ids.detach().cpu().numpy()
            if isinstance(token_ids, torch.Tensor)
            else np.asarray(token_ids)
        )

    # Break exact duplicate rows (common with quantized float16 captures): identical
    # points give zero nearest-neighbour distances, which crash density estimation.
    std = float(X.std())
    if std > 0:
        X = X + rng.normal(0.0, std * 1e-4, size=X.shape)

    try:
        data = Data(X, verbose=False)
        data.compute_distances(maxk=min(maxk, X.shape[0] - 1))
        id_est, _, _ = data.compute_id_2NN()
        # kstarNN (adaptive k-NN) density rather than PAk: PAk runs an iterative
        # likelihood optimization that can fail to converge and hang on certain
        # degenerate activation clouds; kstarNN is closed-form, fast, and robust.
        data.compute_density_kstarNN()
        labels = np.asarray(data.compute_clustering_ADP(Z=z))
    except Exception as exc:  # noqa: BLE001 - ADP can fail on degenerate clouds
        return {**base, "skipped": False, "error": str(exc)[:200]}

    intrinsic_dim = float(np.mean(np.atleast_1d(id_est)))
    valid = sorted({int(c) for c in labels if c >= 0})
    # Sort peaks by size first so we only spend decode calls on the largest ones.
    sized = sorted(
        ((c, int((labels == c).sum())) for c in valid), key=lambda x: x[1], reverse=True
    )
    peaks = []
    for rank, (c, size) in enumerate(sized):
        peak = {"cluster": c, "size": size}
        if rank < max_decoded_peaks:
            mask = labels == c
            if ids_arr is not None:
                peak["top_tokens"] = _decode_top(
                    ids_arr, mask, tokenizer, top_tokens, decode_cache
                )
            if unembedding is not None:
                peak["lens_tokens"] = _lens_tokens(
                    X[mask].mean(axis=0),
                    unembedding,
                    tokenizer,
                    top_tokens,
                    decode_cache,
                )
        peaks.append(peak)

    n_unique_tokens = int(len(np.unique(ids_arr))) if ids_arr is not None else None

    return {
        **base,
        "skipped": False,
        "intrinsic_dim": intrinsic_dim,
        "n_peaks": len(valid),
        "n_unique_tokens": n_unique_tokens,
        "halo_fraction": float((labels < 0).mean()),
        "peaks": peaks,
    }
