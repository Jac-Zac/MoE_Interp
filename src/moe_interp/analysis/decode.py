"""Semantic interpretation of experts and expert clusters.

A geometric cluster is only credible when geometry, decoded tokens, and pursuit EVR
point in the same direction. This module supplies the semantic side: a logit-lens
projection of cluster means onto the vocabulary, plus helpers that reuse the already
computed SOMP pursuit results (top tokens + EVR per expert) to score the coherence of a
cluster and the concept-enrichment of an expert set.
"""

from collections import Counter
from pathlib import Path

import numpy as np
import torch

from moe_interp.pursuit import load_pursuit

PursuitMap = dict[tuple[int, int], dict]


def _norm_token(token: str) -> str:
    return token.strip().lower()


def top_tokens_for_vector(
    vec: torch.Tensor,
    unembedding: torch.Tensor,
    tokenizer,
    k: int = 20,
) -> list[str]:
    """Logit-lens: project ``vec`` onto the unembedding and decode the top-k tokens."""
    scores = unembedding.float() @ vec.float()
    k = min(k, scores.shape[0])
    idx = torch.topk(scores, k).indices.tolist()
    return [tokenizer.decode([i]) for i in idx]


def load_pursuit_results(pursuit_dir: Path) -> PursuitMap:
    """Load precomputed SOMP results, keyed by (layer, expert).

    Returns an empty map if no results.jsonl exists in the directory.
    """
    pursuit_dir = Path(pursuit_dir)
    if not (pursuit_dir / "results.jsonl").exists():
        return {}
    results, _, _ = load_pursuit(pursuit_dir)
    return {(r["layer"], r["expert"]): r for r in results}


def cluster_semantic_coherence(
    member_keys: list[tuple[int, int]],
    pursuit: PursuitMap,
    top_n: int = 10,
) -> dict:
    """Measure how semantically coherent a cluster's member experts are.

    Uses each member's top SOMP tokens: mean pairwise Jaccard overlap of the token
    sets, mean final EVR, and the most common tokens aggregated across members.
    """
    token_sets: list[set[str]] = []
    evrs: list[float] = []
    all_tokens: list[str] = []
    for key in member_keys:
        r = pursuit.get(key)
        if not r:
            continue
        toks = [_norm_token(t) for t in (r.get("tokens") or [])[:top_n]]
        if toks:
            token_sets.append(set(toks))
            all_tokens.extend(toks)
        evr = r.get("evr") or []
        if evr:
            evrs.append(float(evr[-1]))

    n = len(token_sets)
    if n >= 2:
        sims = []
        for i in range(n):
            for j in range(i + 1, n):
                union = token_sets[i] | token_sets[j]
                inter = token_sets[i] & token_sets[j]
                sims.append(len(inter) / len(union) if union else 0.0)
        mean_jaccard = float(np.mean(sims))
    else:
        mean_jaccard = None

    aggregated = [t for t, _ in Counter(all_tokens).most_common(top_n)]
    return {
        "n_members_with_pursuit": n,
        "mean_pairwise_jaccard": mean_jaccard,
        "mean_final_evr": float(np.mean(evrs)) if evrs else None,
        "aggregated_top_tokens": aggregated,
    }


def concept_scores(
    pursuit: PursuitMap,
    concept_words,
    top_n: int = 20,
) -> dict[tuple[int, int], float]:
    """Per-expert fraction of top SOMP tokens that fall in a concept word set."""
    concept_set = {w.strip().lower() for w in concept_words}
    scores: dict[tuple[int, int], float] = {}
    for key, r in pursuit.items():
        toks = [_norm_token(t) for t in (r.get("tokens") or [])[:top_n]]
        if not toks:
            scores[key] = 0.0
            continue
        hits = sum(1 for t in toks if t in concept_set)
        scores[key] = hits / len(toks)
    return scores


def concept_enrichment(
    member_keys: list[tuple[int, int]],
    scores: dict[tuple[int, int], float],
    control_keys: list[tuple[int, int]],
) -> dict:
    """Compare a member set's mean concept score against a control set's mean."""
    member = [scores[k] for k in member_keys if k in scores]
    control = [scores[k] for k in control_keys if k in scores]
    member_mean = float(np.mean(member)) if member else 0.0
    control_mean = float(np.mean(control)) if control else 0.0
    enrichment = member_mean / control_mean if control_mean > 0 else None
    return {
        "member_mean": member_mean,
        "control_mean": control_mean,
        "enrichment": enrichment,
    }
