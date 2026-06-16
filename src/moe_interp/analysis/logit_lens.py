"""Bulk mean-projection logit-lens baseline and SOMP-vs-lens comparison.

The *logit lens* reads an activation by projecting it onto the unembedding and taking
the top tokens. Aggregated over an expert's stored activations, the natural bulk
baseline is the mean direction::

    scores = D @ mean(A)          # one direction, ranked tokens

SOMP instead selects a *basis* of dictionary atoms that maximise explained variance
of the (centred) activations. To compare the two on equal footing, we reconstruct the
same centred activations from each method's selected atoms via least squares and read
off the cumulative explained-variance ratio (EVR). Both EVR curves are computed with
the identical estimator used inside ``pursuit.decomposition.somp`` (sum of per-dim
recon variance / sum of per-dim total variance), so they are directly comparable to
the EVR stored in ``results.jsonl``.

The headline this comparison exposes: the single best logit-lens token explains
little variance, while SOMP's first few atoms explain much more — i.e. a single
top-k token ranking under-reads a polysemantic expert.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from moe_interp.analysis.common import (
    iter_expert_activations,
    load_analysis_inputs,
    load_somp_results,
)

# Cumulative-EVR cut-offs reported per expert and aggregated in the summary.
EVR_AT = (1, 3, 10)


def mean_projection_scores(A: torch.Tensor, dictionary: torch.Tensor) -> torch.Tensor:
    """Logit-lens scores over the vocabulary for an expert's mean activation."""
    return dictionary @ A.mean(dim=0)


def cumulative_evr(A: torch.Tensor, atoms: torch.Tensor) -> list[float]:
    """Cumulative EVR of centred ``A`` reconstructed from a prefix of ``atoms``.

    ``atoms`` is ``(m, d)`` (rows of the unembedding). Returns a length-``m`` list where
    entry ``i`` is the EVR using ``atoms[: i + 1]``. Orthonormalising the atoms (QR) makes
    the prefix subspaces nested — ``span(atoms[:m]) == span(Q[:, :m])`` — so the explained
    variance is just the running sum of squared projections onto each orthonormal
    direction. That is the same orthogonal-projection EVR the per-prefix least-squares
    solve produced, but in one QR + matmul instead of ``m`` separate lstsq solves.
    """
    Ac = (A - A.mean(dim=0, keepdim=True)).double().cpu()
    total = (Ac**2).sum().clamp_min(1e-12)  # == n * total variance
    q, _ = torch.linalg.qr(atoms.double().cpu().T, mode="reduced")  # (d, m) orthonormal
    coords = Ac @ q  # (n, m): centred rows along each orthonormal atom direction
    explained = torch.cumsum((coords**2).sum(dim=0), dim=0)  # (m,)
    return (explained / total).tolist()


def _evr_at(evr: list[float], idx: int) -> float:
    """Cumulative EVR after ``idx`` atoms (1-based), clamped to what's available."""
    if not evr:
        return 0.0
    return float(evr[min(idx, len(evr)) - 1])


def compare_expert(
    A: torch.Tensor,
    dictionary: torch.Tensor,
    tokenizer,
    somp_tokens: list[str],
    somp_evr: list[float],
    k: int = 10,
) -> dict:
    """Compare the mean-projection logit lens against SOMP for one expert."""
    scores = mean_projection_scores(A, dictionary)
    lens_idx = torch.topk(scores, k).indices.tolist()
    lens_tokens = [tokenizer.decode([i]) for i in lens_idx]
    lens_evr = cumulative_evr(A, dictionary[lens_idx])

    a = {t.strip() for t in lens_tokens[:k]}
    b = {t.strip() for t in somp_tokens[:k]}
    jaccard = len(a & b) / len(a | b) if (a | b) else 0.0

    return {
        "lens_tokens": lens_tokens,
        "lens_evr": lens_evr,
        "somp_tokens": somp_tokens[:k],
        "somp_evr": [_evr_at(somp_evr, i) for i in range(1, k + 1)],
        "jaccard_topk": jaccard,
    }


def run_logit_lens_comparison(
    model_name: str,
    dataset: str,
    k: int = 10,
    min_activations: int = 50,
    max_rows: int | None = 2000,
    extractions_dir: Path | str | None = None,
    pursuit_dir: Path | str | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Run the mean-projection logit lens vs SOMP over every analyzable expert.

    ``extractions_dir`` / ``pursuit_dir`` default to the standard local layout; pass
    explicit paths to read from elsewhere. Returns a summary dict and (if ``output_dir``
    set) writes ``logit_lens_comparison.json`` with per-expert records and aggregates.
    """
    from transformers import AutoTokenizer

    extractions_dir, metadata, dictionary, pursuit_dir = load_analysis_inputs(
        model_name, dataset, extractions_dir, pursuit_dir
    )
    results_file = pursuit_dir / "results.jsonl"
    if not results_file.exists():
        raise FileNotFoundError(
            f"No SOMP results at {results_file}. Run `python main.py pursuit` first, "
            "or pass --pursuit_dir pointing at a directory that holds results.jsonl."
        )
    somp = load_somp_results(pursuit_dir)
    tokenizer = AutoTokenizer.from_pretrained(metadata["model_name"])

    records: list[dict] = []
    for layer, expert, A in iter_expert_activations(
        extractions_dir,
        metadata["n_layers"],
        metadata["n_experts"],
        min_activations=min_activations,
        max_rows=max_rows,
    ):
        r = somp.get((layer, expert))
        if r is None:
            continue
        cmp = compare_expert(
            A, dictionary, tokenizer, r.get("tokens", []), r.get("evr", []), k=k
        )
        cmp |= {"layer": layer, "expert": expert, "n_activations": int(A.shape[0])}
        records.append(cmp)

    def _mean_evr_at(key: str, i: int) -> float:
        vals = [_evr_at(r[key], i) for r in records]
        return float(np.mean(vals)) if vals else 0.0

    summary = {
        "model_name": model_name,
        "dataset": dataset,
        "n_experts_compared": len(records),
        "k": k,
        "mean_jaccard_topk": (
            float(np.mean([r["jaccard_topk"] for r in records])) if records else 0.0
        ),
    }
    for i in EVR_AT:
        summary[f"mean_lens_evr_{i}"] = _mean_evr_at("lens_evr", i)
        summary[f"mean_somp_evr_{i}"] = _mean_evr_at("somp_evr", i)
    out = {"summary": summary, "experts": records}

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "logit_lens_comparison.json").write_text(
            json.dumps(out, indent=2)
        )
    return out
