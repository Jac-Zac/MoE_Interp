"""Standard logit-lens baseline and SOMP-vs-lens comparison.

The *logit lens* reads an activation by projecting it onto the unembedding and taking
the top tokens. Aggregated over an expert's stored activations, the natural baseline is
the mean direction::

    scores = D @ mean(A)          # one direction, ranked tokens

That is a single direction. Its top-1 token names one unembedding row; we measure how
much of the expert's activation variance that one direction captures (EVR), using the
same estimator as ``pursuit.decomposition.somp`` (squared projection / total variance),
so the number is directly comparable to the EVR stored in ``results.jsonl``.

SOMP instead selects a *basis* of dictionary atoms that maximise explained variance.
The comparison is deliberately asymmetric: the logit lens reads along one direction
picked by mean alignment, while SOMP keeps adding variance-maximising atoms. The
headline this exposes: the single logit-lens direction explains little variance, while
SOMP's first few atoms explain much more — a single top-k token ranking under-reads a
polysemantic expert.
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


def direction_evr(A: torch.Tensor, direction: torch.Tensor) -> float:
    """EVR of centred ``A`` captured by a single ``direction`` (one row of the unembedding).

    This is the standard logit-lens read: project the centred activations onto the one
    direction and measure the fraction of total variance it recovers. Same estimator as
    SOMP's EVR (squared projection / total variance), so the number is directly
    comparable to SOMP's first atom. Scale-invariant: the direction is normalised.
    """
    Ac = (A - A.mean(dim=0, keepdim=True)).double().cpu()
    total = (Ac**2).sum().clamp_min(1e-12)  # == n * total variance
    u = direction.double().cpu()
    u = u / u.norm().clamp_min(1e-12)
    explained = (Ac @ u).pow(2).sum()
    return float(explained / total)


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
    lens_evr = direction_evr(A, dictionary[lens_idx[0]])  # single top-1 direction

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
        "mean_lens_evr": (
            float(np.mean([r["lens_evr"] for r in records])) if records else 0.0
        ),
    }
    for i in EVR_AT:
        summary[f"mean_somp_evr_{i}"] = _mean_evr_at("somp_evr", i)
    out = {"summary": summary, "experts": records}

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "logit_lens_comparison.json").write_text(
            json.dumps(out, indent=2)
        )
    return out
