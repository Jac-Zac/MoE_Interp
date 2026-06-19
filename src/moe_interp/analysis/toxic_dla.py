"""Direct Logit Attribution: which experts *write toward* toxic vocabulary (no model).

Gradient-free and model-free — reads only the stored expert OUTPUT contributions (HDF5
extractions) and the unembedding. Following "The Expert Strikes Back" (arXiv:2604.02178),
an expert's toxic score is its output contribution projected onto the toxic-token logit
direction, averaged over the tokens routed to it:

    score(l, e) = mean_tokens( contribution · toxic_dir ),
    toxic_dir   = mean(U[toxic_ids]) - mean(U)        # relative toxic-logit direction

The router gate is already folded into the stored contribution (``expert_forward × gate ×
RMSNorm`` in ``capture.py``), so this *is* the expert's additive push on the toxic logits.
Positive ⇒ the expert writes toward toxic tokens. This is the cheap, activations-only
companion to causal gate-ablation patching: it needs no model forward, so it runs on the
existing extractions in seconds.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from moe_interp.analysis.common import iter_expert_activations, load_analysis_inputs
from moe_interp.pursuit.concepts import CONCEPT_WORDS


def build_toxic_token_ids(tokenizer, words: list[str] | None = None) -> list[int]:
    """Vocabulary ids of single-token toxic words (with and without a leading space)."""
    words = words or CONCEPT_WORDS["offensive"]
    ids: set[int] = set()
    for w in words:
        for variant in (w, " " + w):
            toks = tokenizer(variant, add_special_tokens=False).input_ids
            if len(toks) == 1:
                ids.add(int(toks[0]))
    return sorted(ids)


def toxic_direction(dictionary: torch.Tensor, toxic_ids: list[int]) -> torch.Tensor:
    """Relative toxic-logit direction in residual space: mean toxic row minus overall mean."""
    return (dictionary[toxic_ids].mean(0) - dictionary.mean(0)).double()


def dla_toxic_grid(
    model_name: str,
    dataset: str,
    *,
    min_activations: int = 50,
    max_rows: int | None = 2000,
    extractions_dir: Path | str | None = None,
) -> dict:
    """Per-(layer, expert) DLA toxic score over stored contributions. No model forward.

    Returns ``grid`` (``(n_layers, n_experts)`` tensor; NaN where an expert is unsampled),
    ``top`` (experts ranked by score), and the ``toxic_ids`` used.
    """
    from transformers import AutoTokenizer

    extractions_dir, metadata, dictionary, _ = load_analysis_inputs(
        model_name, dataset, extractions_dir
    )
    tokenizer = AutoTokenizer.from_pretrained(metadata["model_name"])
    toxic_ids = build_toxic_token_ids(tokenizer)
    tdir = toxic_direction(dictionary, toxic_ids)

    n_layers, n_experts = metadata["n_layers"], metadata["n_experts"]
    grid = torch.full((n_layers, n_experts), float("nan"))
    for layer, expert, A in iter_expert_activations(
        extractions_dir, n_layers, n_experts, min_activations, max_rows
    ):
        grid[layer, expert] = float((A.double() @ tdir).mean())

    flat = grid.flatten()
    order = torch.argsort(torch.nan_to_num(flat, nan=-1e30), descending=True)
    top = [
        {"layer": int(i // n_experts), "expert": int(i % n_experts), "score": float(flat[i])}
        for i in order.tolist()
        if not torch.isnan(flat[i])
    ][:25]
    return {
        "grid": grid,
        "top": top,
        "toxic_ids": toxic_ids,
        "n_toxic_ids": len(toxic_ids),
        "n_scored": int((~torch.isnan(grid)).sum()),
    }


def plot_dla_grid(grid: torch.Tensor, output_path, *, title: str) -> None:
    """Save a layer×expert heatmap of the DLA toxic score (diverging, centred at 0)."""
    import numpy as np
    import plotly.graph_objects as go

    z = grid.cpu().numpy()
    vmax = float(np.nanmax(np.abs(z))) or 1.0
    fig = go.Figure(
        go.Heatmap(
            z=z, zmid=0, zmin=-vmax, zmax=vmax, colorscale="RdBu_r",
            colorbar={"title": "DLA toxic score"},
        )
    )
    fig.update_layout(
        title=title, xaxis_title="expert", yaxis_title="layer", height=560,
        yaxis={"autorange": "reversed"},  # layer 0 at top
    )
    fig.write_html(str(output_path))


def run_dla(model_name: str, dataset: str, output_dir: Path, **kwargs) -> dict:
    """Compute the DLA toxic grid and write ``dla_grid.npy`` + heatmap + top-experts JSON."""
    import numpy as np

    res = dla_toxic_grid(model_name, dataset, **kwargs)
    if res["n_scored"] == 0:
        raise ValueError(
            f"No experts had >= min_activations rows in {dataset!r}; this is likely a "
            "last-token extraction (too sparse). Use an all-token dataset (e.g. pile10k)."
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "dla_grid.npy", res["grid"].numpy())
    plot_dla_grid(
        res["grid"], output_dir / "dla_grid.html",
        title=f"DLA toxic score per expert — {model_name} · {dataset}",
    )
    (output_dir / "dla_top_experts.json").write_text(json.dumps(res["top"], indent=2))
    return res
