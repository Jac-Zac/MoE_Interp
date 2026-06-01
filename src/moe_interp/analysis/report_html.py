"""Shared building blocks for self-contained HTML analysis reports.

The pipeline report (``report.py``) and the standalone exam scripts
(``scripts/nlp_report.py``, ``scripts/unsupervised_report.py``) all emit a single
self-contained HTML file with the same look: a title, a grey subtitle line, Plotly
figures (with Plotly JS inlined exactly once), and bordered tables. Keeping that
markup — and the SOMP pursuit lookup the scripts share — in one place avoids three
near-identical copies drifting apart.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import plotly.graph_objects as go


def _css(max_width: int) -> str:
    return (
        "body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:"
        f"{max_width}px;margin:2rem auto;padding:0 1rem;color:#1a1a1a;line-height:1.55}}"
        "h1{margin-bottom:0}.sub{color:#666;margin-top:.25rem}"
        "h2{margin-top:2.4rem;border-bottom:2px solid #eee;padding-bottom:.3rem}"
        "table{border-collapse:collapse;width:100%;font-size:.85rem;margin:1rem 0}"
        "th,td{border:1px solid #ddd;padding:.35rem .5rem;text-align:left}"
        "th{background:#f5f7fa}tr:nth-child(even){background:#fafbfc}"
        "li{margin-bottom:.6rem}code{background:#f3f3f3;padding:0 .3rem}"
        ".caveat{background:#fff8e6;border-left:4px solid #f0c000;padding:.6rem 1rem;"
        "border-radius:4px}"
    )


def table(headers: Sequence[Any], rows: Iterable[Sequence[Any]]) -> str:
    """Render a bordered HTML table; cells are stringified as-is (pre-escape if needed)."""
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def fig_html(fig: go.Figure, first: bool) -> str:
    """One Plotly figure as embeddable HTML; inline Plotly JS only on the first."""
    return fig.to_html(full_html=False, include_plotlyjs=("inline" if first else False))


def figs_to_html(figs: Iterable[go.Figure]) -> str:
    """Concatenate figures, inlining Plotly JS exactly once."""
    return "".join(fig_html(f, i == 0) for i, f in enumerate(figs))


def html_page(
    *, title: str, heading: str, subtitle: str, body: str, max_width: int = 1000
) -> str:
    """Wrap report ``body`` in the shared page chrome (head, CSS, title, subtitle)."""
    return (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        f"<title>{title}</title><style>{_css(max_width)}</style></head><body>"
        f'<h1>{heading}</h1><p class="sub">{subtitle}</p>{body}</body></html>'
    )


def load_pursuit_map(model_name: str, dataset: str) -> dict[tuple[int, int], dict]:
    """``(layer, expert) -> SOMP pursuit record``, preferring local results then synced Orfeo.

    Returns an empty map when no ``results.jsonl`` exists. SOMP tokens read off these
    records are display LABELS only — never clustering/analysis inputs.
    """
    from moe_interp.analysis.decode import load_pursuit_results
    from moe_interp.config import resolve_pursuit_dir

    pursuit_dir = resolve_pursuit_dir(model_name, dataset)
    if pursuit_dir is None:
        return {}
    print(f"Loaded pursuit results from {pursuit_dir}")
    return load_pursuit_results(pursuit_dir)
