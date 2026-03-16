import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Expert Pursuit Explorer

    Interactive visualization of expert specialization in MoE models.
    Each expert's top-k tokens are ranked by their individual **Explained Variance Ratio (EVR)**
    — how much of the expert's output variance is explained by projecting onto that token direction.

    `uv run marimo run notebooks/notebook_pursuit_marimo.py`
    """)
    return


@app.cell
def _():
    from src.concepts import CONCEPT_WORDS
    from src.environment import get_data_dir, load_env, set_seed
    from src.pursuit import load_pursuit, run_pursuit

    return (
        CONCEPT_WORDS,
        get_data_dir,
        load_env,
        load_pursuit,
        run_pursuit,
        set_seed,
    )


@app.cell
def _(load_env, set_seed):
    load_env()
    set_seed(1337)
    return


@app.cell
def _(get_data_dir):
    data_dir = get_data_dir()
    data_dir = data_dir / "orfeo"  # NOTE: Get the orfeo data for now
    extractions_dir = data_dir / "extractions"
    pursuit_base = data_dir / "pursuit"
    if not (extractions_dir / "metadata.json").exists():
        raise Exception(
            f"Run extract first — missing {extractions_dir / 'metadata.json'}"
        )
    return data_dir, extractions_dir, pursuit_base


@app.cell
def _(CONCEPT_WORDS, mo, pursuit_base):
    # Only show concepts that have pre-computed results
    _options = {}
    if (pursuit_base / "results.jsonl").exists():
        _options["Full vocabulary"] = None
    for _c in CONCEPT_WORDS:
        if (pursuit_base / _c / "results.jsonl").exists():
            _options[_c.capitalize()] = _c
    concept_dd = mo.ui.dropdown(
        options=_options,
        value=next(iter(_options)) if _options else None,
        label="Concept",
        searchable=True,
        full_width=False,
    )
    return (concept_dd,)


@app.cell
def _(concept_dd, pursuit_base):
    concept = concept_dd.value
    pursuit_dir = pursuit_base / concept if concept else pursuit_base
    return concept, pursuit_dir


@app.cell
def _(
    concept,
    data_dir,
    extractions_dir,
    load_pursuit,
    pursuit_dir,
    run_pursuit,
):
    force = False
    if (
        not force
        and (pursuit_dir / "results.jsonl").exists()
        and (pursuit_dir / "evr_matrix.npy").exists()
    ):
        results, _evr_matrix, _count_matrix = load_pursuit(pursuit_dir)
        print(f"Loaded pursuit results from {pursuit_dir}")
    else:
        # output_dir enables incremental results.jsonl flushing — safe to interrupt
        results, _evr_matrix, _count_matrix = run_pursuit(
            extractions_dir,
            min_activations=5,
            k=50,
            output_dir=pursuit_dir,
            data_dir=data_dir,
            concept=concept,
        )
    return (results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Expert Rankings
    """)
    return


@app.cell
def _(mo, results):
    _layers = sorted({r["layer"] for r in results})
    layer_dd_ov = mo.ui.dropdown(
        options={"All layers": None, **{f"Layer {l}": l for l in _layers}},
        value="All layers",
        label="Layer",
        searchable=True,
        full_width=False,
    )
    sort_dd = mo.ui.dropdown(
        options={
            "Total EVR ↓": ("total_evr", True),
            "Total EVR ↑": ("total_evr", False),
            "Top-1 EVR ↓": ("top1_evr", True),
            "Activations ↓": ("n_activations", True),
            "Activations ↑": ("n_activations", False),
        },
        value="Total EVR ↓",
        label="Sort by",
        full_width=False,
    )
    top_n_slider = mo.ui.slider(
        start=5,
        stop=100,
        step=5,
        value=10,
        label="Show top-n",
        show_value=True,
        debounce=True,
    )
    return layer_dd_ov, sort_dd, top_n_slider


@app.cell
def _(concept_dd, layer_dd_ov, mo, sort_dd, top_n_slider):
    mo.hstack([concept_dd, layer_dd_ov, sort_dd, top_n_slider], justify="center", gap=2)
    return


@app.cell
def _(layer_dd_ov, mo, results, sort_dd, top_n_slider):
    # Filter records to selected layer
    _records = [
        r
        for r in results
        if layer_dd_ov.value is None or r["layer"] == layer_dd_ov.value
    ]

    if not _records:
        _out = mo.callout(mo.md("No data for this selection."), kind="warn")
    else:
        # Build summary rows
        _rows = []
        for _r in _records:
            _evr_list = _r["evr"]
            _tok_list = _r["tokens"]
            _rows.append(
                {
                    "layer": _r["layer"],
                    "expert": _r["expert"],
                    "total_evr": _evr_list[-1] if _evr_list else 0.0,
                    "top1_evr": _evr_list[0] if _evr_list else 0.0,
                    "top1_token": _tok_list[0] if _tok_list else "",
                    "n_activations": _r["n_activations"],
                }
            )

        # Sort
        _sort_field, _reverse = sort_dd.value
        _rows.sort(key=lambda x: x[_sort_field], reverse=_reverse)

        _shown = _rows[: top_n_slider.value]
        _max_evr = max(r["total_evr"] for r in _rows) or 1.0
        _header = (
            "| # | layer | expert | total EVR | | top token | top-1 EVR | activations |\n"
            "|--:|------:|-------:|----------:|-|:----------|---------:|------------:|\n"
        )
        _body = "\n".join(
            "| {i} | {layer} | {expert} | {tevr} | {bar} | `{tok}` | {t1evr} | {acts} |".format(
                i=i + 1,
                layer=r["layer"],
                expert=r["expert"],
                tevr=f"{r['total_evr']:.4f}",
                bar="█" * max(1, round(r["total_evr"] / _max_evr * 20)),
                tok=r["top1_token"].replace("|", "\\|"),
                t1evr=f"{r['top1_evr']:.4f}",
                acts=r["n_activations"],
            )
            for i, r in enumerate(_shown)
        )
        _out = mo.vstack(
            [
                mo.stat(value=f"{len(_shown)} / {len(_rows)}", label="Experts shown"),
                mo.center(mo.md(_header + _body)),
            ],
            align="center",
            gap=1,
        )
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Expert Detail
    """)
    return


@app.cell
def _(mo, results):
    _layers = sorted({r["layer"] for r in results})
    layer_dd_det = mo.ui.dropdown(
        options={f"Layer {l}": l for l in _layers},
        value=f"Layer {_layers[0]}",
        label="Layer",
        searchable=True,
        full_width=False,
    )
    return (layer_dd_det,)


@app.cell
def _(layer_dd_det, mo, results):
    _experts = sorted(
        {r["expert"] for r in results if r["layer"] == layer_dd_det.value}
    )
    expert_dd = mo.ui.dropdown(
        options={f"Expert {e}": e for e in _experts},
        value=f"Expert {_experts[0]}" if _experts else None,
        label="Expert",
        searchable=True,
        full_width=False,
    )
    top_k_slider = mo.ui.slider(
        start=5,
        stop=50,
        step=1,
        value=20,
        label="Top-k tokens",
        show_value=True,
        debounce=True,
    )
    return expert_dd, top_k_slider


@app.cell
def _(expert_dd, layer_dd_det, mo, top_k_slider):
    mo.hstack([layer_dd_det, expert_dd, top_k_slider], justify="center", gap=2)
    return


@app.cell
def _(expert_dd, layer_dd_det, mo, results, top_k_slider):
    _record = next(
        (
            r
            for r in results
            if r["layer"] == layer_dd_det.value and r["expert"] == expert_dd.value
        ),
        None,
    )

    if _record is None:
        _out = mo.callout(
            mo.md("No data for this layer / expert combination."), kind="warn"
        )
    else:
        _k = min(top_k_slider.value, len(_record["tokens"]))
        _tokens = _record["tokens"][:_k]
        _cum_evr = _record["evr"][:_k]

        # Individual contribution = diff of consecutive cumulative EVR values
        _indiv = [_cum_evr[0]] + [_cum_evr[i] - _cum_evr[i - 1] for i in range(1, _k)]
        _total_evr = _cum_evr[-1]
        _n_act = _record["n_activations"]

        # Token table (markdown — theme-neutral)
        _max_iv = max(_indiv) or 1.0
        _rows = "\n".join(
            "| {rank} | `{tok}` | {bar} | {iv} | {cv} |".format(
                rank=i + 1,
                tok=t.replace("|", "\\|"),
                bar="█" * max(1, round(iv / _max_iv * 20)),
                iv=f"{iv:.4f}",
                cv=f"{cv:.4f}",
            )
            for i, (t, iv, cv) in enumerate(zip(_tokens, _indiv, _cum_evr))
        )
        _out = mo.vstack(
            [
                mo.hstack(
                    [
                        mo.stat(value=f"{_total_evr:.3f}", label="Total EVR"),
                        mo.stat(value=str(_n_act), label="Activations"),
                        mo.stat(value=str(_k), label="Tokens shown"),
                    ],
                    justify="center",
                    gap=2,
                ),
                mo.center(
                    mo.md(
                        "| # | token | | indiv EVR | cumul EVR |\n"
                        "|--:|:------|--|----------:|----------:|\n" + _rows
                    )
                ),
            ],
            align="center",
            gap=1,
        )
    _out
    return


if __name__ == "__main__":
    app.run()
