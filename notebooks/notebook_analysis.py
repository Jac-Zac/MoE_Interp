#!/usr/bin/env python
"""Post-hoc analyses on stored activations — interactive `# %%` walkthrough.

No model is loaded: the analysis recomputes from the HDF5 extractions + the SOMP
`results.jsonl`. Mirrors `main.py analysis`.

  1. logit-lens baseline vs SOMP — does a bulk mean-projection logit lens read the same
     thing as SOMP, and how much variance does each explain?

    DATA_DIR=./data HF_HUB_OFFLINE=1 .venv/bin/python notebooks/notebook_analysis.py
"""

# %% Imports
from dotenv import load_dotenv
from rich import print
from rich.table import Table

from moe_interp.analysis import run_logit_lens_comparison
from moe_interp.config import get_default_model, get_model_dir, set_seed

# %% Configuration
load_dotenv()
set_seed(1337)
MODEL_NAME = get_default_model()
# DATASET = "pile10k"  # dense all-token capture; the regime where polysemanticity shows
DATASET = "triviaqa"
MIN_ACTS = 50
MAX_ROWS = 1500
out_dir = get_model_dir(MODEL_NAME) / "analysis" / DATASET

# %% 1. Logit-lens baseline vs SOMP
lens = run_logit_lens_comparison(
    MODEL_NAME, DATASET, min_activations=MIN_ACTS, max_rows=MAX_ROWS, output_dir=out_dir
)
s = lens["summary"]
t = Table(title="Mean-projection logit lens vs SOMP")
t.add_column("metric")
t.add_column("logit lens")
t.add_column("SOMP")
t.add_row("EVR @1", f"{s['mean_lens_evr_1']:.4f}", f"{s['mean_somp_evr_1']:.4f}")
t.add_row("EVR @3", f"{s['mean_lens_evr_3']:.4f}", f"{s['mean_somp_evr_3']:.4f}")
t.add_row("EVR @10", f"{s['mean_lens_evr_10']:.4f}", f"{s['mean_somp_evr_10']:.4f}")
t.add_row("top-10 Jaccard", f"{s['mean_jaccard_topk']:.3f}", "—")
print(t)
