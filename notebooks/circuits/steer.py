#!/usr/bin/env python
"""Circuit · intervene — suppress toxic generation, then assemble the report (needs model).

The causal capstone. During greedy generation we either knock out the top identified
promoter experts (zero their router gate) or project the diff-of-means toxic direction out
of the residual stream, and measure the drop in toxic propensity vs the un-intervened
baseline. The expert sets come from the artifacts the other circuit notebooks wrote
(gate-AtP, the patching grid, DLA, SOMP) plus a matched random control, so this is also
where "causal knockout suppresses toxicity, correlational knockout does nothing" shows up.

The last cell assembles every circuit artifact under data/<model>/circuit/ into one
self-contained HTML report. Mirrors the old `main.py circuit-steer` + `circuit-report`.
Run `patching.py` (and optionally `dla.py`) first so the comparison sets exist.

  DATA_DIR=./data HF_HUB_OFFLINE=1 .venv/bin/python notebooks/circuits/steer.py
"""

# %% Imports
import json

from dotenv import load_dotenv
from nnsight import LanguageModel
from rich import print
from rich.table import Table

from moe_interp.circuit.report import build_report
from moe_interp.circuit.steer import run_steer
from moe_interp.config import get_default_model, get_device, get_model_dir, set_seed

# %% Configuration
load_dotenv()
set_seed(1337)
MODEL_NAME = get_default_model()
CONCEPT = "offensive"  # non-toxic concepts only get the generic project-out edit
KNOCKOUT_K = 15
STEER_LAYER = 12
BATCH_SIZE = 8
MAX_NEW_TOKENS = 24
out_dir = get_model_dir(MODEL_NAME) / "circuit" / "steer"

# %% Load the model
model = LanguageModel(
    MODEL_NAME, device_map=str(get_device()), dtype="auto", dispatch=True
)

# %% Run the intervention experiment (knockout sets + project-out vs baseline)
res = run_steer(
    model,
    MODEL_NAME,
    concept=CONCEPT,
    knockout_k=KNOCKOUT_K,
    steer_layer=STEER_LAYER,
    batch_size=BATCH_SIZE,
    max_new_tokens=MAX_NEW_TOKENS,
)
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "intervention.json").write_text(json.dumps(res, indent=2))
print(f"intervention results -> {out_dir}")

# %% Propensity per method (lower = less of the concept)
concept = res["meta"]["concept"]
base = res["methods"]["baseline"]["eliciting_propensity"]
t = Table(title=f"'{concept}' suppression (baseline propensity = {base:+.3f})")
for col in ("method", "elic propensity", "Δ vs base", "neutral", "word frac"):
    t.add_column(col, justify="right" if col != "method" else "left")
for name, b in res["methods"].items():
    t.add_row(
        name,
        f"{b['eliciting_propensity']:+.3f}",
        f"{base - b['eliciting_propensity']:+.3f}",
        f"{b['neutral_propensity']:+.3f}",
        f"{b['eliciting_word_frac']:.2f}",
    )
print(t)

# %% Assemble the self-contained HTML report from all circuit artifacts
report_path = build_report(MODEL_NAME)
print(f"report -> {report_path}")
