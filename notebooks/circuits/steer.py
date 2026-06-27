#!/usr/bin/env python
"""Circuit · intervene — expert-level suppression of a concept, then assemble the report (needs model).

The causal capstone, done entirely at the **expert** level (no residual-stream edits). For the
concept's experts under each selector — gate-AtP (causal), SOMP / Expert Pursuit (correlational),
and a matched random control — we run two interventions during greedy generation and measure the
change in concept propensity vs the un-intervened baseline:

  * knockout (α=0)  — zero the expert's router gate (necessity test)
  * esteer(α)       — add α·v_e to the expert's *output* (v_e = diff-of-means in expert-output
                      space; influence test)

scored with a neutral-prompt collateral check and a distinct-1 coherence guard. This is where
"causal knockout is near-inert (top-k redundancy), correlational SOMP only 'works' by breaking
generation" shows up. A cumulative dose-response curve (toxicity vs #experts steered) follows.
Prompts default to a RealToxicityPrompts split (high- vs low-toxicity); the concept-restricted
SOMP pursuit at pursuit/<dataset>/<concept>/ must exist (run `main.py pursuit --concept` first).

The last cell assembles every circuit artifact under data/<model>/circuit/ into one
self-contained HTML report. Run `localize.py` first so the gate-AtP grid is cached.

  DATA_DIR=./data HF_HUB_OFFLINE=1 .venv/bin/python notebooks/circuits/steer.py
"""

# %% Imports
import json
import os

from dotenv import load_dotenv
from nnsight import LanguageModel
from rich import print
from rich.table import Table

from moe_interp.circuit.compare import plot_intervention
from moe_interp.circuit.prompts import rtp_split
from moe_interp.circuit.report import build_report
from moe_interp.circuit.steer import run_dose_response, run_expert_steer
from moe_interp.config import get_default_model, get_device, get_model_dir, set_seed

# %% Configuration
load_dotenv()
set_seed(1337)
MODEL_NAME = get_default_model()
CONCEPT = "offensive"
DATASET = "rtp"
KNOCKOUT_K = int(os.environ.get("KNOCKOUT_K", 15))
BATCH_SIZE = int(os.environ.get("STEER_BATCH_SIZE", 8))
# Identification (train) prompts MUST match localize.py's N_PROMPTS so the gate-AtP grid is
# reused (cached as atp_grid_n<N>.npy) and SOMP/AtP/esteer all train on the same elic[:N_PROMPTS]
# prefix, then score on the disjoint held-out slice elic[N_PROMPTS:N_PROMPTS+N_TEST].
N_PROMPTS = int(os.environ.get("N_PROMPTS", 100))
N_TEST = int(os.environ.get("N_TEST", 50))
MAX_NEW_TOKENS = 24
out_dir = get_model_dir(MODEL_NAME) / "circuit" / "steer" / CONCEPT

# %% Load the model
device_map = os.environ.get("DEVICE_MAP", str(get_device()))
model = LanguageModel(MODEL_NAME, device_map=device_map, dtype="auto", dispatch=True)

# %% Disjoint identify/evaluate split (held out — see note above)
elic_tr, elic_te, neut_tr, neut_te = rtp_split(
    model.tokenizer, n_train=N_PROMPTS, n_test=N_TEST
)

# %% Run the expert-level intervention experiment (knockout + esteer per selector vs baseline)
res = run_expert_steer(
    model,
    MODEL_NAME,
    concept=CONCEPT,
    dataset=DATASET,
    k=KNOCKOUT_K,
    batch_size=BATCH_SIZE,
    max_new_tokens=MAX_NEW_TOKENS,
    train=(elic_tr, neut_tr),
    test=(elic_te, neut_te),
)
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "expert_intervention.json").write_text(json.dumps(res, indent=2))
plot_intervention(
    res["methods"],
    out_dir / "expert_intervention.html",
    title=f"Expert-intervention propensity — {MODEL_NAME} · concept={CONCEPT}",
)
print(f"expert-intervention results -> {out_dir}")

# %% Propensity per method (lower = less concept; distinct-1 guards against degenerate text)
concept = res["meta"]["concept"]
base = res["methods"]["baseline"]["eliciting_propensity"]
base_neu = res["methods"]["baseline"]["neutral_propensity"]
t = Table(title=f"'{concept}' suppression (baseline propensity = {base:+.3f})")
for col in (
    "method",
    "elic",
    "Δelic",
    "neutral",
    "specificity",
    "word frac",
    "distinct-1",
):
    t.add_column(col, justify="right" if col != "method" else "left")
for name, b in res["methods"].items():
    de = base - b["eliciting_propensity"]
    dn = base_neu - b["neutral_propensity"]
    t.add_row(
        name,
        f"{b['eliciting_propensity']:+.3f}",
        f"{de:+.3f}",
        f"{b['neutral_propensity']:+.3f}",
        f"{de - dn:+.3f}",
        f"{b['eliciting_word_frac']:.2f}",
        f"{b.get('eliciting_distinct1', 0):.2f}",
    )
print(t)

# %% Cumulative dose-response (toxicity vs #experts steered, per selector vs random)
dose = run_dose_response(
    model,
    MODEL_NAME,
    concept=CONCEPT,
    dataset=DATASET,
    k=KNOCKOUT_K,
    max_new_tokens=MAX_NEW_TOKENS,
    train=(elic_tr, neut_tr),
    test=(elic_te, neut_te),
)
(out_dir / "dose_response.json").write_text(json.dumps(dose, indent=2))
print(f"dose-response -> {out_dir / 'dose_response.json'}")

# %% Assemble the self-contained HTML report from all circuit artifacts
report_path = build_report(MODEL_NAME)
print(f"report -> {report_path}")
