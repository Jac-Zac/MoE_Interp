#!/usr/bin/env python
"""Circuit · localized project-out — is the toxic direction carried at the causal experts?

Tests whether the diff-of-means project-out can be *localized* to the residual positions
routed to a handful of identified experts, instead of scrubbing every position. The expert
sets come straight from the main comparison (``circuit/steer/offensive/intervention.json``
``meta.sets``) so no recompute is needed:

    global-projectout   project v out at every position of one layer — rough control
    routed-AtP          project out only where the top gate-AtP experts fired
    routed-patching     project out only where the top patching experts fired
    routed-SOMP         project out only where the top SOMP experts fired
    routed-random       project out only where matched random experts fired

The decisive comparison is *between routed variants* (patching/AtP vs SOMP/random): same
machinery, different selector, so it isolates causal vs correlational localization. A drop only
counts if it is **toxicity-specific** — it must suppress the eliciting prompts MORE than the
neutral ones (specificity = Δelic - Δneut > 0); a large eliciting drop with an equal neutral
drop is blunt suppression, not localization. Needs a sizeable held-out split to tell them apart.

Small by default so it fits on a Mac (subset of prompts, short continuations); scale up the
env vars on Orfeo. OOM-prone: this loads OLMoE and traces generation.

  DATA_DIR=./data HF_HUB_OFFLINE=1 .venv/bin/python notebooks/circuits/localized_steer.py
"""

# %% Imports
import json
import os

from dotenv import load_dotenv
from nnsight import LanguageModel
from rich import print
from rich.table import Table

from moe_interp.circuit.prompts import rtp_split
from moe_interp.circuit.steer import run_localized_steer
from moe_interp.config import get_default_model, get_device, get_model_dir, set_seed

# %% Configuration (small for local; bump these on Orfeo)
load_dotenv()
set_seed(1337)
MODEL_NAME = get_default_model()
CONCEPT = "offensive"
STEER_LAYER = int(os.environ.get("STEER_LAYER", 12))  # layer for the global control direction
N_TRAIN = int(os.environ.get("N_PROMPTS", 8))  # prompts to build the directions
N_TEST = int(os.environ.get("N_TEST", 6))  # held-out prompts to score on
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", 12))
BATCH_SIZE = int(os.environ.get("STEER_BATCH_SIZE", 8))
out_dir = get_model_dir(MODEL_NAME) / "circuit" / "steer" / CONCEPT

# Reuse the already-computed expert sets from the main intervention run (incl. AtP).
sets_path = out_dir / "intervention.json"
SETS = json.loads(sets_path.read_text())["meta"]["sets"]

# %% Load the model
device_map = os.environ.get("DEVICE_MAP", str(get_device()))
model = LanguageModel(MODEL_NAME, device_map=device_map, dtype="auto", dispatch=True)

# %% Disjoint identify/evaluate split
elic_tr, elic_te, neut_tr, neut_te = rtp_split(
    model.tokenizer, n_train=N_TRAIN, n_test=N_TEST
)

# %% Build + score every method on the held-out test split (one shared code path)
out = run_localized_steer(
    model,
    concept=CONCEPT,
    sets=SETS,
    train=(elic_tr, neut_tr),
    test=(elic_te, neut_te),
    steer_layer=STEER_LAYER,
    batch_size=BATCH_SIZE,
    max_new_tokens=MAX_NEW_TOKENS,
)
res = out["methods"]
out_path = out_dir / "localized_intervention.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"localized results -> {out_path}")

# %% Table — specificity is Δelic - Δneut (>0 means the drop is toxicity-specific, not blunt)
base = res["baseline"]["eliciting_propensity"]
base_neu = res["baseline"]["neutral_propensity"]
t = Table(title=f"Localized project-out (baseline propensity = {base:+.3f})")
for col in ("method", "elic", "Δelic", "neutral", "Δneut", "specificity", "word frac"):
    t.add_column(col, justify="right" if col != "method" else "left")
for name, b in res.items():
    de = base - b["eliciting_propensity"]
    dn = base_neu - b["neutral_propensity"]
    t.add_row(
        name,
        f"{b['eliciting_propensity']:+.3f}",
        f"{de:+.3f}",
        f"{b['neutral_propensity']:+.3f}",
        f"{dn:+.3f}",
        f"{de - dn:+.3f}",
        f"{b['eliciting_word_frac']:.2f}",
    )
print(t)
