#!/usr/bin/env python
"""Circuit · localized project-out — is the toxic direction carried by a few experts?

Tests whether the diff-of-means project-out (the only intervention that cleanly suppresses
toxicity) can be *localized* to the residual positions routed to a handful of toxic experts,
instead of scrubbing every position. The expert sets come straight from the published
comparison (``circuit/steer/offensive/intervention.json`` ``meta.sets``) so no AtP/SOMP
recompute is needed:

    global              project v out at every position (the current method) — control
    routed-patching     project out only where the top patching experts fired
    routed-SOMP         project out only where the top SOMP experts fired
    routed-random       project out only where matched random experts fired

If routed-patching ~= global, toxicity is localized and steering can be made surgical (and
patching/AtP/SOMP earn their keep as selectors). If only global works, the direction is
distributed. Comparing routed-patching vs routed-SOMP is the causal-vs-correlational test on
the intervention that actually works.

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

from moe_interp.circuit.intervene import (
    compose_interventions,
    localized_projectout_intervention,
    projectout_intervention,
    run_intervention_experiment,
)
from moe_interp.circuit.prompts import rtp_split
from moe_interp.circuit.steer import collect_last_token_residuals
from moe_interp.config import get_default_model, get_device, get_model_dir, set_seed
from moe_interp.pursuit.concepts import CONCEPT_WORDS, build_toxic_token_ids

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

# Reuse the already-computed expert sets from the main intervention run.
sets_path = out_dir / "intervention.json"
SETS = json.loads(sets_path.read_text())["meta"]["sets"]
ROUTED = {k: SETS[k] for k in ("patching", "SOMP", "random") if k in SETS}

# %% Load the model
device_map = os.environ.get("DEVICE_MAP", str(get_device()))
model = LanguageModel(MODEL_NAME, device_map=device_map, dtype="auto", dispatch=True)

# %% Disjoint identify/evaluate split
elic_tr, elic_te, neut_tr, neut_te = rtp_split(
    model.tokenizer, n_train=N_TRAIN, n_test=N_TEST
)
concept_words = CONCEPT_WORDS[CONCEPT]
concept_ids = build_toxic_token_ids(model.tokenizer, concept_words)


# %% Per-layer diff-of-means toxic directions (train split)
def direction_at(layer: int):
    tox = collect_last_token_residuals(model, elic_tr, layer, BATCH_SIZE).mean(0)
    neu = collect_last_token_residuals(model, neut_tr, layer, BATCH_SIZE).mean(0)
    return tox - neu


needed_layers = {STEER_LAYER} | {l for s in ROUTED.values() for l, _ in s}
print(f"computing diff-of-means directions for layers {sorted(needed_layers)} ...")
DIRS = {l: direction_at(l) for l in sorted(needed_layers)}

# %% Build methods: baseline, global control, and one routed variant per selector
methods = {
    "baseline": None,
    f"global-projectout@L{STEER_LAYER}": projectout_intervention(
        STEER_LAYER, DIRS[STEER_LAYER]
    ),
}
for name, experts in ROUTED.items():
    by_layer: dict[int, list[int]] = {}
    for layer, e in experts:
        by_layer.setdefault(layer, []).append(e)
    fns = [
        localized_projectout_intervention(layer, DIRS[layer], elist)
        for layer, elist in sorted(by_layer.items())
    ]
    methods[f"routed-{name}"] = compose_interventions(fns)

# %% Score every method on the held-out test split
res = run_intervention_experiment(
    model,
    elic_te,
    neut_te,
    concept_ids,
    methods,
    concept_words=concept_words,
    max_new_tokens=MAX_NEW_TOKENS,
)
out = {
    "methods": res,
    "meta": {
        "concept": CONCEPT,
        "steer_layer": STEER_LAYER,
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "max_new_tokens": MAX_NEW_TOKENS,
        "routed_sets": ROUTED,
    },
}
out_path = out_dir / "localized_intervention.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"localized results -> {out_path}")

# %% Table
base = res["baseline"]["eliciting_propensity"]
t = Table(title=f"Localized project-out (baseline propensity = {base:+.3f})")
for col in ("method", "elic propensity", "Δ vs base", "neutral", "word frac"):
    t.add_column(col, justify="right" if col != "method" else "left")
for name, b in res.items():
    t.add_row(
        name,
        f"{b['eliciting_propensity']:+.3f}",
        f"{base - b['eliciting_propensity']:+.3f}",
        f"{b['neutral_propensity']:+.3f}",
        f"{b['eliciting_word_frac']:.2f}",
    )
print(t)
