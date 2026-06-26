#!/usr/bin/env python
"""Circuit · counterfactual expert editing — do the SOMP concept experts *carry* the concept?

Interchange test on minimal pairs ("The sum of 2 and 3 is" -> 5 vs "... 2 and 4 is" -> 6): splice
the residual the counterfactual run writes into the factual run, but only at the positions routed
to the concept's SOMP experts, and check whether the predicted number flips toward the
counterfactual answer — more than for a random expert group at the same layers. This is the
editing experiment neither the Basic-Refinement paper nor Head Pursuit did; it is the cleanest
place to show *specific* causal control, because numbers are a genuinely localized concept (unlike
toxicity). Single-token log-prob eval (proof of concept); extend to multi-token later.

To try another concept: add a sibling of ``numbers_counterfactual_pairs`` and point ``CONCEPT`` /
``CONCEPT_WORDS`` at it.

  DATA_DIR=./data HF_HUB_OFFLINE=1 .venv/bin/python notebooks/circuits/editing.py
"""

# %% Imports
import json
import os
import random

from dotenv import load_dotenv
from nnsight import LanguageModel
from rich import print
from rich.table import Table

from moe_interp.capture.model_adapter import model_num_experts
from moe_interp.circuit.editing import run_counterfactual_edit
from moe_interp.circuit.prompts import numbers_counterfactual_pairs
from moe_interp.circuit.steer import somp_concept_experts
from moe_interp.config import get_default_model, get_device, get_model_dir, set_seed
from moe_interp.pursuit.concepts import CONCEPT_WORDS

# %% Configuration
load_dotenv()
set_seed(1337)
MODEL_NAME = get_default_model()
CONCEPT = "numbers"
GROUP_K = int(os.environ.get("GROUP_K", 8))  # how many SOMP experts form the concept group
out_dir = get_model_dir(MODEL_NAME) / "circuit" / "editing"

# %% Load the model
device_map = os.environ.get("DEVICE_MAP", str(get_device()))
model = LanguageModel(MODEL_NAME, device_map=device_map, dtype="auto", dispatch=True)

# %% Minimal pairs + expert groups (SOMP concept group vs a matched random control)
pairs = numbers_counterfactual_pairs(model.tokenizer)
somp = somp_concept_experts(MODEL_NAME, CONCEPT_WORDS[CONCEPT], GROUP_K)
if not somp:
    raise SystemExit(
        f"No SOMP experts for '{CONCEPT}'. Run concept-restricted pursuit first "
        "(see report tab:numbers) so results.jsonl exists."
    )

ne = model_num_experts(model)
rng = random.Random(0)
used = set(somp)
rand: list[tuple[int, int]] = []
for layer, _ in somp:  # same layers as SOMP, distinct random experts (specificity control)
    while (layer, e := rng.randrange(ne)) in used:
        pass
    used.add((layer, e))
    rand.append((layer, e))

groups = {f"{CONCEPT}-SOMP": somp, "random": rand}
print(f"{len(pairs)} pairs · SOMP group {somp}")

# %% Run the interchange and score
res = run_counterfactual_edit(model, pairs, groups)
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"{CONCEPT}.json"
out_path.write_text(
    json.dumps({"results": res, "groups": {k: v for k, v in groups.items()}}, indent=2)
)
print(f"editing results -> {out_path}")

# %% Table — a real causal group beats random on both swing and flip rate
t = Table(title=f"Counterfactual editing · {CONCEPT} ({len(pairs)} pairs)")
for col in ("group", "#experts", "toward-cf swing", "flip rate"):
    t.add_column(col, justify="right" if col != "group" else "left")
for name, b in res.items():
    t.add_row(
        name, str(b["n_experts"]), f"{b['toward_cf_swing']:+.3f}", f"{b['flip_rate']:.2f}"
    )
print(t)
