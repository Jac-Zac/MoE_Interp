#!/usr/bin/env python
"""Circuit · localize — causal activation patching + gate-AtP faithfulness (needs model).

Two stages, both over a high-toxicity RealToxicityPrompts split:

  1. The causal *ground truth*: for every routed (layer, expert) zero its router gate in one
     forward pass and record the change in the toxic-logit metric. Positive = the expert
     promotes toxicity, negative = it suppresses it.
  2. gate-AtP estimates that whole grid from a single backward pass, and we score its
     faithfulness (Pearson r) against the patching grid.

Loads OLMoE via nnsight (Apple MPS ok).

  DATA_DIR=./data HF_HUB_OFFLINE=1 .venv/bin/python notebooks/circuits/patching.py
"""

# %% Imports
import json
import os

import numpy as np
import torch
from dotenv import load_dotenv
from nnsight import LanguageModel
from rich import print
from rich.table import Table

from moe_interp.circuit.attribution import gate_attribution
from moe_interp.circuit.compare import faithfulness, plot_faithfulness
from moe_interp.circuit.patching import (
    expert_patching_grid,
    plot_expert_effect_grid,
    top_grid_experts,
)
from moe_interp.circuit.prompts import rtp_split
from moe_interp.config import get_default_model, get_device, get_model_dir, set_seed
from moe_interp.pursuit.concepts import build_toxic_token_ids

# %% Configuration
load_dotenv()
set_seed(1337)
MODEL_NAME = get_default_model()
# The grid is identified on the train split elic[:N_PROMPTS]; steer.py re-derives that same
# deterministic prefix (so its AtP/SOMP/diff-of-means + this grid all share it) and scores
# every method on the disjoint held-out tail elic[N_PROMPTS:N_PROMPTS+N_TEST]. Keep N_PROMPTS
# in sync across both notebooks (the env var does this); N_TEST only sizes steer's eval set.
N_PROMPTS = int(os.environ.get("N_PROMPTS", 100))
N_TEST = int(os.environ.get("N_TEST", 50))
BATCH_SIZE = int(os.environ.get("CIRCUIT_BATCH_SIZE", 24))
ATP_BATCH_SIZE = int(os.environ.get("ATP_BATCH_SIZE", 8))
LAYERS = None  # restrict to e.g. [10, 11, 12] for a faster sweep; None = all layers
cdir = get_model_dir(MODEL_NAME) / "circuit"

# %% Load the model + RTP eliciting prompts/probe
# Identify the circuit on the TRAIN split only; the held-out test split is used by steer.py
# to score the interventions out-of-sample (same split — both notebooks are deterministic).
device_map = os.environ.get("DEVICE_MAP", str(get_device()))
model = LanguageModel(
    MODEL_NAME, device_map=device_map, dtype="auto", dispatch=True
)
toxic_prompts, _, _, _ = rtp_split(model.tokenizer, n_train=N_PROMPTS, n_test=N_TEST)
toxic_ids = build_toxic_token_ids(model.tokenizer)
print(f"{len(toxic_prompts)} RTP eliciting prompts (train) · {len(toxic_ids)} toxic ids")
for ids in toxic_prompts[:5]:  # show a few prompts for clarity
    print(f"  · {model.tokenizer.decode(ids)!r}")

# %% Causal patching grid (one forward per routed expert)
patch_dir = cdir / "patching"
patch_dir.mkdir(parents=True, exist_ok=True)
grid_path = patch_dir / "patching_grid.npy"
if grid_path.exists():
    print(f"Loading existing patching grid from {patch_dir}")
    grid = torch.from_numpy(np.load(grid_path)).float()
else:
    grid = expert_patching_grid(
        model, toxic_prompts, toxic_ids, batch_size=BATCH_SIZE, layers=LAYERS
    )
    np.save(grid_path, grid.numpy())
    plot_expert_effect_grid(
        grid,
        patch_dir / "patching_grid.html",
        title=f"Expert ablation effect on toxic-logit — {MODEL_NAME}",
    )
    print(f"patching grid + heatmap -> {patch_dir}")
top = top_grid_experts(grid)
(patch_dir / "top_experts.json").write_text(json.dumps(top, indent=2))

# %% Top causal experts (by |ablation effect|)
t = Table(title="Top causal toxic experts (signed effect)")
t.add_column("rank", justify="right")
t.add_column("layer", justify="right")
t.add_column("expert", justify="right")
t.add_column("effect", justify="right")
for rank, r in enumerate(top[:10], start=1):
    t.add_row(str(rank), str(r["layer"]), str(r["expert"]), f"{r['effect']:+.4f}")
print(t)

# %% gate-AtP (1 backward pass) + faithfulness vs the causal grid
# gate-AtP is the cheap *causal* attributor; the question is whether one backward pass
# reproduces the expensive per-expert ablation grid (it does: pooled r ≈ 0.80).
# Flush fragmented VRAM from the patching sweep before the backward pass.
if torch.cuda.is_available():
    torch.cuda.empty_cache()
patching = grid.float()  # already in memory from the cell above
grids = {"gate-AtP": gate_attribution(model, toxic_prompts, toxic_ids, batch_size=ATP_BATCH_SIZE)}
scores = faithfulness(grids, patching)
cmp_dir = cdir / "compare"
cmp_dir.mkdir(parents=True, exist_ok=True)
(cmp_dir / "faithfulness.json").write_text(json.dumps(scores, indent=2))
plot_faithfulness(
    scores,
    cmp_dir / "faithfulness.html",
    title=f"Attributor faithfulness vs causal patching — {MODEL_NAME}",
)
print("faithfulness vs causal patching grid (pooled Pearson r):")
for name, s in scores.items():
    print(f"  {name:18s} r = {s['pooled_r']:+.3f}")
print(f"comparison -> {cmp_dir}")
