#!/usr/bin/env python
"""Circuit · localize — gate-AtP causal localization grid (needs model).

For every routed (layer, expert) we estimate, from a *single backward pass*, how much zeroing
its router gate would change the toxic-logit metric (gate-AtP, ``g·dL/dg``). Positive = the
expert promotes toxicity, negative = it suppresses it. The result is an (n_layers, n_experts)
grid, cached for the intervention notebooks (steer.py / circuit_runner.py) and plotted as a
heatmap.

gate-AtP is a first-order approximation of exhaustive activation patching; the two were checked
once and agreed closely (see ``moe_interp.circuit.attribution`` for the method + validation).

Loads OLMoE via nnsight (Apple MPS ok).

  DATA_DIR=./data HF_HUB_OFFLINE=1 .venv/bin/python notebooks/circuits/localize.py
"""

# %% Imports
import os

import numpy as np
from dotenv import load_dotenv
from nnsight import LanguageModel
from rich import print
from rich.table import Table

from moe_interp.circuit.attribution import gate_attribution
from moe_interp.circuit.prompts import rtp_split
from moe_interp.config import get_default_model, get_device, get_model_dir, set_seed
from moe_interp.grids import top_experts
from moe_interp.io.plots import diverging_expert_heatmap
from moe_interp.pursuit.concepts import build_toxic_token_ids

# %% Configuration
load_dotenv()
set_seed(1337)
MODEL_NAME = get_default_model()
# The grid is identified on the train split elic[:N_PROMPTS]; steer.py re-derives that same
# deterministic prefix (so its AtP set + diff-of-means + this grid all share it) and scores every
# method on the disjoint held-out tail. Keep N_PROMPTS in sync across both notebooks.
N_PROMPTS = int(os.environ.get("N_PROMPTS", 100))
N_TEST = int(os.environ.get("N_TEST", 50))
ATP_BATCH_SIZE = int(os.environ.get("ATP_BATCH_SIZE", 8))
cdir = get_model_dir(MODEL_NAME) / "circuit"

# %% Load the model + RTP eliciting prompts/probe
device_map = os.environ.get("DEVICE_MAP", str(get_device()))
model = LanguageModel(MODEL_NAME, device_map=device_map, dtype="auto", dispatch=True)
toxic_prompts, _, _, _ = rtp_split(model.tokenizer, n_train=N_PROMPTS, n_test=N_TEST)
toxic_ids = build_toxic_token_ids(model.tokenizer)
print(
    f"{len(toxic_prompts)} RTP eliciting prompts (train) · {len(toxic_ids)} toxic ids"
)
for ids in toxic_prompts[:5]:  # show a few prompts for clarity
    print(f"  · {model.tokenizer.decode(ids)!r}")

# %% gate-AtP localization grid (one backward pass)
atp_dir = cdir / "attribution"
atp_dir.mkdir(parents=True, exist_ok=True)
grid_path = atp_dir / f"atp_grid_n{len(toxic_prompts)}.npy"
if grid_path.exists():
    print(f"Loading existing gate-AtP grid from {grid_path}")
    grid = np.load(grid_path)
else:
    grid = gate_attribution(
        model, toxic_prompts, toxic_ids, batch_size=ATP_BATCH_SIZE
    ).numpy()
    np.save(grid_path, grid)
    diverging_expert_heatmap(
        grid,
        title=f"gate-AtP attribution per expert — {MODEL_NAME}",
        colorbar_title="gate-AtP<br>(promotes toxicity)",
        output_path=atp_dir / "atp_grid.html",
    )
    print(f"gate-AtP grid + heatmap -> {atp_dir}")

# %% Top causal experts (by |attribution|)
t = Table(title="Top causal toxic experts (gate-AtP, signed effect)")
for col in ("rank", "layer", "expert", "AtP"):
    t.add_column(col, justify="right")
for rank, (layer, e, v) in enumerate(top_experts(grid, 10, by="abs"), start=1):
    t.add_row(str(rank), str(layer), str(e), f"{v:+.4f}")
print(t)
