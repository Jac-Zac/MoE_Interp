#!/usr/bin/env python
"""Circuit · classify — Direct Logit Attribution toxic-expert score (no model).

Gradient-free and model-free: reads only the stored expert OUTPUT contributions (HDF5
extractions) + the unembedding, and scores each expert by how much it writes toward toxic
vocabulary. This is the cheap *correlational* baseline for the causal patching grid in
`patching.py` — the contrast that makes the "association != causation" finding concrete.
Mirrors the old `main.py toxic-dla`.

  DATA_DIR=./data HF_HUB_OFFLINE=1 .venv/bin/python notebooks/circuits/dla.py
"""

# %% Imports
from dotenv import load_dotenv
from rich import print
from rich.table import Table

from moe_interp.analysis.toxic_dla import run_dla
from moe_interp.config import get_default_model, get_model_dir, set_seed

# %% Configuration
load_dotenv()
set_seed(1337)
MODEL_NAME = get_default_model()
# All-token capture: rtp/last-token extractions are too sparse to score an expert.
DATASET = "pile10k"
MIN_ACTS = 50
MAX_ROWS = 2000
out_dir = get_model_dir(MODEL_NAME) / "circuit" / "dla" / DATASET

# %% Score every expert's toxic-write contribution
res = run_dla(
    MODEL_NAME, DATASET, out_dir, min_activations=MIN_ACTS, max_rows=MAX_ROWS
)
print(
    f"scored {res['n_scored']} experts against {res['n_toxic_ids']} toxic token ids; "
    f"grid + heatmap written to {out_dir}"
)

# %% Experts that write most toward toxic vocabulary
t = Table(title="DLA — experts writing toward toxic vocab")
t.add_column("rank", justify="right")
t.add_column("layer", justify="right")
t.add_column("expert", justify="right")
t.add_column("score", justify="right")
for rank, r in enumerate(res["top"][:15], start=1):
    t.add_row(str(rank), str(r["layer"]), str(r["expert"]), f"{r['score']:+.4f}")
print(t)
