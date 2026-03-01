#!/usr/bin/env python

# %% Imports
import plotly.express as px
from transformers import AutoTokenizer

from src.cache import load_metadata
from src.environment import get_data_dir, load_env, set_seed
from src.pursuit import run_pursuit

# %% Configuration
seed = 1337
load_env()
set_seed(seed)

tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924-Instruct")

# %% Setup
data_dir = get_data_dir()
encodings_dir = data_dir / "encodings"
pursuit_dir = data_dir / "pursuit"

metadata_path = encodings_dir / "metadata.json"

if not metadata_path.exists():
    raise Exception(f"You should get activation first, in this path {metadata_path}")

metadata = load_metadata(encodings_dir)
n_layers = metadata["n_layers"]
n_experts = metadata["n_experts"]

# %% Simple projection-based expert pursuit
# %% Run on all experts
min_activations = 5
results, evr_matrix = run_pursuit(
    encodings_dir,
    tokenizer,
    min_activations=min_activations,
    k=50,
    output_dir=pursuit_dir,
    data_dir=data_dir,
)

# %% Display sample results
for r in results[:5]:
    print(f"\nLayer {r['layer']}, Expert {r['expert']}:")
    for t, e in zip(r["tokens"][:10], r["evr"][:10]):
        print(f"  {t}: {e:.4f}")

# %% Plot EVR heatmap per expert
fig = px.imshow(
    evr_matrix,
    x=[f"E{i}" for i in range(n_experts)],
    y=[f"L{i}" for i in range(n_layers)],
    color_continuous_scale="Blues",
    labels=dict(x="Expert", y="Layer", color="Top EVR"),
    title="Expert Pursuit: Top Explained Variance Ratio per Expert",
)
fig.update_layout(width=1600, height=600)
fig.show()
