#!/usr/bin/env python

# %% Imports
from transformers import AutoTokenizer

from src.environment import get_data_dir, load_env, set_seed
from src.plots import plot_evr_heatmap
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

# %% Simple projection-based expert pursuit
min_activations = 5
results, evr_matrix = run_pursuit(
    encodings_dir,
    tokenizer,
    min_activations=min_activations,
    k=50,
    data_dir=data_dir,
)

# %% Display sample results
for r in results[:5]:
    print(f"\nLayer {r['layer']}, Expert {r['expert']}:")
    for t, e in zip(r["tokens"][:10], r["evr"][:10]):
        print(f"  {t}: {e:.4f}")

# %% Plot EVR heatmap per expert
plot_evr_heatmap(evr_matrix).show()
