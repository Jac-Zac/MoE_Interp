#!/usr/bin/env python

# %% Imports
import json

from rich import print
from rich.table import Table
from transformers import AutoTokenizer

from src.cache import load_metadata, load_unembedding
from src.environment import (
    get_data_dir,
    get_extractions_dir,
    get_pursuit_dir,
    get_unembedding_dir,
    load_env,
    set_seed,
)
from src.plots import plot_count_heatmap, plot_evr_heatmap, plot_label_grid
from src.pursuit import load_pursuit, run_pursuit
from src.word_dictionary import build_word_dictionary

# %% Configuration
seed = 1337
# MODEL_NAME = "openai/gpt-oss-20b"
MODEL_NAME = "allenai/OLMoE-1B-7B-0924-Instruct"
load_env()
set_seed(seed)

# %% Setup
data_dir = get_data_dir()
extractions_dir = get_extractions_dir(MODEL_NAME)
output_dir = get_pursuit_dir(MODEL_NAME)
metadata_path = extractions_dir / "metadata.json"
# labeled_path = output_dir / "results_labeled.json"

if not metadata_path.exists():
    raise Exception(f"You should get activation first, in this path {metadata_path}")

# %% Simple projection-based expert pursuit
# Specify a concept to restrict the unembedding dictionary ("offensive", "countries", "numbers")
# Set to None to probe all tokens — useful as a general-purpose baseline
CONCEPT = None
USE_WORD_DICTIONARY = False
FORCE = True

word_top_k = 25000

min_activations = 5
pursuit_dir = get_pursuit_dir(MODEL_NAME, CONCEPT)
word_dictionary = None
if USE_WORD_DICTIONARY:
    metadata = load_metadata(metadata_path)
    tokenizer = AutoTokenizer.from_pretrained(metadata["model_name"])
    base_dictionary = load_unembedding(
        get_unembedding_dir(MODEL_NAME) / "dictionary.h5"
    ).float()
    word_dictionary = build_word_dictionary(
        tokenizer,
        base_dictionary,
        top_k=word_top_k,
    )
    pursuit_dir = get_pursuit_dir(MODEL_NAME, "words")
if (
    not FORCE
    and (pursuit_dir / "results.jsonl").exists()
    and (pursuit_dir / "evr_matrix.npy").exists()
):
    results, evr_matrix, count_matrix = load_pursuit(pursuit_dir)
    print(f"Loaded existing pursuit results from {pursuit_dir}")
else:
    # output_dir enables incremental results.jsonl flushing — safe to interrupt
    results, evr_matrix, count_matrix = run_pursuit(
        extractions_dir,
        min_activations=min_activations,
        k=50,
        output_dir=pursuit_dir,
        data_dir=data_dir,
        concept=CONCEPT,
        word_dictionary=word_dictionary,
    )

# %% Top experts for the current concept
top_n = 10
top_experts = sorted(
    results,
    key=lambda record: record["evr"][-1] if record["evr"] else 0.0,
    reverse=True,
)[:top_n]

print(f"\nTop {len(top_experts)} experts by final EVR")
if CONCEPT:
    print(f"Concept: {CONCEPT}")

table = Table(show_header=True, header_style="bold")
table.add_column("Rank", justify="right")
table.add_column("Layer", justify="right")
table.add_column("Expert", justify="right")
table.add_column("EVR", justify="right")
table.add_column("n", justify="right")
table.add_column("Top Tokens")

for rank, record in enumerate(top_experts, start=1):
    final_evr = record["evr"][-1] if record["evr"] else 0.0
    table.add_row(
        str(rank),
        str(record["layer"]),
        str(record["expert"]),
        f"{final_evr:.4f}",
        str(record["n_activations"]),
        ", ".join(record["tokens"][:5]),
    )

print(table)

# %% Plot EVR heatmap per expert
plot_evr_heatmap(evr_matrix).show()
if count_matrix is not None:
    plot_count_heatmap(count_matrix).show()

# %% Label grid — requires results_labeled.json (run label_experts.py first)
# TODO: Deal with this and perhaps have this in a separate file
# if labeled_path.exists():
#     labeled_results = json.loads(labeled_path.read_text())
#     plot_label_grid(labeled_results).show()
