#!/usr/bin/env python

# %% Imports
import logging
from collections import Counter

import nnsight
import plotly.express as px
import torch
from nnsight import LanguageModel
from residual.sparse_decomposition import SOMP
from tqdm import tqdm

from src.capture import capture_batch
from src.data import load_triviaqa
from src.environment import get_device, set_seed
from src.pursuit import build_filtered_dictionary

logging.getLogger("httpx").setLevel(logging.WARNING)


# %% Configuration
seed = 1337
set_seed(seed)
device = get_device()

model = LanguageModel(
    "allenai/OLMoE-1B-7B-0924-Instruct",
    device_map="auto",
    dtype=torch.float16,
    dispatch=True,
)
tokenizer = model.tokenizer

n_layers = model.config.num_hidden_layers
n_experts = model.config.num_experts
d_model = model.config.hidden_size

# %% Load TriviaQA prompts
n_docs = 24
batch_size = 8
prompts = load_triviaqa(tokenizer, n_docs=n_docs)
print(f"Loaded {len(prompts)} TriviaQA prompts")

# %% Batched last-token capture
# With left-padding, the last token is always at position seq_len - 1.
# This is the simplest capture: no content boundary tracking needed.

all_batches: list[torch.Tensor] = []

for start in tqdm(range(0, len(prompts), batch_size), desc="Tracing batches"):
    batch = prompts[start : start + batch_size]
    batch_result = capture_batch(model, batch)
    all_batches.append(batch_result)

activations = torch.cat(all_batches, dim=0)
print(f"Captured activations shape: {activations.shape}")

# %% Build Filtered Concept Dictionary
property_name = "countries"
unembed = model.lm_head.weight.detach().cpu()
dictionary, tokens_data = build_filtered_dictionary(unembed, tokenizer, property_name)

# %% Run Expert Pursuit (SOMP)
k = min(50, len(tokens_data))
decomposition = SOMP(k=k)
evr_matrix = torch.zeros(n_layers, n_experts, k)
zscore_matrix = torch.zeros(n_layers, n_experts)
expert_concepts: dict[tuple[int, int], list[str]] = {}

for li in tqdm(range(n_layers), desc="Expert Pursuit"):
    for ei in range(n_experts):
        X = activations[:, li, ei, :]
        if X.norm() < 1e-6:
            continue

        decomp_out = decomposition(
            X=X.double(),
            dictionary=dictionary,
            descriptors=list(range(len(dictionary))),
            device="cpu",
        )
        chosen = decomp_out["chosen"]
        evr_matrix[li, ei] = decomp_out["evr"]

        vocab_ids = [tokens_data[t] for t in chosen.tolist()]
        tokens = [tokenizer.decode([vid]).strip() for vid in vocab_ids]
        expert_concepts[(li, ei)] = tokens

        D_chosen = dictionary[chosen]
        sim_matrix = D_chosen @ dictionary.T
        mean_sim = sim_matrix.mean()
        std_sim = sim_matrix.std()
        if std_sim > 1e-8:
            zscore_matrix[li, ei] = (
                (D_chosen @ D_chosen.T).mean() - mean_sim
            ) / std_sim

print(f"Analyzed {len(expert_concepts)} active experts across {n_layers} layers")

# %% EVR Heatmap
fig = px.imshow(
    evr_matrix[:, :, -1].numpy(),
    x=[f"E{i}" for i in range(n_experts)],
    y=[f"L{i}" for i in range(n_layers)],
    color_continuous_scale="Blues",
    labels=dict(x="Experts", y="Layers", color="EVR"),
)
fig.update_layout(title=f"Expert Pursuit EVR ({property_name})", width=1400, height=600)
fig.show()

# %% Z-Score Heatmap
fig = px.imshow(
    zscore_matrix.numpy(),
    x=[f"E{i}" for i in range(n_experts)],
    y=[f"L{i}" for i in range(n_layers)],
    color_continuous_scale=px.colors.diverging.RdYlBu_r,
    color_continuous_midpoint=0.0,
    labels=dict(x="Experts", y="Layers", color="Z-Score"),
)
fig.update_layout(
    title=f"Expert Concept Coherence ({property_name})", width=1400, height=600
)
fig.show()

# %% Concept Frequency Analysis
concept_counter: Counter = Counter()
for tokens in expert_concepts.values():
    concept_counter.update(tokens[:5])

print(f"\nTop 20 concepts across all experts ({property_name}):")
for word, count in concept_counter.most_common(20):
    print(f"  {word}: {count}")
