#!/usr/bin/env python

# %% Imports
import nnsight
import torch
from nnsight import LanguageModel

from src.environment import set_seed

# %% Model Definition
set_seed(1337)

model = LanguageModel(
    "allenai/OLMoE-1B-7B-0924-Instruct",
    device_map="auto",
    dtype=torch.bfloat16,
    dispatch=False,
)

# %% Prompts
# Define prompts
prompts = ["The capital of France is", "The capital of Italy is"]
print(model)

# %% Running the inference
# Capture expert activations
print(f"\nProcessing {len(prompts)} prompts...")

with model.trace() as tracer:
    hidden_dims = []
    for prompt_idx, prompt in enumerate(prompts):
        with tracer.invoke(prompt):
            # Get expert indices - shape is typically [batch, seq_len, top_k]
            # expert_indices = layer.mlp.experts.source.expert_idx.save()
            hidden_dims.append(model.model.layers[0].mlp.gate.output[0].shape)

    nnsight.save(hidden_dims)

print(hidden_dims)

# %% Analyses with those activations
# Let's do something with those now
