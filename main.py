#!/usr/bin/env python

import torch
from nnsight import LanguageModel
from transformers import BitsAndBytesConfig

from src.capture import capture_expert_activations
from src.environment import set_seed


def main():
    set_seed(1337)
    # Load OLMoE model with 4-bit quantization

    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    model = LanguageModel(
        "allenai/OLMoE-1B-7B-0924-Instruct",
        device_map="auto",
        quantization_config=quantization_config,
    )

    # Define prompts
    prompts = ["The capital of France is", "The capital of Italy is"]
    print(model)

    # Capture expert activations
    print(f"\nProcessing {len(prompts)} prompts...")
    results = capture_expert_activations(model, prompts)
    print(results)


if __name__ == "__main__":
    main()
