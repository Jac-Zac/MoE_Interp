#!/bin/bash

# Download model once
python -c "from transformers import AutoTokenizer; from nnsight import LanguageModel; LanguageModel('allenai/OLMoE-1B-7B-0924-Instruct'); AutoTokenizer.from_pretrained('allenai/OLMoE-1B-7B-0924-Instruct')"
