#!/bin/bash

# Download model and dataset for offline use
python - <<EOF
from huggingface_hub import snapshot_download
from datasets import load_dataset

snapshot_download(
    repo_id="allenai/OLMoE-1B-7B-0924-Instruct",
    cache_dir=None  # uses HF_HOME
)

load_dataset("mandarjoshi/trivia_qa", "rc", split="train")
load_dataset("mandarjoshi/trivia_qa", "rc", split="validation")
print("Model and dataset downloaded successfully")
EOF
