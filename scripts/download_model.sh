#!/bin/bash

# Download model once
python - <<EOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="allenai/OLMoE-1B-7B-0924-Instruct",
    cache_dir=None  # uses HF_HOME
)
EOF
