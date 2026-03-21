import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets import load_dataset
from dotenv import load_dotenv

from src.data import DATASET_SPECS

DEFAULT_MODELS = [
    "allenai/OLMoE-1B-7B-0924-Instruct",
    "openai/gpt-oss-20b",
]


def main():
    parser = argparse.ArgumentParser(
        description="Download models and datasets for offline use"
    )
    parser.add_argument(
        "--model",
        nargs="*",
        choices=DEFAULT_MODELS,
        help="Model repo ID(s) to download (default: all)",
    )
    parser.add_argument(
        "--dataset",
        nargs="*",
        choices=sorted(DATASET_SPECS),
        help="Dataset(s) to download (default: all)",
    )
    args = parser.parse_args()

    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN")

    models_to_download = args.model if args.model else DEFAULT_MODELS
    datasets_to_download = args.dataset if args.dataset else sorted(DATASET_SPECS)

    for repo_id in models_to_download:
        print(f"Downloading model: {repo_id}")
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=repo_id, token=hf_token)
        print(f"Model '{repo_id}' ready")

    for name in datasets_to_download:
        spec = DATASET_SPECS[name]
        print(
            f"Downloading dataset: {spec.hf_id} (config={spec.config}, split={spec.split})"
        )
        load_dataset(spec.hf_id, spec.config, split=spec.split, token=hf_token)
        print(f"Dataset '{name}' ready")

    print("All downloads complete")


if __name__ == "__main__":
    main()
