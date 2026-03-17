import argparse
import logging
import os

from datasets import load_dataset
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_MODELS = [
    "allenai/OLMoE-1B-7B-0924-Instruct",
    "openai/gpt-oss-20b",
]
DATASET_NAME = "mandarjoshi/trivia_qa"
DATASET_CONFIG = "rc"


def main():
    parser = argparse.ArgumentParser(
        description="Download model and datasets for offline use"
    )
    parser.add_argument(
        "--model",
        nargs="*",
        help="Model repo ID(s) to download (default: download both OLMoE and gpt-oss)",
    )
    parser.add_argument(
        "--datasets", action="store_true", help="Download datasets only"
    )
    parser.add_argument(
        "--all", action="store_true", help="Download both models and datasets"
    )
    args = parser.parse_args()

    load_dotenv()

    hf_token = os.environ.get("HF_TOKEN")

    download_models = args.all or args.model is not None or (not args.datasets)
    download_datasets = args.all or args.datasets

    models_to_download = args.model if args.model else DEFAULT_MODELS

    if download_models:
        for repo_id in models_to_download:
            logger.info(f"Downloading model: {repo_id}")
            from huggingface_hub import snapshot_download

            snapshot_download(repo_id=repo_id, token=hf_token)
            logger.info(f"Model downloaded successfully")

    if download_datasets:
        logger.info(f"Downloading dataset: {DATASET_NAME}/{DATASET_CONFIG}")
        load_dataset(DATASET_NAME, DATASET_CONFIG, split="train", token=hf_token)
        load_dataset(DATASET_NAME, DATASET_CONFIG, split="validation", token=hf_token)
        logger.info("Datasets downloaded successfully")

    logger.info("All downloads complete")


if __name__ == "__main__":
    main()
