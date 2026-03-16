import argparse
import logging
import os

from datasets import load_dataset
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MODEL_REPO = "allenai/OLMoE-1B-7B-0924-Instruct"
DATASET_NAME = "mandarjoshi/trivia_qa"
DATASET_CONFIG = "rc"


def main():
    parser = argparse.ArgumentParser(
        description="Download model and datasets for offline use"
    )
    parser.add_argument("--model", action="store_true", help="Download model only")
    parser.add_argument(
        "--datasets", action="store_true", help="Download datasets only"
    )
    parser.add_argument("--all", action="store_true", help="Download both (default)")
    args = parser.parse_args()

    load_dotenv()

    hf_token = os.environ.get("HF_TOKEN")

    download_all = args.all or (not args.model and not args.datasets)

    if download_all or args.model:
        logger.info(f"Downloading model: {MODEL_REPO}")
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=MODEL_REPO, token=hf_token)
        logger.info("Model downloaded successfully")

    if download_all or args.datasets:
        logger.info(f"Downloading dataset: {DATASET_NAME}/{DATASET_CONFIG}")
        load_dataset(DATASET_NAME, DATASET_CONFIG, split="train", token=hf_token)
        load_dataset(DATASET_NAME, DATASET_CONFIG, split="validation", token=hf_token)
        logger.info("Datasets downloaded successfully")

    logger.info("All downloads complete")


if __name__ == "__main__":
    main()
