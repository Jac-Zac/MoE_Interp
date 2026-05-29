import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


def is_rank0() -> bool:
    """Return True if we are on rank 0 or not in a distributed setup."""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_data_dir() -> Path:
    """Get data directory from environment variable or use default.

    Returns:
        Path object for the data directory (creates if needed)
    """
    data_dir = os.environ.get("DATA_DIR", "./data")
    path = Path(data_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device() -> torch.device:
    """Determine the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_default_model() -> str:
    """Return the default model name from the environment, with a built-in fallback."""
    return os.environ.get("DEFAULT_MODEL", "allenai/OLMoE-1B-7B-0924-Instruct")


def get_model_dir(model_name: str) -> Path:
    """Convert model name to safe directory name.

    'openai/gpt-oss-20b' -> 'openai_gpt_oss_20b'
    'allenai/OLMoE-1B-7B-0924-Instruct' -> 'allenai_OLMoE_1B_7B_0924_Instruct'
    """
    safe_name = model_name.replace("/", "_").replace("-", "_")
    return get_data_dir() / safe_name


def get_extractions_dir(model_name: str, dataset: str | None = None) -> Path:
    """Get the extractions directory for a specific model."""
    extractions_dir = get_model_dir(model_name) / "extractions"
    if dataset:
        extractions_dir = extractions_dir / dataset
    return extractions_dir


def get_unembedding_dir(model_name: str) -> Path:
    """Get the unembedding directory for a specific model."""
    return get_model_dir(model_name) / "unembedding"


def get_pursuit_dir(
    model_name: str,
    dataset: str | None = None,
    concept: str | None = None,
) -> Path:
    """Get the pursuit output directory for a specific model."""
    pursuit_dir = get_model_dir(model_name) / "pursuit"
    if dataset:
        pursuit_dir = pursuit_dir / dataset
    if concept:
        pursuit_dir = pursuit_dir / concept
    return pursuit_dir


def get_analysis_dir(model_name: str, dataset: str | None = None) -> Path:
    """Get the unsupervised-analysis output directory for a specific model."""
    analysis_dir = get_model_dir(model_name) / "analysis"
    if dataset:
        analysis_dir = analysis_dir / dataset
    return analysis_dir
