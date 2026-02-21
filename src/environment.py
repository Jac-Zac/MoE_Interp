import os
import random
from pathlib import Path

import numpy as np
import torch
from nnsight import LanguageModel


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


def load_model(model_name="allenai/OLMoE-1B-7B-0924-Instruct") -> LanguageModel:
    """Load OLMoE model via nnsight."""
    return LanguageModel(
        model_name,
        device_map="auto",
        dtype=torch.float16,
        dispatch=True,
    )
