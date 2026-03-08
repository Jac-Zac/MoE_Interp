import os
import random
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv


def load_env(override: bool = False) -> None:
    """Load environment variables from .env file.

    Args:
        override: If True, .env values overwrite variables already set in the environment
                  (e.g. from a SLURM job script). Defaults to False so shell-exported
                  variables always take priority.
    """
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=override)


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
