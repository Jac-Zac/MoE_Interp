import os
import random
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table


def get_model_config(model, verbose: bool = False) -> dict:
    """Extract model configuration with multi-model support."""
    model_type = model.config.model_type
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size

    if model_type == "gpt_oss":
        n_experts = model.config.num_local_experts
        experts_per_tok = model.config.num_experts_per_tok
    else:
        n_experts = model.config.num_experts
        experts_per_tok = model.config.experts_per_token

    if verbose:
        console = Console()
        model_table = Table(title="Model Config")
        model_table.add_column("Property", style="cyan")
        model_table.add_column("Value", style="magenta")
        model_table.add_row("Model", model.config._name_or_path)
        model_table.add_row("Model Type", model_type)
        model_table.add_row("Layers", str(n_layers))
        model_table.add_row("Hidden Size", str(d_model))
        model_table.add_row("Experts", str(n_experts))
        model_table.add_row("Experts per Token", str(experts_per_tok))
        console.print(model_table)

    return {
        "n_layers": n_layers,
        "n_experts": n_experts,
        "d_model": d_model,
        "experts_per_tok": experts_per_tok,
    }


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


def get_model_dir(model_name: str) -> Path:
    """Convert model name to safe directory name.

    'openai/gpt-oss-20b' -> 'openai_gpt_oss_20b'
    'allenai/OLMoE-1B-7B-0924-Instruct' -> 'allenai_OLMoE_1B_7B_0924_Instruct'
    """
    safe_name = model_name.replace("/", "_").replace("-", "_")
    return get_data_dir() / safe_name


def get_extractions_dir(model_name: str) -> Path:
    """Get the extractions directory for a specific model."""
    return get_model_dir(model_name) / "extractions"


def get_unembedding_dir(model_name: str) -> Path:
    """Get the unembedding directory for a specific model."""
    return get_model_dir(model_name) / "unembedding"


def get_pursuit_dir(model_name: str, concept: str | None = None) -> Path:
    """Get the pursuit output directory for a specific model."""
    pursuit_dir = get_model_dir(model_name) / "pursuit"
    if concept:
        pursuit_dir = pursuit_dir / concept
    return pursuit_dir
