"""Dictionary utilities for Expert Pursuit.

Handles unembedding matrix extraction and concept-restricted dictionaries.
"""

from pathlib import Path

import h5py
import torch
import torch.nn.functional as F


def extract_unembedding(model, save_dir: Path | None = None) -> torch.Tensor:
    """Extract unembedding matrix (lm_head.weight) from model.

    Args:
        model: nnsight LanguageModel instance
        save_dir: If provided, saves to save_dir/unembedding.h5

    Returns:
        Unembedding matrix [vocab_size, d_model]
    """
    unembed = model.lm_head.weight.detach().cpu().float()

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        with h5py.File(save_dir / "unembedding.h5", "w") as f:
            f.create_dataset("weight", data=unembed.numpy(), dtype="float32")

    return unembed


def load_unembedding(path: Path) -> torch.Tensor:
    """Load saved unembedding matrix.

    Args:
        path: Path to unembedding.h5

    Returns:
        Unembedding matrix [vocab_size, d_model]
    """
    with h5py.File(path, "r") as f:
        return torch.from_numpy(f["weight"][:]).float()


def make_concept_dictionary(
    unembedding: torch.Tensor,
    tokenizer,
    words: list[str],
) -> tuple[torch.Tensor, list[int]]:
    """Build concept-restricted dictionary from a word list.

    Tokenizes each word, selects corresponding rows from the
    unembedding matrix, and L2-normalizes them.

    Args:
        unembedding: Full unembedding matrix [vocab_size, d_model]
        tokenizer: Model tokenizer
        words: List of concept words (e.g., country names)

    Returns:
        Tuple of (dictionary [n_tokens, d_model], token_ids)
    """
    token_ids: set[int] = set()
    for word in words:
        # Tokenize with and without leading space to catch both forms
        for form in [word, f" {word}"]:
            ids = tokenizer.encode(form, add_special_tokens=False)
            token_ids.update(ids)

    token_ids_sorted = sorted(token_ids)
    dictionary = unembedding[token_ids_sorted]
    dictionary = F.normalize(dictionary, dim=-1)

    return dictionary, token_ids_sorted


# Concept word lists for testing
CONCEPTS: dict[str, list[str]] = {
    "countries": [
        "Afghanistan",
        "Albania",
        "Algeria",
        "Argentina",
        "Australia",
        "Austria",
        "Bangladesh",
        "Belgium",
        "Brazil",
        "Canada",
        "Chile",
        "China",
        "Colombia",
        "Croatia",
        "Cuba",
        "Denmark",
        "Egypt",
        "England",
        "Ethiopia",
        "Finland",
        "France",
        "Germany",
        "Greece",
        "Hungary",
        "Iceland",
        "India",
        "Indonesia",
        "Iran",
        "Iraq",
        "Ireland",
        "Israel",
        "Italy",
        "Japan",
        "Kenya",
        "Korea",
        "Mexico",
        "Morocco",
        "Netherlands",
        "Nigeria",
        "Norway",
        "Pakistan",
        "Peru",
        "Philippines",
        "Poland",
        "Portugal",
        "Romania",
        "Russia",
        "Scotland",
        "Singapore",
        "Spain",
        "Sweden",
        "Switzerland",
        "Thailand",
        "Turkey",
        "Ukraine",
        "Vietnam",
        "Wales",
    ],
    "colors": [
        "red",
        "blue",
        "green",
        "yellow",
        "orange",
        "purple",
        "pink",
        "black",
        "white",
        "brown",
        "gray",
        "grey",
        "cyan",
        "magenta",
        "violet",
        "indigo",
        "crimson",
        "scarlet",
        "turquoise",
        "gold",
        "silver",
        "maroon",
        "navy",
        "teal",
        "coral",
        "beige",
        "ivory",
    ],
    "numbers": [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "hundred",
        "thousand",
        "million",
        "billion",
    ],
}
