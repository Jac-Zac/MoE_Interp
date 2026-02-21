"""Expert Pursuit: SOMP-based concept decomposition of MoE experts.

Robust extraction of the notebook analysis logic. Builds filtered concept
dictionaries from word lists, runs SOMP on all experts across layers, and
collects EVR, z-scores, and per-expert concept decompositions.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.constants import WORD_LISTS
from residual.sparse_decomposition import SOMP


def build_filtered_dictionary(
    unembed: torch.Tensor,
    tokenizer: Any,
    word_list: str | list[str] | set[str] = "countries",
) -> tuple[torch.Tensor, list[int]]:
    """Build a filtered, normalized unembedding dictionary from a word list.

    Tokenizes each word in the list, collects unique token IDs, and slices
    the unembedding matrix to only those rows. Returns both the normalized
    dictionary and the remapping table (tokens_data[i] -> original vocab ID).

    Args:
        unembed: [vocab_size, d_model] raw unembedding matrix (lm_head.weight)
        tokenizer: HuggingFace tokenizer with __call__ support
        word_list: Either a key in WORD_LISTS ("countries", "colors",
            "quantity") or an explicit list/set of words. Pass "all"
            to use the full unembedding.

    Returns:
        dictionary: [n_tokens, d_model] L2-normalized filtered unembedding
        tokens_data: sorted list of original vocab IDs (remapping table)
    """
    if word_list == "all":
        tokens_data = list(range(unembed.shape[0]))
        dictionary = F.normalize(unembed.float(), dim=-1)
        return dictionary, tokens_data

    if isinstance(word_list, str):
        if word_list not in WORD_LISTS:
            raise ValueError(
                f"Unknown word list: {word_list!r}. "
                f"Choose from {list(WORD_LISTS.keys())} or pass explicit words."
            )
        names = WORD_LISTS[word_list]
    else:
        names = word_list

    token_id_set: set[int] = set()
    for name in names:
        ids = tokenizer(name, add_special_tokens=False)["input_ids"]
        token_id_set.update(ids)

    tokens_data = sorted(token_id_set)
    dictionary = F.normalize(unembed[tokens_data].float(), dim=-1)
    return dictionary, tokens_data


@dataclass
class ExpertConceptResult:
    """SOMP decomposition result for a single expert."""

    layer: int
    expert_id: int
    tokens: list[str]
    token_ids: list[int]
    evr: list[float]
    zscore: float = 0.0


@dataclass
class PursuitResult:
    """Full Expert Pursuit analysis results."""

    n_layers: int
    n_experts: int
    k: int
    property_name: str
    experts: list[ExpertConceptResult] = field(default_factory=list)
    evr_matrix: torch.Tensor = field(default_factory=lambda: torch.zeros(0))
    zscore_matrix: torch.Tensor = field(default_factory=lambda: torch.zeros(0))

    def concept_frequency(self, top_n: int = 5) -> Counter:
        """Aggregate top-N concepts across all experts."""
        counter: Counter = Counter()
        for e in self.experts:
            counter.update(e.tokens[:top_n])
        return counter

    def save(self, path: Path) -> None:
        """Save results to JSON + tensors."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save tensors
        torch.save(self.evr_matrix, path / "evr.pt")
        torch.save(self.zscore_matrix, path / "zscore.pt")

        # Save expert concepts as JSON
        data = {
            "n_layers": self.n_layers,
            "n_experts": self.n_experts,
            "k": self.k,
            "property": self.property_name,
            "experts": [
                {
                    "layer": e.layer,
                    "expert_id": e.expert_id,
                    "tokens": e.tokens,
                    "token_ids": e.token_ids,
                    "evr": e.evr,
                    "zscore": e.zscore,
                }
                for e in self.experts
            ],
        }
        with open(path / "pursuit_results.json", "w") as f:
            json.dump(data, f, indent=2)


def run_expert_pursuit(
    activations: torch.Tensor,
    dictionary: torch.Tensor,
    tokens_data: list[int],
    tokenizer: Any,
    k: int = 50,
    property_name: str = "countries",
) -> PursuitResult:
    """Run SOMP on all experts across all layers.

    Args:
        activations: [n_docs, n_layers, n_experts, d_model] mean gated outputs
        dictionary: [n_tokens, d_model] L2-normalized filtered dictionary
        tokens_data: remapping table (tokens_data[i] -> original vocab ID)
        tokenizer: HuggingFace tokenizer for decoding
        k: Number of SOMP atoms to select per expert
        property_name: Name of the word list used (for metadata)

    Returns:
        PursuitResult with per-expert decompositions, EVR, and z-scores
    """
    n_docs, n_layers, n_experts, d_model = activations.shape
    k = min(k, len(tokens_data))

    evr_matrix = torch.zeros(n_layers, n_experts, k)
    zscore_matrix = torch.zeros(n_layers, n_experts)
    expert_results: list[ExpertConceptResult] = []

    decomposition = SOMP(k=k)

    for li in tqdm(range(n_layers), desc="Expert Pursuit"):
        for ei in range(n_experts):
            X = activations[:, li, ei, :]  # [n_docs, d_model]
            if X.norm() < 1e-6:
                continue

            res = decomposition(
                X=X.double(),
                dictionary=dictionary,
                descriptors=list(range(len(dictionary))),
                device=X.device,
            )
            evr_matrix[li, ei] = res["evr"]

            # Remap filtered indices -> vocab IDs -> decoded tokens
            chosen = res["chosen"]
            vocab_ids = [tokens_data[t] for t in chosen.tolist()]
            tokens = [tokenizer.decode([vid]).strip() for vid in vocab_ids]

            # Z-score: coherence of chosen atoms
            D_chosen = dictionary[chosen]  # [k, d_model]
            mean_sim = (D_chosen @ dictionary.T).mean()
            std_sim = (D_chosen @ dictionary.T).std()
            zs = 0.0
            if std_sim > 1e-8:
                internal_sim = (D_chosen @ D_chosen.T).mean()
                zs = float((internal_sim - mean_sim) / std_sim)
            zscore_matrix[li, ei] = zs

            expert_results.append(
                ExpertConceptResult(
                    layer=li,
                    expert_id=ei,
                    tokens=tokens,
                    token_ids=vocab_ids,
                    evr=res["evr"].tolist(),
                    zscore=zs,
                )
            )

    return PursuitResult(
        n_layers=n_layers,
        n_experts=n_experts,
        k=k,
        property_name=property_name,
        experts=expert_results,
        evr_matrix=evr_matrix,
        zscore_matrix=zscore_matrix,
    )
