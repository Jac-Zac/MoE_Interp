"""Expert Pursuit: apply SOMP to expert activations and rank by concept."""

from pathlib import Path

import torch
from tqdm import tqdm

from src.cache import ExpertActivationStore
from src.dictionary import CONCEPTS, load_unembedding, make_concept_dictionary
from src.somp import somp


def expert_pursuit(
    data_dir: Path,
    concept: str,
    tokenizer,
    unembed_path: Path | None = None,
    k: int = 50,
    device: str = "cpu",
) -> dict:
    """Run SOMP for all experts using a concept dictionary.

    Args:
        data_dir: Root directory with encoded expert activations
        concept: Concept name (key in CONCEPTS dict) or "full" for full unembed
        tokenizer: Model tokenizer for building concept dictionary
        unembed_path: Path to unembedding.h5 (default: data_dir/../unembedding.h5)
        k: Number of SOMP iterations
        device: Computation device

    Returns:
        Dictionary with:
            evr: [n_layers, n_experts, k] explained variance ratios
            chosen: dict[(layer, expert)] -> [k] chosen atom indices
    """
    meta = ExpertActivationStore.load_metadata(data_dir)
    n_layers = meta["n_layers"]
    n_experts = meta["n_experts"]

    # Load unembedding matrix
    if unembed_path is None:
        unembed_path = data_dir.parent / "unembedding.h5"
    unembedding = load_unembedding(unembed_path)

    # Build dictionary
    if concept == "full":
        import torch.nn.functional as F

        dictionary = F.normalize(unembedding, dim=-1)
        token_ids = list(range(unembedding.shape[0]))
    else:
        words = CONCEPTS[concept]
        dictionary, token_ids = make_concept_dictionary(unembedding, tokenizer, words)

    dictionary = dictionary.to(device)

    evr_scores = torch.zeros(n_layers, n_experts, k)
    chosen_atoms: dict[tuple[int, int], torch.Tensor] = {}

    total = n_layers * n_experts
    pbar = tqdm(total=total, desc=f"Expert Pursuit ({concept})")

    for layer_idx in range(n_layers):
        for expert_id, X in ExpertActivationStore.stream_experts(
            data_dir, layer_idx, device=device
        ):
            # Filter to documents where expert was activated (non-zero rows)
            active_mask = X.abs().sum(dim=-1) > 0
            X_active = X[active_mask]

            if X_active.shape[0] < 3:
                # Too few samples for meaningful SOMP
                pbar.update(1)
                continue

            result = somp(X_active, dictionary, k=k)

            evr_scores[layer_idx, expert_id] = result["evr"]

            # Map chosen indices back to token IDs
            chosen_token_ids = torch.tensor(
                [token_ids[i] for i in result["chosen"].tolist()]
            )
            chosen_atoms[(layer_idx, expert_id)] = chosen_token_ids

            pbar.update(1)

    pbar.close()
    return {"evr": evr_scores, "chosen": chosen_atoms}


def save_pursuit_results(
    results: dict,
    tokenizer,
    output_dir: Path,
) -> None:
    """Save SOMP results: scores tensor + per-expert decomposition files.

    Args:
        results: Output from expert_pursuit()
        tokenizer: Model tokenizer for decoding tokens
        output_dir: Directory to save results
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save EVR scores
    torch.save(results["evr"], output_dir / "scores.pt")

    # Save per-expert decompositions
    for (layer_idx, expert_id), token_ids in results["chosen"].items():
        layer_dir = output_dir / f"layer_{layer_idx:02d}"
        layer_dir.mkdir(exist_ok=True)

        # Save token IDs
        torch.save(token_ids, layer_dir / f"expert_{expert_id:02d}.pt")

        # Save decoded tokens as human-readable text
        tokens = [tokenizer.decode([tid]) for tid in token_ids.tolist()]
        text = "\n".join(f"{i:3d}: {tok}" for i, tok in enumerate(tokens))
        (layer_dir / f"expert_{expert_id:02d}.txt").write_text(text)


def print_top_experts(
    evr_scores: torch.Tensor,
    n: int = 20,
) -> list[tuple[int, int, float]]:
    """Print and return top experts ranked by final EVR.

    Args:
        evr_scores: [n_layers, n_experts, k] from expert_pursuit()
        n: Number of top experts to show

    Returns:
        List of (layer, expert_id, evr) tuples
    """
    # Use final EVR (last SOMP iteration)
    final_evr = evr_scores[:, :, -1]  # [n_layers, n_experts]
    flat = final_evr.flatten()
    top_indices = flat.topk(n).indices

    n_experts = evr_scores.shape[1]
    results = []
    for idx in top_indices:
        layer = idx.item() // n_experts
        expert = idx.item() % n_experts
        score = flat[idx].item()
        results.append((layer, expert, score))
        print(f"  Layer {layer:2d}, Expert {expert:2d}: EVR = {score:.4f}")

    return results
