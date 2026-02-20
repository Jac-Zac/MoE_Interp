"""SOMP: Simultaneous Orthogonal Matching Pursuit.

Port of the HeadPursuit SOMP algorithm adapted for Expert Pursuit.
Reference: Tropp (2006), "Algorithms for simultaneous sparse approximation."
"""

import torch


@torch.no_grad()
def somp(
    X: torch.Tensor,
    dictionary: torch.Tensor,
    k: int = 50,
    criterion: str = "l1",
    center: bool = True,
) -> dict[str, torch.Tensor]:
    """Simultaneous Orthogonal Matching Pursuit.

    Greedily selects dictionary atoms that best explain the variance
    in X across all samples simultaneously.

    Args:
        X: [n_samples, d] activation matrix
        dictionary: [n_atoms, d] L2-normalized dictionary rows
        k: Number of dictionary atoms to select
        criterion: Aggregation criterion ("l1" or "std")
        center: Whether to center X before decomposition

    Returns:
        Dictionary with:
            chosen: [k] indices of selected atoms
            evr: [k] cumulative explained variance ratio
            weights: [k] L2 norms of coefficient vectors
    """
    # Work in float64 for numerical stability (lstsq is sensitive)
    X = X.double()
    dictionary = dictionary.double()

    if center:
        X = X - X.mean(dim=0, keepdim=True)

    total_var = X.var(dim=0).sum()
    if total_var < 1e-12:
        return {
            "chosen": torch.zeros(k, dtype=torch.long),
            "evr": torch.zeros(k),
            "weights": torch.zeros(k),
        }

    residual = X.clone()
    mask = torch.ones(dictionary.shape[0], device=X.device, dtype=X.dtype)
    chosen: list[int] = []
    evr = torch.zeros(k)
    weights = torch.zeros(k)

    for i in range(k):
        # Cross-correlation between residual and all dictionary atoms
        cross = residual @ dictionary.T  # [n_samples, n_atoms]
        cross = cross * mask

        # Aggregate scores across samples
        if criterion == "l1":
            scores = cross.abs().sum(dim=0)
        elif criterion == "std":
            scores = cross.std(dim=0)
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

        # Greedy selection
        atom_idx = int(scores.argmax().item())
        chosen.append(atom_idx)
        mask[atom_idx] = 0

        # Least-squares refit using all chosen atoms
        D_chosen = dictionary[chosen]  # [i+1, d]
        W = torch.linalg.lstsq(D_chosen.T, X.T).solution  # [i+1, n_samples]
        recon = (D_chosen.T @ W).T  # [n_samples, d]

        # Update residual
        residual = X - recon

        # Explained variance ratio
        evr[i] = recon.var(dim=0).sum() / total_var
        weights[i] = W.norm(dim=1).mean()

    return {
        "chosen": torch.tensor(chosen, dtype=torch.long),
        "evr": evr.float(),
        "weights": weights.float(),
    }
