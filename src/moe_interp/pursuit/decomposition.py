"""Sparse decomposition used by Expert Pursuit.

Adapted from the ResiDual sparse decomposition implementation:
https://github.com/Flegyas/ResiDual
(trimmed to the l1-criterion, chosen+EVR path this repo uses)
"""

import torch


class SOMP:
    """Simultaneous orthogonal matching pursuit over activation rows."""

    def __init__(self, k: int, pc: int | None = None) -> None:
        self.k = k
        self.pc = pc

    def __call__(
        self,
        X: torch.Tensor,
        dictionary: torch.Tensor,
        device: torch.device | str,
        dict_t: torch.Tensor | None = None,
    ) -> dict:
        orig_X = X
        if self.pc is not None:
            components, weights = _pca_components(X, self.pc)
            X = components * weights.unsqueeze(1) ** 2

        return somp(
            X=X,
            orig_X=orig_X,
            pc=self.pc,
            dictionary=dictionary,
            k=self.k,
            device=device,
            centering=self.pc is None,
            dict_t=dict_t,
        )


def _pca_components(X: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    X_centered = X - X.mean(dim=0)
    _, singular_values, vh = torch.linalg.svd(X_centered, full_matrices=False)
    return vh[:k], singular_values[:k]


@torch.no_grad()
def somp(
    X: torch.Tensor,
    orig_X: torch.Tensor,
    pc,
    dictionary: torch.Tensor,
    k: int,
    device: torch.device | str,
    centering: bool = True,
    dict_t: torch.Tensor | None = None,
) -> dict:
    """Greedy SOMP decomposition (l1 selection criterion).

    Returns ``{"chosen": (k,) atom indices, "evr": (k,) cumulative EVR}``.

    `dict_t` is the transposed dictionary (d_model × n_atoms). When pursuit runs
    over many experts with the same dictionary, pass it in precomputed to skip
    the per-call 412 MB transpose-and-copy.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if dictionary.shape[0] < k:
        raise ValueError(f"Dictionary has {dictionary.shape[0]} rows, k={k}")
    if dictionary.shape[1] != X.shape[1]:
        raise ValueError(
            f"Dictionary width {dictionary.shape[1]} != X width {X.shape[1]}"
        )

    X = X.to(device)
    dictionary = dictionary.to(device)
    if dict_t is None:
        dict_t = dictionary.T.contiguous()
    elif dict_t.device != dictionary.device or dict_t.shape != (
        dictionary.shape[1],
        dictionary.shape[0],
    ):
        raise ValueError("dict_t inconsistent with dictionary")

    orig_X = orig_X.to(device)
    orig_X_mean = orig_X.mean(dim=0, keepdim=True)
    orig_X_centered = orig_X - orig_X_mean
    std_orig_sum = (torch.std(orig_X, dim=0) ** 2).sum(dim=-1).clamp_min(1e-12)

    if centering:
        X_mean = X.mean(dim=0, keepdim=True)
        X = X - X_mean
    else:
        X_mean = torch.zeros((1, X.shape[1]), device=device, dtype=X.dtype)

    chosen = torch.zeros(k, dtype=torch.long, device=device)
    available = torch.ones(dictionary.shape[0], dtype=torch.bool, device=device)
    residual = X.clone()
    recon = torch.zeros_like(X)
    evr = torch.zeros(k, device=device)

    # `X` is constant inside the loop; only `current_atoms` grows. Materialize the
    # lstsq right-hand side once instead of re-transferring it every iteration
    # (on MPS the solve runs on CPU, so this also avoids k host transfers of X).
    is_mps = torch.device(device).type == "mps"
    X_rhs = X.T.float().cpu() if is_mps else X.T.double()

    for i in range(k):
        cross = residual @ dict_t
        proj_scores = cross.abs().sum(dim=0)

        atom_idx = proj_scores.masked_fill(~available, -torch.inf).argmax()
        chosen[i] = atom_idx
        available[atom_idx] = False

        current_atoms = torch.index_select(dictionary, 0, chosen[: i + 1])
        if is_mps:
            lstsq_weights = torch.linalg.lstsq(
                current_atoms.T.float().cpu(), X_rhs
            ).solution.to(device=device, dtype=X.dtype)
        else:
            lstsq_weights = torch.linalg.lstsq(
                current_atoms.T.double(), X_rhs
            ).solution.to(dtype=X.dtype)

        recon = (current_atoms.T @ lstsq_weights).T
        residual = X - recon

        if pc is None:
            recon_full = X_mean + recon
        else:
            u, _, vh = torch.linalg.svd(recon, full_matrices=False)
            somp_pcs = u @ vh
            recon_full = orig_X_centered @ somp_pcs.T @ somp_pcs + orig_X_mean
        std_recon = torch.std(recon_full, dim=0) ** 2
        evr[i] = std_recon.sum(dim=-1) / std_orig_sum

    return {"chosen": chosen.cpu(), "evr": evr.cpu().float()}
