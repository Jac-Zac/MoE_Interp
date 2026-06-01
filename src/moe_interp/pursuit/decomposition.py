"""Sparse decomposition used by Expert Pursuit.

Adapted from the ResiDual sparse decomposition implementation:
https://github.com/Flegyas/ResiDual
"""

import numpy as np
import torch
import torch.nn.functional as F


class SOMP:
    """Simultaneous orthogonal matching pursuit over activation rows."""

    def __init__(
        self,
        k: int,
        criterion: str = "l1",
        pc: int | None = None,
        compute_evr: bool = False,
        return_full: bool = True,
    ) -> None:
        self.k = k
        self.criterion = criterion
        self.pc = pc
        self.compute_evr = compute_evr
        self.return_full = return_full

    @property
    def key(self) -> str:
        return f"somp_{self.k}_{self.criterion}_{self.pc}"

    def __call__(
        self,
        X: torch.Tensor,
        dictionary: torch.Tensor,
        descriptors: list,
        device: torch.device | str,
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
            descriptors=descriptors,
            k=self.k,
            device=device,
            criterion=self.criterion,
            centering=self.pc is None,
            compute_evr=self.compute_evr,
            return_full=self.return_full,
        )

    def forward(
        self,
        X: torch.Tensor,
        dictionary: torch.Tensor,
        descriptors: list,
        device: torch.device | str,
    ) -> dict:
        return self(
            X=X,
            dictionary=dictionary,
            descriptors=descriptors,
            device=device,
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
    descriptors: list,
    k: int,
    device: torch.device | str,
    criterion: str = "l1",
    centering: bool = True,
    compute_evr: bool = False,
    return_full: bool = True,
) -> dict:
    """Greedy SOMP decomposition."""
    if dictionary.shape[0] < k:
        raise ValueError(f"Dictionary has {dictionary.shape[0]} rows, k={k}")
    if dictionary.shape[1] != X.shape[1]:
        raise ValueError(
            f"Dictionary width {dictionary.shape[1]} != X width {X.shape[1]}"
        )
    if len(descriptors) != dictionary.shape[0]:
        raise ValueError(
            f"descriptors length {len(descriptors)} != dictionary rows {dictionary.shape[0]}"
        )

    X = X.to(device)
    dictionary = dictionary.to(device)
    dict_t = dictionary.T.contiguous()

    if compute_evr:
        orig_X = orig_X.to(device)
        orig_X_mean = orig_X.mean(dim=0, keepdim=True)
        orig_X_centered = orig_X - orig_X_mean
        std_orig_sum = (torch.std(orig_X, dim=0) ** 2).sum(dim=-1).clamp_min(1e-12)
    else:
        orig_X_mean = torch.zeros((1, orig_X.shape[1]), device=device, dtype=X.dtype)
        orig_X_centered = torch.empty(0, device=device, dtype=X.dtype)
        std_orig_sum = torch.tensor(1.0, device=device)

    if centering:
        X_mean = X.mean(dim=0, keepdim=True)
        X = X - X_mean
    else:
        X_mean = torch.zeros((1, X.shape[1]), device=device, dtype=X.dtype)

    chosen = torch.zeros(k, dtype=torch.long, device=device)
    notchosen = torch.ones(dictionary.shape[0], device=device)
    residual = X.clone()
    recon = torch.zeros_like(X)
    evr = torch.zeros(k, device=device)
    lstsq_weights = torch.empty((0, X.shape[0]), device=device, dtype=X.dtype)

    l2 = torch.zeros(k, device=device) if return_full else None
    cosine = torch.zeros(k, device=device) if return_full else None

    for i in range(k):
        cross = residual @ dict_t
        if criterion == "l1":
            proj_scores = cross.abs().sum(dim=0)
        elif criterion == "std":
            proj_scores = cross.std(dim=0)
        else:
            raise ValueError(f"Unknown SOMP criterion '{criterion}'")

        atom_idx = (proj_scores * notchosen).argmax()
        chosen[i] = atom_idx
        notchosen[atom_idx] = 0

        current_atoms = torch.index_select(dictionary, 0, chosen[: i + 1])
        if torch.device(device).type == "mps":
            lstsq_weights = torch.linalg.lstsq(
                current_atoms.T.float().cpu(), X.T.float().cpu()
            ).solution.to(device=device, dtype=X.dtype)
        else:
            lstsq_weights = torch.linalg.lstsq(
                current_atoms.T.double(), X.T.double()
            ).solution.to(dtype=X.dtype)

        recon = (current_atoms.T @ lstsq_weights).T
        residual = X - recon

        if compute_evr:
            if pc is None:
                recon_full = X_mean + recon
            else:
                u, _, vh = torch.linalg.svd(recon, full_matrices=False)
                somp_pcs = u @ vh
                recon_full = orig_X_centered @ somp_pcs.T @ somp_pcs + orig_X_mean
            std_recon = torch.std(recon_full, dim=0) ** 2
            evr[i] = std_recon.sum(dim=-1) / std_orig_sum
            if return_full:
                assert cosine is not None and l2 is not None
                cosine[i] = F.cosine_similarity(orig_X, recon_full).mean()
                l2[i] = F.mse_loss(orig_X, recon_full)

    chosen_cpu = chosen.cpu()
    evr_cpu = evr.cpu().float()
    if not return_full:
        return {"chosen": chosen_cpu, "evr": evr_cpu}

    assert l2 is not None and cosine is not None
    weights = lstsq_weights.norm(dim=1).cpu().float()
    order = torch.argsort(weights, descending=True).cpu().numpy()
    results = np.asarray(
        [descriptors[idx] for idx in chosen_cpu.tolist()], dtype=object
    )

    return {
        "recon": (X_mean + recon).cpu().float(),
        "residual": residual.cpu().float(),
        "results": results,
        "chosen": chosen_cpu,
        "weights": weights,
        "weights_full": lstsq_weights.cpu().float().T,
        "order": order,
        "evr": evr_cpu,
        "l2": l2.cpu().float(),
        "cosine": cosine.cpu().float(),
    }
