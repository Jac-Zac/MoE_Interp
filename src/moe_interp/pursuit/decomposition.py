"""
Code taken and slightly readapted from the ResiDual repo
https://github.com/Flegyas/ResiDual/blob/004b0aac16a74e73a5ac29a47b76c9f5b39531fc/src/residual/sparse_decomposition.py#L353
"""

from abc import abstractmethod
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Projection(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    @abstractmethod
    def forward(
        self,
        X: torch.Tensor,
        dictionary: torch.Tensor,
        descriptors: list,
        k: int,
        device,
        *args,
        **kwargs,
    ):
        raise NotImplementedError

    @property
    def key(self):
        return self.name


class PCA(Projection):
    def __init__(self, k: int, compute_evr: bool = True):
        super().__init__("pca")
        self.k = k
        self.compute_evr = compute_evr

    @property
    def key(self):
        return f"{self.name}_{self.k}"

    def forward(self, X: torch.Tensor, *args, **kwargs):
        return pca(*args, X=X, k=self.k, compute_evr=self.compute_evr, **kwargs)


def pca(X, k, compute_evr: bool = True, *args, **kwargs):
    # Center the data by subtracting the mean of each feature
    X_mean = torch.mean(X, dim=0)
    X_centered = X - X_mean

    # Compute the SVD of the centered data
    U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)

    # Select the number of components we want to keep
    components = Vt[:k]  # Transpose Vt to get right singular vectors in columns
    # explained_variance = S**2 / (X.size(0) - 1)  # Variance explained by each singular value
    # explained_variance_ratio = explained_variance[:k] / explained_variance.sum()

    evr = l2 = cosine = None
    if compute_evr:
        std_orig = torch.std(X, dim=0) ** 2

        evr = torch.zeros(k)
        l2 = torch.zeros(k)
        cosine = torch.zeros(k)
        for i in range(1, k + 1):
            filtering = components[:i].T @ components[:i]
            recon_i = X_centered @ filtering + X_mean
            std_recon = torch.std(recon_i, dim=0) ** 2
            evr[i - 1] = std_recon.sum(dim=-1) / std_orig.sum(dim=-1)
            cosine[i - 1] = F.cosine_similarity(X, recon_i).mean().item()
            l2[i - 1] = F.mse_loss(X, recon_i).item()

    recon = X_centered @ components.T @ components + torch.mean(X, dim=0)
    results = []
    chosen = -torch.ones(k, dtype=torch.long)
    weights = S[:k]

    return dict(
        recon=recon,
        results=results,
        chosen=chosen,
        weights=weights,
        order=None,
        components=components,
        evr=evr,
        l2=l2,
        cosine=cosine,
    )


class OMP(Projection):
    def __init__(self, k: int):
        super().__init__("omp")
        self.k = k

    def forward(
        self,
        X: torch.Tensor,
        dictionary: torch.Tensor,
        descriptors: list,
        device,
        *args,
        **kwargs,
    ):
        x_pc = pca(X, 1)["components"][0]
        omp_result = omp(
            *args,
            X=x_pc,
            orig_X=X,
            dictionary=dictionary,
            descriptors=descriptors,
            k=self.k,
            device=device,
            **kwargs,
        )
        return omp_result


@torch.no_grad()
def omp(
    X: torch.Tensor,
    orig_X: torch.Tensor,
    dictionary: torch.Tensor,
    descriptors: list,
    k: int,
    device,
    compute_evr: bool = False,
    *args,
    **kwargs,
):
    assert dictionary.shape[1] == X.shape[0], (
        f"Dictionary: {dictionary.shape[1]}, X: {X.shape[0]}"
    )
    assert len(descriptors) == dictionary.shape[0], (
        f"descriptors: {len(descriptors)}, Dictionary: {dictionary.shape[0]}"
    )

    X = X.to(device)
    dictionary = dictionary.to(device)
    # NOTE: Since it is already normalized when saved
    # dictionary = torch.nn.functional.normalize(dictionary, dim=1)
    chosen = []
    notchosen = torch.ones(dictionary.shape[0]).to(device)
    results = []
    recon = torch.zeros_like(X)  # +X_mean
    residual = X.clone()
    evr = torch.zeros(k)
    cosine = torch.zeros(k)
    l2 = torch.zeros(k)
    std_orig = torch.std(orig_X, dim=0) ** 2
    for j in range(k):
        cross = residual @ dictionary.T
        cross = cross * notchosen
        proj_std = cross.abs()
        atom_idx = proj_std.argmax()
        chosen.append(atom_idx.item())
        notchosen[atom_idx] = 0
        results.append(descriptors[atom_idx])
        current_atoms = torch.index_select(
            dictionary, 0, torch.as_tensor(chosen).to(device)
        )
        lstsq_weights = torch.linalg.lstsq(
            current_atoms.T.double(), X.double()
        ).solution.float()
        recon = current_atoms.T @ lstsq_weights
        residual = X - recon
        recon_X = (
            orig_X
            @ (recon / recon.norm()).reshape(-1, 1)
            @ (recon / recon.norm()).reshape(1, -1)
        )
        std_recon = torch.std(recon_X, dim=0) ** 2
        evr[j] = std_recon.sum(dim=-1) / std_orig.sum(dim=-1)
        cosine[j] = F.cosine_similarity(orig_X, recon_X).mean().item()
        l2[j] = F.mse_loss(orig_X, recon_X).item()
    results = np.array(results, dtype="object")
    chosen = torch.as_tensor(chosen, dtype=torch.long)
    return dict(
        recon=recon,
        results=results,
        chosen=chosen,
        weights=lstsq_weights.abs().cpu(),
        order=None,
        evr=evr,
        l2=l2,
        cosine=cosine,
    )


class SOMP(Projection):
    def __init__(
        self,
        k: int,
        criterion="l1",
        pc: Optional[int] = None,
        compute_evr: bool = False,
        return_full: bool = True,
    ):
        super().__init__(name="somp")
        self.k = k
        self.criterion = criterion
        self.pc = pc
        self.compute_evr = compute_evr
        self.return_full = return_full

    @property
    def key(self):
        return f"{self.name}_{self.k}_{self.criterion}_{self.pc}"

    def forward(
        self,
        X: torch.Tensor,
        dictionary: torch.Tensor,
        descriptors: list,
        device,
        *args,
        **kwargs,
    ):
        orig_X = X
        # X_mean = torch.mean(X, dim=0)
        # X_centered = X - X_mean

        if self.pc is not None:
            pca_out = pca(X, self.pc)
            weights = pca_out["weights"].unsqueeze(1)
            X = pca_out["components"] * weights**2

        result = somp(
            X=X,
            orig_X=orig_X,
            pc=self.pc,
            dictionary=dictionary,
            descriptors=descriptors,
            k=self.k,
            device=device,
            criterion=self.criterion,
            compute_evr=self.compute_evr,
            return_full=self.return_full,
        )

        return result


@torch.no_grad()
def somp(
    X: torch.Tensor,
    orig_X: torch.Tensor,
    pc,
    dictionary: torch.Tensor,
    descriptors: list,
    k: int,
    device,
    criterion="l1",
    centering: bool = True,
    compute_evr: bool = False,
    return_full: bool = True,
    *args,
    **kwargs,
):
    assert dictionary.shape[0] >= k, f"Dictionary: {dictionary.shape[0]}, k: {k}"
    assert dictionary.shape[1] == X.shape[1], (
        f"Dictionary: {dictionary.shape[1]}, X: {X.shape[1]}"
    )
    assert len(descriptors) == dictionary.shape[0], (
        f"descriptors: {len(descriptors)}, Dictionary: {dictionary.shape[0]}"
    )

    X = X.to(device)
    dictionary = dictionary.to(device)
    # PERF: Materialize the transposed dictionary once outside the selection loop.
    dict_T = dictionary.T.contiguous()

    if compute_evr:
        orig_X = orig_X.to(device)
        orig_X_mean = orig_X.mean(dim=0)
        orig_X_centered = orig_X - orig_X_mean
        std_orig = torch.std(orig_X, dim=0) ** 2
    else:
        orig_X_mean = torch.zeros(X.shape[1], device=device, dtype=X.dtype)
        orig_X_centered = torch.zeros_like(X)
        std_orig = torch.ones(X.shape[1], device=device, dtype=X.dtype)

    if centering:
        X_mean = X.mean(dim=0, keepdim=True)
        X = X - X_mean
    else:
        X_mean = torch.zeros_like(X)

    # PERF: Keep selected indices on device instead of rebuilding them each step.
    chosen = torch.zeros(k, dtype=torch.long, device=device)

    notchosen = torch.ones(dictionary.shape[0], device=device)
    recon = torch.zeros_like(X)  # +X_mean
    residual = X.clone()
    evr = torch.zeros(k, device=device)
    lstsq_weights = torch.empty((0, X.shape[0]), device=device, dtype=X.dtype)
    if return_full:
        l2 = torch.zeros(k, device=device)
        cosine = torch.zeros(k, device=device)
    else:
        l2 = None
        cosine = None
    std_orig_sum = (
        std_orig.sum(dim=-1) if compute_evr else torch.tensor(1.0, device=device)
    )
    for i in range(k):
        cross = residual @ dict_T

        if criterion == "l1":
            proj_scores = torch.sum(cross.abs(), dim=0)
        elif criterion == "std":
            proj_scores = torch.std(cross, dim=0)
        else:
            raise ValueError(f"Criterion {criterion} not recognized")
        # PERF: Mask the reduced scores instead of the full cross matrix.
        proj_scores = proj_scores * notchosen

        atom_idx = proj_scores.argmax()

        chosen[i] = atom_idx
        notchosen[atom_idx] = 0

        # PERF: Reuse the on-device prefix instead of recreating an index tensor.
        current_atoms = torch.index_select(dictionary, 0, chosen[: i + 1])
        lstsq_weights = torch.linalg.lstsq(
            current_atoms.T.double(), X.T.double()
        ).solution.to(dtype=X.dtype)
        recon = (current_atoms.T @ lstsq_weights).T

        residual = X - recon
        if compute_evr:
            if pc is None:
                std_recon = torch.std((X_mean + recon), dim=0) ** 2
                evr[i] = std_recon.sum(dim=-1) / std_orig_sum
                if return_full:
                    assert cosine is not None and l2 is not None
                    cosine[i] = F.cosine_similarity(orig_X, X_mean + recon).mean()
                    l2[i] = F.mse_loss(orig_X, X_mean + recon)
            else:
                assert orig_X_centered is not None and orig_X_mean is not None
                u, _, v = torch.linalg.svd(recon, full_matrices=False)
                somp_pcs = u @ v
                X_recon = orig_X_centered @ somp_pcs.T @ somp_pcs + orig_X_mean
                std_recon = torch.std(X_recon, dim=0) ** 2
                evr[i] = std_recon.sum(dim=-1) / std_orig_sum
                if return_full:
                    assert cosine is not None and l2 is not None
                    cosine[i] = F.cosine_similarity(orig_X, X_recon).mean()
                    l2[i] = F.mse_loss(orig_X, X_recon)

    chosen = chosen.cpu()
    evr = evr.cpu().float()
    if not return_full:
        return {
            "chosen": chosen,
            "evr": evr,
        }

    weights = lstsq_weights.norm(dim=1).cpu()
    # weights = lstsq_weights.mean(dim=1).cpu()
    order = torch.argsort(weights, descending=True).cpu().numpy()

    # PERF: Convert selected indices once after the loop to avoid per-step syncs.
    results = np.asarray([descriptors[idx] for idx in chosen.tolist()], dtype=object)

    recon = X_mean + recon
    residual = X - recon
    assert l2 is not None and cosine is not None

    return dict(
        recon=recon.cpu().float(),
        residual=residual.cpu().float(),
        results=results,
        chosen=chosen,
        weights=weights.float(),
        weights_full=lstsq_weights.cpu().float().T,
        order=order,
        evr=evr,
        # return_full guarantees these metrics were allocated and filled.
        l2=l2.cpu().float(),
        cosine=cosine.cpu().float(),
    )
