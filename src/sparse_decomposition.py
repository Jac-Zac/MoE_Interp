"""Sparse decomposition utilities (SOMP, OMP, PCA)."""

# This code is mostly taken from this repo https://github.com/Flegyas/ResiDual/blob/main/src/residual/sparse_decomposition.py
# WARNING: The improvements are AI Generated and should be reviewd
# It has some improvements to make it faster

from abc import abstractmethod
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def jaccard_similarity(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union


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


def _as_float_tensor(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    return value.to(device).float()


def pca(X, k, compute_evr: bool = True, *args, **kwargs) -> dict:
    X = X.float()
    # Center the data by subtracting the mean of each feature
    X_mean = torch.mean(X, dim=0)
    X_centered = X - X_mean

    # Compute the SVD of the centered data
    U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)

    # Select the number of components we want to keep
    components = Vt[:k]  # Transpose Vt to get right singular vectors in columns
    # explained_variance = S**2 / (X.size(0) - 1)  # Variance explained by each singular value
    # explained_variance_ratio = explained_variance[:k] / explained_variance.sum()

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
    else:
        evr = l2 = cosine = torch.zeros(k)

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

    device = torch.device(device)
    X = _as_float_tensor(X, device)
    dictionary = _as_float_tensor(dictionary, device)
    orig_X = _as_float_tensor(orig_X, device)

    chosen = []
    notchosen = torch.ones(dictionary.shape[0], device=device)
    results = []
    recon = torch.zeros_like(X)
    residual = X.clone()
    evr = torch.zeros(k)
    cosine = torch.zeros(k)
    l2 = torch.zeros(k)
    lstsq_weights = torch.zeros((dictionary.shape[0], 0), device=device)
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
            dictionary, 0, torch.as_tensor(chosen, device=device)
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
    ):
        super().__init__(name="somp")
        self.k = k
        self.criterion = criterion
        self.pc = pc
        self.compute_evr = compute_evr

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
            X=X.double(),
            orig_X=orig_X.double(),
            pc=self.pc,
            dictionary=dictionary.double(),
            descriptors=descriptors,
            k=self.k,
            device=device,
            criterion=self.criterion,
            compute_evr=self.compute_evr,
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
    orig_X = orig_X.to(device)
    orig_X_mean = orig_X.mean(dim=0)
    orig_X_centered = orig_X - orig_X_mean
    dictionary = dictionary.to(device)

    std_orig = torch.std(orig_X, dim=0) ** 2
    if centering:
        X_mean = X.mean(dim=0, keepdims=True)
        X = X - X_mean
    else:
        X_mean = torch.zeros_like(X)

    chosen = []
    notchosen = torch.ones(dictionary.shape[0]).to(device)
    results = []
    recon = torch.zeros_like(X)  # +X_mean
    residual = X.clone()
    evr = torch.zeros(k)
    l2 = torch.zeros(k)
    cosine = torch.zeros(k)
    lstsq_weights = torch.zeros((0, X.shape[0]), device=device, dtype=X.dtype)
    for i in range(k):
        cross = residual @ dictionary.T
        cross = cross * notchosen

        if criterion == "l1":
            proj_scores = torch.sum(cross.abs(), dim=0)
        elif criterion == "std":
            proj_scores = torch.std(cross, dim=0)
        else:
            raise ValueError(f"Criterion {criterion} not recognized")

        atom_idx = proj_scores.argmax()

        chosen.append(atom_idx.item())
        notchosen[atom_idx] = 0
        results.append(descriptors[atom_idx])
        current_atoms = torch.index_select(
            dictionary, 0, torch.as_tensor(chosen).to(device)
        )
        lstsq_weights = torch.linalg.lstsq(current_atoms.T, X.T).solution
        recon = (current_atoms.T @ lstsq_weights).T

        residual = X - recon
        if pc is None:
            std_recon = torch.std((X_mean + recon), dim=0) ** 2
            evr[i] = std_recon.sum(dim=-1) / std_orig.sum(dim=-1)
            cosine[i] = F.cosine_similarity(orig_X, X_mean + recon).mean().item()
            l2[i] = F.mse_loss(orig_X, X_mean + recon).item()
        else:
            u, _, v = torch.linalg.svd(recon, full_matrices=False)
            somp_pcs = u @ v
            X_recon = orig_X_centered @ somp_pcs.T @ somp_pcs + orig_X_mean
            std_recon = torch.std(X_recon, dim=0) ** 2
            evr[i] = std_recon.sum(dim=-1) / std_orig.sum(dim=-1)
            cosine[i] = F.cosine_similarity(orig_X, X_recon).mean().item()
            l2[i] = F.mse_loss(orig_X, X_recon).item()

    results = np.asarray(results, dtype=object)
    weights = lstsq_weights.norm(dim=1).cpu()
    order = torch.argsort(weights, descending=True).cpu().numpy()
    chosen = torch.tensor(chosen).cpu()

    recon = X_mean + recon
    residual = X - recon

    return dict(
        recon=recon.cpu().float(),
        residual=residual.cpu().float(),
        results=results,
        chosen=chosen,
        weights=weights.float(),
        weights_full=lstsq_weights.float().T,
        order=order,
        evr=evr.float(),
        l2=l2.float(),
        cosine=cosine.float(),
    )
