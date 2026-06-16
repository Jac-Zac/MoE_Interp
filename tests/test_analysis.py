"""Tests for the post-hoc analyses (no model forward)."""

import torch

from moe_interp.analysis.logit_lens import cumulative_evr


def _cumulative_evr_lstsq(A: torch.Tensor, atoms: torch.Tensor) -> list[float]:
    """Reference: the original per-prefix least-squares formulation of ``cumulative_evr``.

    Kept here as the parity oracle for the faster QR implementation.
    """
    Ac = A - A.mean(dim=0, keepdim=True)
    total = (Ac.var(dim=0, unbiased=False)).sum().clamp_min(1e-12)
    out: list[float] = []
    for m in range(1, atoms.shape[0] + 1):
        Dm = atoms[:m]
        w = torch.linalg.lstsq(Dm.T.double().cpu(), Ac.T.double().cpu()).solution
        recon = (Dm.T.double().cpu() @ w).T
        evr = (recon.var(dim=0, unbiased=False).sum() / total.double().cpu()).item()
        out.append(float(evr))
    return out


def test_cumulative_evr_matches_lstsq_reference():
    """QR-based cumulative_evr must match the lstsq oracle.

    Both compute the orthogonal projection onto ``span(atoms[:m])``, so they agree up to
    floating-point rounding (~1e-8 here — identical to ~8 significant figures, well beyond
    the 3-4 figures ever reported), not bit-for-bit.
    """
    torch.manual_seed(0)
    for n, d, m in [(200, 64, 10), (50, 128, 8), (500, 2048, 10)]:
        A = torch.randn(n, d)
        atoms = torch.nn.functional.normalize(torch.randn(m, d), dim=1)
        fast = cumulative_evr(A, atoms)
        ref = _cumulative_evr_lstsq(A, atoms)
        assert len(fast) == len(ref) == m
        assert torch.allclose(
            torch.tensor(fast), torch.tensor(ref), atol=1e-7, rtol=0
        ), f"\nfast={fast}\n ref={ref}"


def test_cumulative_evr_is_monotone_and_bounded():
    torch.manual_seed(1)
    A = torch.randn(120, 32)
    atoms = torch.nn.functional.normalize(torch.randn(10, 32), dim=1)
    evr = cumulative_evr(A, atoms)
    assert all(0.0 <= v <= 1.0 + 1e-9 for v in evr)
    assert all(evr[i] <= evr[i + 1] + 1e-9 for i in range(len(evr) - 1))
