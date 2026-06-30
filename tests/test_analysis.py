"""Tests for the post-hoc analyses (no model forward)."""

import torch

from moe_interp.analysis.logit_lens import direction_evr


def test_direction_evr_extremes():
    """A direction along the only varying axis recovers all variance; orthogonal, none."""
    torch.manual_seed(0)
    n, d = 200, 8
    A = torch.zeros(n, d)
    A[:, 0] = torch.randn(n)  # variance lives only on axis 0
    e0 = torch.zeros(d)
    e0[0] = 1.0
    e1 = torch.zeros(d)
    e1[1] = 1.0
    assert abs(direction_evr(A, e0) - 1.0) < 1e-9
    assert direction_evr(A, e1) < 1e-9


def test_direction_evr_matches_projection_fraction():
    """EVR of a direction equals its squared-projection share of total variance."""
    torch.manual_seed(1)
    A = torch.randn(150, 16).double()
    u = torch.randn(16).double()
    Ac = A - A.mean(dim=0, keepdim=True)
    un = u / u.norm()
    expected = ((Ac @ un).pow(2).sum() / (Ac**2).sum()).item()
    assert abs(direction_evr(A, u) - expected) < 1e-9
    assert 0.0 <= direction_evr(A, u) <= 1.0 + 1e-9


def test_direction_evr_is_scale_invariant():
    torch.manual_seed(2)
    A = torch.randn(100, 12)
    u = torch.randn(12)
    assert abs(direction_evr(A, u) - direction_evr(A, 5.0 * u)) < 1e-9
