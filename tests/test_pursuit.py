"""Tests for greedy projection pursuit."""

import torch
import torch.nn.functional as F

from src.pursuit import projection_pursuit
from src.sparse_decomposition import somp


class _DummyTokenizer:
    def decode(self, token_ids):
        return f"tok_{token_ids[0]}"


def test_projection_pursuit_greedy_monotonic():
    torch.manual_seed(0)
    X = torch.randn(64, 16)
    dictionary = F.normalize(torch.randn(50, 16), dim=1)
    tokenizer = _DummyTokenizer()

    tokens, evr = projection_pursuit(X, dictionary, tokenizer, device="cpu", k=10)

    assert len(tokens) == len(evr)
    assert len(tokens) <= 10
    assert all(0.0 <= val <= 1.0 + 1e-6 for val in evr)
    assert all(evr[i] <= evr[i + 1] for i in range(len(evr) - 1))


def test_projection_pursuit_empty_on_zero_variance():
    X = torch.zeros(8, 4)
    dictionary = torch.randn(10, 4)
    tokenizer = _DummyTokenizer()

    tokens, evr = projection_pursuit(X, dictionary, tokenizer, device="cpu", k=5)

    assert tokens == []
    assert evr == []


def test_projection_pursuit_empty_on_non_positive_k():
    X = torch.randn(8, 4)
    dictionary = torch.randn(10, 4)
    tokenizer = _DummyTokenizer()

    tokens, evr = projection_pursuit(X, dictionary, tokenizer, device="cpu", k=0)

    assert tokens == []
    assert evr == []


def test_somp_residual_shrinks():
    """Residual norm must decrease (or stay equal) at every SOMP step."""
    torch.manual_seed(42)
    n, d, vocab, k = 32, 16, 60, 8
    X = torch.randn(n, d).double()
    dictionary = F.normalize(torch.randn(vocab, d), dim=1).double()
    descriptors = list(range(vocab))

    result = somp(
        X=X,
        orig_X=X,
        pc=None,
        dictionary=dictionary,
        descriptors=descriptors,
        k=k,
        device="cpu",
        compute_evr=True,
    )

    # residual should be in original space: orig_X - recon
    res_norm = torch.as_tensor(result["residual"]).float().norm().item()
    total_norm = X.float().norm().item()
    assert res_norm < total_norm, "Residual should be smaller than original signal"

    # EVR must be monotonically non-decreasing
    evr = result["evr"].tolist()
    assert all(
        evr[i] <= evr[i + 1] + 1e-6 for i in range(len(evr) - 1)
    ), f"EVR not monotone: {evr}"

    # chosen atoms must be unique
    chosen = result["chosen"].tolist()
    assert len(chosen) == len(set(chosen)), "SOMP selected duplicate atoms"
