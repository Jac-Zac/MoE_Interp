"""Tests for greedy projection pursuit."""

import torch
import torch.nn.functional as F

from src.pursuit import projection_pursuit


class _DummyTokenizer:
    def decode(self, token_ids):
        return f"tok_{token_ids[0]}"


def test_projection_pursuit_greedy_monotonic():
    torch.manual_seed(0)
    X = torch.randn(64, 16)
    dictionary = F.normalize(torch.randn(50, 16), dim=1)
    tokenizer = _DummyTokenizer()

    tokens, evr = projection_pursuit(X, dictionary, tokenizer, k=10)

    assert len(tokens) == len(evr)
    assert len(tokens) <= 10
    assert all(0.0 <= val <= 1.0 + 1e-6 for val in evr)


def test_projection_pursuit_empty_on_zero_variance():
    X = torch.zeros(8, 4)
    dictionary = torch.randn(10, 4)
    tokenizer = _DummyTokenizer()

    tokens, evr = projection_pursuit(X, dictionary, tokenizer, k=5)

    assert tokens == []
    assert evr == []


def test_projection_pursuit_empty_on_non_positive_k():
    X = torch.randn(8, 4)
    dictionary = torch.randn(10, 4)
    tokenizer = _DummyTokenizer()

    tokens, evr = projection_pursuit(X, dictionary, tokenizer, k=0)

    assert tokens == []
    assert evr == []
