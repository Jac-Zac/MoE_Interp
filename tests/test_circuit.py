"""Unit tests for the model-free pieces of the circuit comparison (no model needed)."""

from __future__ import annotations

import torch

from moe_interp.circuit import compare, relp


def test_pearson_perfect_and_anti():
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert compare._pearson(a, 2 * a + 1) > 0.999
    assert compare._pearson(a, -a) < -0.999


def test_toxic_relevance_direction_is_relative():
    U = torch.zeros(8, 3)
    U[0, 0] = U[1, 0] = 4.0  # toxic rows point +x
    d = relp.toxic_relevance_direction(U, [0, 1])
    assert d[0] > 0 and torch.allclose(d[1:], torch.zeros(2, dtype=d.dtype))


def test_expert_effect_sums_neurons():
    attr = torch.tensor([[1.0, -2.0, 0.5], [0.0, 0.0, 0.0]])
    assert torch.allclose(relp.expert_effect(attr), torch.tensor([-0.5, 0.0]))


def test_top_neurons_orders_by_abs():
    attr = torch.tensor([[0.1, -0.9], [0.4, 0.0]])
    top = relp.top_neurons(attr, k=2)
    assert top[0] == (0, 1, -0.9000000357627869) or top[0][:2] == (0, 1)


def test_faithfulness_recovers_known_correlation():
    patching = torch.zeros(2, 4)
    patching[0] = torch.tensor([1.0, 2.0, 3.0, 4.0])  # layer 0 scored
    grids = {"good": patching.clone(), "flat": torch.ones(2, 4)}
    scores = compare.faithfulness(grids, patching)
    assert scores["good"]["pooled_r"] > 0.999
    assert abs(scores["flat"]["pooled_r"]) < 1e-6
