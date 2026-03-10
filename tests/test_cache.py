"""Tests for HDF5 cache helpers."""

import torch

from src.cache import append_expert_h5, load_layer_h5


def test_load_layer_h5_reads_one_layer_file(tmp_path):
    layer_path = tmp_path / "layer_00.h5"
    acts_0 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    acts_2 = torch.tensor([[5.0, 6.0]])

    append_expert_h5(layer_path, 0, acts_0, torch.tensor([10, 11]))
    append_expert_h5(layer_path, 2, acts_2, torch.tensor([12]))

    loaded = load_layer_h5(tmp_path, layer_idx=0, n_experts=4, min_activations=2)

    assert set(loaded) == {0}
    assert torch.equal(loaded[0], acts_0)


def test_load_layer_h5_returns_all_matching_experts(tmp_path):
    layer_path = tmp_path / "layer_01.h5"
    acts_1 = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    acts_3 = torch.tensor([[3.0, 3.0], [4.0, 4.0], [5.0, 5.0]])

    append_expert_h5(layer_path, 1, acts_1, torch.tensor([1, 2]))
    append_expert_h5(layer_path, 3, acts_3, torch.tensor([3, 4, 5]))

    loaded = load_layer_h5(tmp_path, layer_idx=1, n_experts=5, min_activations=1)

    assert set(loaded) == {1, 3}
    assert torch.equal(loaded[1], acts_1)
    assert torch.equal(loaded[3], acts_3)
