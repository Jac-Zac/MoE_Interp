"""Tests for cache storage."""

from pathlib import Path

import torch

from src.cache import append_expert_h5, load_expert_h5


class TestExpertStorage:
    """Test per-expert HDF5 append/load."""

    def test_append_and_load_expert_h5(self, tmp_path: Path):
        acts_a = torch.randn(3, 16, dtype=torch.float16)
        toks_a = torch.randint(0, 1000, (3,))
        acts_b = torch.randn(2, 16, dtype=torch.float16)
        toks_b = torch.randint(0, 1000, (2,))

        h5_path = tmp_path / "layer_00.h5"
        append_expert_h5(h5_path, 0, acts_a, toks_a)
        append_expert_h5(h5_path, 0, acts_b, toks_b)

        loaded = load_expert_h5(h5_path, 0)
        assert loaded["activations"].shape == (5, 16)
        assert loaded["tokens"].shape == (5,)
        assert torch.allclose(
            loaded["activations"],
            torch.cat([acts_a, acts_b], dim=0),
            atol=0.01,
        )
        assert torch.equal(loaded["tokens"], torch.cat([toks_a, toks_b], dim=0))

    def test_overwrite_replaces_existing(self, tmp_path: Path):
        acts_a = torch.randn(3, 16, dtype=torch.float16)
        toks_a = torch.randint(0, 1000, (3,))
        acts_b = torch.randn(2, 16, dtype=torch.float16)
        toks_b = torch.randint(0, 1000, (2,))

        h5_path = tmp_path / "layer_00.h5"
        append_expert_h5(h5_path, 0, acts_a, toks_a)
        append_expert_h5(h5_path, 0, acts_b, toks_b, overwrite=True)

        loaded = load_expert_h5(h5_path, 0)
        assert loaded["activations"].shape == (2, 16)
        assert loaded["tokens"].shape == (2,)
        assert torch.allclose(loaded["activations"], acts_b, atol=0.01)
        assert torch.equal(loaded["tokens"], toks_b)
