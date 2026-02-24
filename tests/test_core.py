"""Tests for cache (safetensor storage)."""

from pathlib import Path

import torch

from src.cache import (
    load_expert,
    load_metadata,
    load_unembedding,
    save_expert,
    save_metadata,
    save_unembedding,
)


class TestExpertStorage:
    """Test per-expert safetensor save/load."""

    def test_save_and_load_expert(self, tmp_path: Path):
        activations = torch.randn(10, 128, dtype=torch.float16)
        tokens = torch.randint(0, 1000, (10,))

        save_expert(tmp_path / "expert.safetensors", activations, tokens)
        loaded = load_expert(tmp_path / "expert.safetensors")

        assert loaded["activations"].shape == (10, 128)
        assert loaded["tokens"].shape == (10,)
        assert torch.allclose(loaded["activations"], activations, atol=0.01)
        assert torch.equal(loaded["tokens"], tokens)


class TestMetadata:
    """Test metadata JSON save/load."""

    def test_metadata_round_trip(self, tmp_path: Path):
        save_metadata(tmp_path, n_docs=100, n_layers=16, n_experts=64, d_model=2048)
        meta = load_metadata(tmp_path)

        assert meta["n_docs"] == 100
        assert meta["n_layers"] == 16
        assert meta["n_experts"] == 64
        assert meta["d_model"] == 2048


class TestUnembedding:
    """Test unembedding matrix safetensor save/load."""

    def test_unembedding_round_trip(self, tmp_path: Path):
        unembed = torch.randn(1000, 128)

        save_unembedding(tmp_path / "unembed.safetensors", unembed)
        loaded = load_unembedding(tmp_path / "unembed.safetensors")

        assert loaded.shape == (1000, 128)
        assert torch.allclose(loaded, unembed, atol=1e-6)
