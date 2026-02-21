"""Tests for cache (HDF5 storage) and pursuit modules."""

from pathlib import Path

import torch

from src.cache import (
    load_layer,
    load_metadata,
    load_unembedding,
    save_layer,
    save_metadata,
    save_unembedding,
)


class TestLayerStorage:
    """Test per-layer HDF5 save/load."""

    def test_save_and_load_layer(self, tmp_path: Path):
        """Round-trip per-layer activations through HDF5."""
        n_docs, n_experts, d_model = 5, 4, 16
        data = torch.randn(n_docs, n_experts, d_model)

        save_layer(tmp_path, layer=0, data=data)
        loaded = load_layer(tmp_path, layer=0)

        assert loaded.shape == (n_docs, n_experts, d_model)
        # float16 round-trip tolerance
        assert torch.allclose(loaded, data, atol=0.01)

    def test_multiple_layers(self, tmp_path: Path):
        """Save and load multiple layers independently."""
        n_docs, n_experts, d_model = 3, 4, 8
        n_layers = 3

        originals = []
        for li in range(n_layers):
            data = torch.randn(n_docs, n_experts, d_model)
            save_layer(tmp_path, li, data)
            originals.append(data)

        for li in range(n_layers):
            loaded = load_layer(tmp_path, li)
            assert loaded.shape == (n_docs, n_experts, d_model)
            assert torch.allclose(loaded, originals[li], atol=0.01)


class TestMetadata:
    """Test metadata JSON save/load."""

    def test_metadata_round_trip(self, tmp_path: Path):
        """Metadata values are preserved through JSON."""
        save_metadata(tmp_path, n_docs=100, n_layers=16, n_experts=64, d_model=2048)
        meta = load_metadata(tmp_path)

        assert meta["n_docs"] == 100
        assert meta["n_layers"] == 16
        assert meta["n_experts"] == 64
        assert meta["d_model"] == 2048


class TestUnembedding:
    """Test unembedding matrix HDF5 save/load."""

    def test_unembedding_round_trip(self, tmp_path: Path):
        """Unembedding matrix survives HDF5 round-trip."""
        unembed = torch.randn(1000, 128)

        save_unembedding(tmp_path, unembed)
        loaded = load_unembedding(tmp_path)

        assert loaded.shape == (1000, 128)
        assert torch.allclose(loaded, unembed, atol=1e-6)
