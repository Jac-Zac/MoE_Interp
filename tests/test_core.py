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


class TestPursuitResult:
    """Test PursuitResult save/load."""

    def test_save_and_load(self, tmp_path: Path):
        """PursuitResult round-trips through JSON + .pt files."""
        from src.pursuit import ExpertConceptResult, PursuitResult

        result = PursuitResult(
            n_layers=2,
            n_experts=4,
            k=3,
            property_name="test",
            experts=[
                ExpertConceptResult(
                    layer=0,
                    expert_id=1,
                    tokens=["foo", "bar", "baz"],
                    token_ids=[10, 20, 30],
                    evr=[0.1, 0.5, 0.8],
                    zscore=1.5,
                ),
            ],
            evr_matrix=torch.randn(2, 4, 3),
            zscore_matrix=torch.randn(2, 4),
        )

        out_dir = tmp_path / "pursuit_out"
        result.save(out_dir)

        loaded = PursuitResult.load(out_dir)
        assert loaded.n_layers == 2
        assert loaded.n_experts == 4
        assert loaded.k == 3
        assert loaded.property_name == "test"
        assert len(loaded.experts) == 1
        assert loaded.experts[0].tokens == ["foo", "bar", "baz"]
        assert loaded.experts[0].zscore == 1.5
        assert torch.allclose(loaded.evr_matrix, result.evr_matrix)
        assert torch.allclose(loaded.zscore_matrix, result.zscore_matrix)

    def test_concept_frequency(self):
        """concept_frequency aggregates top-N tokens across experts."""
        from src.pursuit import ExpertConceptResult, PursuitResult

        result = PursuitResult(
            n_layers=1,
            n_experts=2,
            k=3,
            property_name="test",
            experts=[
                ExpertConceptResult(
                    layer=0,
                    expert_id=0,
                    tokens=["a", "b", "c"],
                    token_ids=[1, 2, 3],
                    evr=[0.1, 0.2, 0.3],
                ),
                ExpertConceptResult(
                    layer=0,
                    expert_id=1,
                    tokens=["a", "d", "e"],
                    token_ids=[1, 4, 5],
                    evr=[0.1, 0.2, 0.3],
                ),
            ],
        )

        freq = result.concept_frequency(top_n=2)
        assert freq["a"] == 2  # appears in both experts' top-2
        assert freq["b"] == 1
        assert freq["d"] == 1
        assert "c" not in freq  # not in top-2
