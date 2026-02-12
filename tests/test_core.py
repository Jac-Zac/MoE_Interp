"""Tests for SOMP algorithm, ExpertActivationStore, and pipeline utilities."""

from pathlib import Path

import torch
import torch.nn.functional as F

from src.cache import ExpertActivationStore
from src.capture import _aggregate_document
from src.dictionary import CONCEPTS, make_concept_dictionary
from src.pursuit import print_top_experts
from src.somp import somp


class TestSOMP:
    """Test SOMP with synthetic data where we know the answer."""

    def test_recovers_single_direction(self):
        """SOMP should find a dictionary atom that matches a 1D signal."""
        d = 64
        n_samples = 50

        # Dictionary: random normalized vectors
        dictionary = F.normalize(torch.randn(100, d), dim=-1)

        # Signal: varying magnitudes along dictionary atom 42
        # After centering, the variance is along atom 42's direction
        target = dictionary[42]
        coeffs = torch.randn(n_samples, 1)  # varying coefficients
        X = coeffs * target.unsqueeze(0) + 0.01 * torch.randn(n_samples, d)

        result = somp(X, dictionary, k=5)

        # First chosen atom should be 42
        assert result["chosen"][0].item() == 42
        # EVR should be very high after first atom
        assert result["evr"][0].item() > 0.9

    def test_recovers_two_directions(self):
        """SOMP should find both directions in a 2-component signal."""
        d = 64
        n_samples = 100

        dictionary = F.normalize(torch.randn(200, d), dim=-1)

        # Signal: varying mix of atoms 10 and 50
        c1 = torch.randn(n_samples, 1)
        c2 = torch.randn(n_samples, 1)
        X = (
            3.0 * c1 * dictionary[10].unsqueeze(0)
            + 1.0 * c2 * dictionary[50].unsqueeze(0)
            + 0.01 * torch.randn(n_samples, d)
        )

        result = somp(X, dictionary, k=5)

        top2 = set(result["chosen"][:2].tolist())
        assert 10 in top2
        assert 50 in top2

    def test_evr_monotonically_increases(self):
        """EVR should increase (or stay same) with each new atom."""
        d = 32
        X = torch.randn(30, d)
        dictionary = F.normalize(torch.randn(50, d), dim=-1)

        result = somp(X, dictionary, k=10)

        for i in range(1, 10):
            assert result["evr"][i] >= result["evr"][i - 1] - 1e-6

    def test_zero_signal_returns_zeros(self):
        """Zero-variance signal should return zero EVR."""
        d = 32
        X = torch.ones(20, d)  # constant signal, zero variance after centering

        dictionary = F.normalize(torch.randn(50, d), dim=-1)
        result = somp(X, dictionary, k=5)

        assert result["evr"].abs().max() < 1e-6

    def test_output_shapes(self):
        """Check output tensor shapes."""
        k = 7
        result = somp(
            X=torch.randn(25, 16),
            dictionary=F.normalize(torch.randn(40, 16), dim=-1),
            k=k,
        )

        assert result["chosen"].shape == (k,)
        assert result["evr"].shape == (k,)
        assert result["weights"].shape == (k,)


class TestExpertActivationStore:
    """Test HDF5-backed storage with synthetic data."""

    def test_write_and_read(self, tmp_path: Path):
        """Write documents and read back correctly."""
        n_layers, n_experts, d_model = 2, 4, 8
        n_docs = 3

        with ExpertActivationStore(
            tmp_path, n_layers, n_experts, d_model, n_docs_estimate=10
        ) as store:
            docs_data = []
            for i in range(n_docs):
                means = torch.randn(n_layers, n_experts, d_model)
                counts = torch.randint(0, 10, (n_layers, n_experts))
                store.add_document(means, counts, doc_id=i * 10)
                docs_data.append(means)

            assert store.n_docs == n_docs

        # Read back
        for layer in range(n_layers):
            layer_data = ExpertActivationStore.load_layer(tmp_path, layer)
            assert layer_data.shape == (n_docs, n_experts, d_model)

            # Check values match (float16 precision)
            for doc_idx in range(n_docs):
                expected = docs_data[doc_idx][layer].float()
                actual = layer_data[doc_idx]
                # float16 round-trip tolerance
                assert torch.allclose(actual, expected, atol=0.01)

    def test_load_single_expert(self, tmp_path: Path):
        """load_expert returns correct shape and values."""
        n_layers, n_experts, d_model = 2, 4, 16

        with ExpertActivationStore(tmp_path, n_layers, n_experts, d_model) as store:
            means = torch.randn(n_layers, n_experts, d_model)
            store.add_document(means, torch.ones(n_layers, n_experts), doc_id=0)

        expert_data = ExpertActivationStore.load_expert(tmp_path, layer=1, expert_id=2)
        assert expert_data.shape == (1, d_model)

    def test_stream_experts(self, tmp_path: Path):
        """stream_experts yields all experts in order."""
        n_layers, n_experts, d_model = 1, 3, 8

        with ExpertActivationStore(tmp_path, n_layers, n_experts, d_model) as store:
            store.add_document(
                torch.randn(n_layers, n_experts, d_model),
                torch.ones(n_layers, n_experts),
                doc_id=0,
            )

        expert_ids = []
        for eid, data in ExpertActivationStore.stream_experts(tmp_path, layer=0):
            expert_ids.append(eid)
            assert data.shape == (1, d_model)

        assert expert_ids == [0, 1, 2]

    def test_routing_counts(self, tmp_path: Path):
        """Routing counts are stored and loaded correctly."""
        n_layers, n_experts, d_model = 2, 4, 8

        counts_in = torch.tensor([[5, 3, 0, 7], [2, 8, 1, 4]])

        with ExpertActivationStore(tmp_path, n_layers, n_experts, d_model) as store:
            store.add_document(
                torch.randn(n_layers, n_experts, d_model),
                counts_in,
                doc_id=42,
            )

        counts_out = ExpertActivationStore.load_routing_counts(tmp_path)
        assert counts_out.shape == (n_layers, 1, n_experts)
        assert torch.equal(counts_out[:, 0, :], counts_in.long())

    def test_metadata_and_doc_ids(self, tmp_path: Path):
        """Metadata and doc IDs are saved correctly."""
        n_layers, n_experts, d_model = 2, 4, 8

        with ExpertActivationStore(tmp_path, n_layers, n_experts, d_model) as store:
            for i in range(3):
                store.add_document(
                    torch.randn(n_layers, n_experts, d_model),
                    torch.ones(n_layers, n_experts),
                    doc_id=i * 100,
                )

        meta = ExpertActivationStore.load_metadata(tmp_path)
        assert meta["n_layers"] == n_layers
        assert meta["n_experts"] == n_experts
        assert meta["n_docs"] == 3

        doc_ids = ExpertActivationStore.load_doc_ids(tmp_path)
        assert doc_ids == [0, 100, 200]


class TestAggregateDocument:
    """Test per-expert mean gated output computation."""

    def test_single_expert_single_token(self):
        """One expert processes one token: mean == gated output."""
        n_layers, n_experts, d_model = 1, 4, 8
        seq_len, k = 5, 2

        expert_indices = torch.zeros(n_layers, seq_len, k, dtype=torch.long)
        expert_weights = torch.ones(n_layers, seq_len, k)

        # Expert 2 processes token 3 at top-k position 0
        raw_output = torch.randn(1, d_model)
        gate_weight = 0.7
        expert_weights[0, 3, 0] = gate_weight

        means, counts = _aggregate_document(
            expert_indices=expert_indices,
            expert_weights=expert_weights,
            active_experts_per_layer=[torch.tensor([2])],
            token_indices_per_layer=[[torch.tensor([3])]],
            raw_outputs_per_layer=[[raw_output]],
            top_k_pos_per_layer=[[torch.tensor([0])]],
            n_experts=n_experts,
            d_model=d_model,
        )

        assert means.shape == (n_layers, n_experts, d_model)
        assert counts.shape == (n_layers, n_experts)
        expected = gate_weight * raw_output
        assert torch.allclose(means[0, 2], expected.squeeze(0), atol=1e-6)
        assert counts[0, 2] == 1
        # Other experts should be zero
        assert means[0, 0].abs().sum() == 0
        assert means[0, 1].abs().sum() == 0
        assert means[0, 3].abs().sum() == 0

    def test_multiple_tokens_same_expert(self):
        """Expert with multiple tokens: mean is average of gated outputs."""
        n_layers, n_experts, d_model = 1, 2, 4
        seq_len, k = 10, 2

        expert_indices = torch.zeros(n_layers, seq_len, k, dtype=torch.long)
        expert_weights = torch.ones(n_layers, seq_len, k)

        # Expert 0 processes tokens 1, 4, 7 at top-k position 0
        token_idxs = torch.tensor([1, 4, 7])
        raw_outputs = torch.randn(3, d_model)
        gate_weights_vals = torch.tensor([0.5, 0.8, 0.3])
        for i, tidx in enumerate(token_idxs):
            expert_weights[0, tidx, 0] = gate_weights_vals[i]

        means, counts = _aggregate_document(
            expert_indices=expert_indices,
            expert_weights=expert_weights,
            active_experts_per_layer=[torch.tensor([0])],
            token_indices_per_layer=[[token_idxs]],
            raw_outputs_per_layer=[[raw_outputs]],
            top_k_pos_per_layer=[[torch.tensor([0, 0, 0])]],
            n_experts=n_experts,
            d_model=d_model,
        )

        gated = gate_weights_vals.unsqueeze(-1) * raw_outputs
        expected_mean = gated.mean(dim=0)
        assert torch.allclose(means[0, 0], expected_mean, atol=1e-6)
        assert counts[0, 0] == 3

    def test_empty_expert_stays_zero(self):
        """Expert with no tokens should have zero mean and count."""
        n_layers, n_experts, d_model = 1, 4, 8
        seq_len, k = 5, 2

        expert_indices = torch.zeros(n_layers, seq_len, k, dtype=torch.long)
        expert_weights = torch.ones(n_layers, seq_len, k)

        # No active experts at all
        means, counts = _aggregate_document(
            expert_indices=expert_indices,
            expert_weights=expert_weights,
            active_experts_per_layer=[torch.tensor([1])],
            token_indices_per_layer=[[torch.tensor([], dtype=torch.long)]],
            raw_outputs_per_layer=[[torch.zeros(0, d_model)]],
            top_k_pos_per_layer=[[torch.tensor([], dtype=torch.long)]],
            n_experts=n_experts,
            d_model=d_model,
        )

        assert means.abs().sum() == 0
        assert counts.sum() == 0

    def test_output_shape_multilayer(self):
        """Output shapes are correct with multiple layers."""
        n_layers, n_experts, d_model = 3, 8, 16
        seq_len, k = 20, 4

        expert_indices = torch.randint(0, n_experts, (n_layers, seq_len, k))
        expert_weights = torch.rand(n_layers, seq_len, k)

        # One active expert per layer with one token each
        active_per_layer = [torch.tensor([i]) for i in range(n_layers)]
        token_per_layer = [[torch.tensor([0])] for _ in range(n_layers)]
        raw_per_layer = [[torch.randn(1, d_model)] for _ in range(n_layers)]
        kpos_per_layer = [[torch.tensor([0])] for _ in range(n_layers)]

        means, counts = _aggregate_document(
            expert_indices=expert_indices,
            expert_weights=expert_weights,
            active_experts_per_layer=active_per_layer,
            token_indices_per_layer=token_per_layer,
            raw_outputs_per_layer=raw_per_layer,
            top_k_pos_per_layer=kpos_per_layer,
            n_experts=n_experts,
            d_model=d_model,
        )

        assert means.shape == (n_layers, n_experts, d_model)
        assert counts.shape == (n_layers, n_experts)


class TestDictionary:
    """Test concept dictionary construction."""

    def test_dictionary_is_normalized(self):
        """Dictionary rows should be L2-normalized."""
        vocab_size, d_model = 100, 32
        unembedding = torch.randn(vocab_size, d_model)

        # Minimal mock tokenizer
        class MockTokenizer:
            def encode(self, text, add_special_tokens=False):
                # Return deterministic token IDs based on text hash
                return [abs(hash(text)) % vocab_size]

        dictionary, token_ids = make_concept_dictionary(
            unembedding, MockTokenizer(), ["red", "blue", "green"]
        )

        norms = dictionary.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_dictionary_deduplicates_tokens(self):
        """Same token ID from "word" and " word" should appear once."""
        vocab_size, d_model = 100, 16
        unembedding = torch.randn(vocab_size, d_model)

        class MockTokenizer:
            def encode(self, text, add_special_tokens=False):
                # Both "cat" and " cat" map to same ID
                return [42]

        dictionary, token_ids = make_concept_dictionary(
            unembedding, MockTokenizer(), ["cat"]
        )

        assert len(token_ids) == 1
        assert token_ids[0] == 42
        assert dictionary.shape == (1, d_model)

    def test_concept_word_lists_exist(self):
        """All expected concept lists are present and non-empty."""
        for name in ["countries", "colors", "numbers"]:
            assert name in CONCEPTS
            assert len(CONCEPTS[name]) > 0


class TestPrintTopExperts:
    """Test expert ranking utilities."""

    def test_top_experts_ordering(self):
        """Top experts should be returned in descending EVR order."""
        n_layers, n_experts, k = 4, 8, 10
        evr = torch.rand(n_layers, n_experts, k)

        # Plant a known max at layer=2, expert=5
        evr[2, 5, -1] = 100.0

        results = print_top_experts(evr, n=5)

        assert len(results) == 5
        assert results[0] == (2, 5, 100.0)
        # Rest should be in descending order
        for i in range(1, len(results)):
            assert results[i][2] <= results[i - 1][2]

    def test_top_experts_respects_n(self):
        """Should return exactly n results."""
        evr = torch.rand(2, 4, 5)
        results = print_top_experts(evr, n=3)
        assert len(results) == 3
