"""Tests for ExpertActivationStore, aggregation, and content boundaries."""

from pathlib import Path

import torch

from src.cache import ExpertActivationStore
from src.capture import _aggregate_document
from src.data import TokenizedQuestion, _find_content_boundaries


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

    def test_sum_experts_matches_moe_output(self):
        """Summed gated expert outputs equal the MoE output per token."""
        torch.manual_seed(0)
        n_layers, n_experts, d_model = 2, 6, 8
        seq_len, k = 5, 3

        expert_indices = torch.randint(0, n_experts, (n_layers, seq_len, k))
        expert_weights = torch.rand(n_layers, seq_len, k)
        raw_outputs = torch.randn(n_layers, seq_len, k, d_model)

        active_experts_per_layer: list[torch.Tensor] = []
        token_indices_per_layer: list[list[torch.Tensor]] = []
        raw_outputs_per_layer: list[list[torch.Tensor]] = []
        top_k_pos_per_layer: list[list[torch.Tensor]] = []

        for layer_idx in range(n_layers):
            active_experts = torch.unique(expert_indices[layer_idx])
            token_indices_list: list[torch.Tensor] = []
            down_projs_list: list[torch.Tensor] = []
            top_k_pos_list: list[torch.Tensor] = []

            for expert_id in active_experts:
                positions = (expert_indices[layer_idx] == expert_id).nonzero(
                    as_tuple=False
                )
                token_idxs = positions[:, 0]
                k_positions = positions[:, 1]
                down_proj = raw_outputs[layer_idx, token_idxs, k_positions]

                token_indices_list.append(token_idxs)
                down_projs_list.append(down_proj)
                top_k_pos_list.append(k_positions)

            active_experts_per_layer.append(active_experts)
            token_indices_per_layer.append(token_indices_list)
            raw_outputs_per_layer.append(down_projs_list)
            top_k_pos_per_layer.append(top_k_pos_list)

        moe_output = (expert_weights.unsqueeze(-1) * raw_outputs).sum(dim=2)
        summed_experts = torch.zeros_like(moe_output)

        for layer_idx in range(n_layers):
            for i, _ in enumerate(active_experts_per_layer[layer_idx]):
                token_idxs = token_indices_per_layer[layer_idx][i]
                raw_output = raw_outputs_per_layer[layer_idx][i]
                k_positions = top_k_pos_per_layer[layer_idx][i]
                gate_weights = expert_weights[layer_idx, token_idxs, k_positions]
                gated = gate_weights.unsqueeze(-1) * raw_output
                summed_experts[layer_idx].index_add_(0, token_idxs, gated)

        assert torch.allclose(summed_experts, moe_output, atol=1e-6)

    def test_content_token_filtering(self):
        """Only tokens within [content_start, content_end) are averaged."""
        torch.manual_seed(1)
        n_layers, n_experts, d_model = 1, 4, 8
        seq_len, k = 10, 2

        expert_indices = torch.randint(0, n_experts, (n_layers, seq_len, k))
        expert_weights = torch.rand(n_layers, seq_len, k)

        # Build per-expert data
        active_experts_per_layer: list[torch.Tensor] = []
        token_indices_per_layer: list[list[torch.Tensor]] = []
        raw_outputs_per_layer: list[list[torch.Tensor]] = []
        top_k_pos_per_layer: list[list[torch.Tensor]] = []

        raw_outputs = torch.randn(n_layers, seq_len, k, d_model)
        for layer_idx in range(n_layers):
            active_experts = torch.unique(expert_indices[layer_idx])
            tl, dl, kl = [], [], []
            for expert_id in active_experts:
                positions = (expert_indices[layer_idx] == expert_id).nonzero(
                    as_tuple=False
                )
                tl.append(positions[:, 0])
                dl.append(raw_outputs[layer_idx, positions[:, 0], positions[:, 1]])
                kl.append(positions[:, 1])
            active_experts_per_layer.append(active_experts)
            token_indices_per_layer.append(tl)
            raw_outputs_per_layer.append(dl)
            top_k_pos_per_layer.append(kl)

        # With content_start=3, content_end=7, only tokens 3-6 should be used
        means_filtered, counts_filtered = _aggregate_document(
            expert_indices,
            expert_weights,
            active_experts_per_layer,
            token_indices_per_layer,
            raw_outputs_per_layer,
            top_k_pos_per_layer,
            n_experts,
            d_model,
            content_start=3,
            content_end=7,
        )

        # With no filtering (all tokens), result should differ
        means_all, counts_all = _aggregate_document(
            expert_indices,
            expert_weights,
            active_experts_per_layer,
            token_indices_per_layer,
            raw_outputs_per_layer,
            top_k_pos_per_layer,
            n_experts,
            d_model,
        )

        # Counts should be <= when filtering
        assert (counts_filtered <= counts_all).all()
        # At least some experts should have different means
        assert not torch.allclose(means_filtered, means_all)


class TestContentBoundaries:
    """Test chat template content boundary detection."""

    def test_finds_content_tokens(self):
        """Content boundaries exclude special tokens."""

        class MockTokenizer:
            def encode(self, text, add_special_tokens=False):
                if text == "<|user|>":
                    return [100]
                if text == "<|assistant|>":
                    return [101]
                return [50]

        # Simulated OLMoE chat template:
        # [EOS=99, <|user|>=100, \n=10, q1, q2, q3, \n=10, <|assistant|>=101, \n=10]
        token_ids = [99, 100, 10, 1, 2, 3, 10, 101, 10]
        start, end = _find_content_boundaries(token_ids, MockTokenizer())
        # Should find content at tokens 3-6 (indices of 1, 2, 3)
        assert start == 3
        assert end == 6

    def test_fallback_when_no_special_tokens(self):
        """Returns full range when special tokens are missing."""

        class MockTokenizer:
            def encode(self, text, add_special_tokens=False):
                return [999]  # never matches anything in token_ids

        token_ids = [1, 2, 3, 4, 5]
        start, end = _find_content_boundaries(token_ids, MockTokenizer())
        assert start == 0
        assert end == len(token_ids)

    def test_tokenized_question_dataclass(self):
        """TokenizedQuestion stores boundaries correctly."""
        q = TokenizedQuestion(
            token_ids=[99, 100, 10, 1, 2, 3, 10, 101, 10],
            content_start=3,
            content_end=6,
            source_idx=42,
        )
        assert q.token_ids[q.content_start : q.content_end] == [1, 2, 3]
        assert q.source_idx == 42
