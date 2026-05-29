"""Tests for greedy projection pursuit."""

import torch
import torch.nn.functional as F

from moe_interp.pursuit.decomposition import SOMP, somp
from moe_interp.pursuit.pursuit import projection_pursuit


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


def test_projection_pursuit_decodes_restricted_token_ids():
    X = torch.eye(2)
    dictionary = torch.eye(2)
    tokenizer = _DummyTokenizer()

    tokens, evr = projection_pursuit(
        X,
        dictionary,
        tokenizer,
        device="cpu",
        k=1,
        token_ids=[10, 42],
    )

    assert tokens == ["tok_10"]
    assert len(evr) == 1
    assert 0.0 <= evr[0] <= 1.0


def test_projection_pursuit_decodes_dataset_labels():
    X = torch.eye(2)
    dictionary = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    tokenizer = _DummyTokenizer()

    tokens, evr = projection_pursuit(
        X,
        dictionary,
        tokenizer,
        device="cpu",
        k=1,
        labels=["alpha", "beta"],
        base_vocab_size=2,
    )

    assert tokens == ["alpha"]
    assert len(evr) == 1


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
    assert all(evr[i] <= evr[i + 1] + 1e-6 for i in range(len(evr) - 1)), (
        f"EVR not monotone: {evr}"
    )

    # chosen atoms must be unique
    chosen = result["chosen"].tolist()
    assert len(chosen) == len(set(chosen)), "SOMP selected duplicate atoms"


def test_projection_pursuit_decodes_kept_token_ids():
    """Verify that kept_token_ids are used correctly for decoding base dictionary tokens."""
    # Create a dictionary where row index != token ID (simulates filtering)
    # Row 0 -> token ID 5, Row 1 -> token ID 10, Row 2 -> token ID 15
    X = torch.eye(2)
    dictionary = torch.tensor(
        [
            [1.0, 0.0],  # row 0 (token ID 5)
            [0.0, 1.0],  # row 1 (token ID 10)
            [0.5, 0.5],  # row 2 (token ID 15) - word atom
        ]
    )
    tokenizer = _DummyTokenizer()

    tokens, evr = projection_pursuit(
        X, dictionary, tokenizer, device="cpu", k=2, token_ids=[5, 10, 15]
    )

    # Should decode using kept_token_ids, not row indices
    assert "tok_5" in tokens
    assert "tok_10" in tokens
    assert len(evr) == 2


def test_projection_pursuit_decodes_concept_labels():
    """Concept-mode WordDictionary (base_vocab_size=0) decodes purely via labels."""
    X = torch.eye(3)
    dictionary = torch.eye(3)
    tokenizer = _DummyTokenizer()

    tokens, evr = projection_pursuit(
        X,
        dictionary,
        tokenizer,
        device="cpu",
        k=2,
        labels=["violence", "hate", "crime"],
        base_vocab_size=0,
    )

    assert tokens == ["violence", "hate"]
    assert len(evr) == 2


def test_run_pursuit_handles_concepts_with_only_single_token_words(
    monkeypatch, tmp_path
):
    from moe_interp.pursuit import pursuit

    class _DummyTokenizerForConcepts:
        def __call__(self, text, add_special_tokens=False):
            token_map = {"one": [0], "two": [1]}
            return type("Out", (), {"input_ids": token_map[text]})()

        def decode(self, token_ids):
            return f"tok_{token_ids[0]}"

    extractions_dir = tmp_path / "extractions"
    extractions_dir.mkdir()
    (extractions_dir / "metadata.json").write_text("{}")

    monkeypatch.setattr(pursuit, "CONCEPT_WORDS", {"tiny": ["one", "two"]})
    monkeypatch.setattr(
        pursuit,
        "load_metadata",
        lambda path: {"model_name": "dummy", "n_layers": 1, "n_experts": 1},
    )
    monkeypatch.setattr(pursuit, "load_unembedding", lambda path: torch.eye(4))
    monkeypatch.setattr(pursuit, "get_unembedding_dir", lambda model_name: tmp_path)
    monkeypatch.setattr(pursuit, "get_device", lambda: "cpu")
    monkeypatch.setattr(pursuit, "load_layer_activations", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda model_name: _DummyTokenizerForConcepts(),
    )

    results, evr_matrix, count_matrix = pursuit.run_pursuit(
        extractions_dir=extractions_dir,
        concept="tiny",
        output_dir=tmp_path / "out",
    )

    assert results == []
    assert evr_matrix.shape == (1, 1)
    assert count_matrix.shape == (1, 1)


def test_somp_uses_double_precision_for_lstsq(monkeypatch):
    from moe_interp.pursuit import decomposition as sd

    captured: dict[str, torch.dtype] = {}
    real_lstsq = torch.linalg.lstsq

    def fake_lstsq(a, b):
        captured["a"] = a.dtype
        captured["b"] = b.dtype
        return real_lstsq(a, b)

    monkeypatch.setattr(torch.linalg, "lstsq", fake_lstsq)

    torch.manual_seed(0)
    X = torch.randn(8, 4)
    dictionary = F.normalize(torch.randn(6, 4), dim=1)

    sd.somp(
        X=X,
        orig_X=X,
        pc=None,
        dictionary=dictionary,
        descriptors=list(range(6)),
        k=2,
        device="cpu",
        compute_evr=True,
    )

    assert captured["a"] == torch.float64
    assert captured["b"] == torch.float64


def test_somp_supports_pca_reduced_path():
    torch.manual_seed(1)
    X = torch.randn(20, 8)
    dictionary = F.normalize(torch.randn(12, 8), dim=1)

    result = SOMP(k=3, pc=4, compute_evr=True, return_full=False)(
        X=X,
        dictionary=dictionary,
        descriptors=list(range(12)),
        device="cpu",
    )

    assert result["chosen"].shape == (3,)
    assert result["evr"].shape == (3,)
    assert len(set(result["chosen"].tolist())) == 3
    assert all(0.0 <= val <= 1.0 + 1e-6 for val in result["evr"].tolist())
