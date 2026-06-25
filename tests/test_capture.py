"""Tests for capture-specific plumbing: append/storage and how
``reconstruct_expert_contributions`` selects/labels the rows it stages (real-token
masking, completeness, last-token selection). Expert-math correctness is covered in
test_model_adapter.py."""

import h5py
import torch
from _helpers import build_experts, fake_model

from moe_interp.capture.cache import append_to_file, load_layer_h5
from moe_interp.capture.model_adapter import get_model_adapter

# A real OLMoE experts block + its adapter; values don't matter for the masking tests,
# only which (token, slot) rows get staged and how they're labelled.
_EXPERTS = build_experts("olmoe")
_ADAPTER = get_model_adapter(fake_model("olmoe"))


def _reconstruct(real_mask, *, n_tokens, top_k_index, top_k_weights, hidden):
    return _ADAPTER.reconstruct_expert_contributions(
        _EXPERTS,
        hidden,
        top_k_index,
        top_k_weights,
        real_mask=real_mask,
        second_moment=torch.ones(n_tokens),
        token_ids=torch.arange(n_tokens),
        norm_weight=torch.ones(_EXPERTS.config.hidden_size),
        norm_eps=0.0,
    )


def test_append_stores_activations_and_routing_weights(tmp_path):
    """append_to_file grows the expert group and keeps optional routing_weights."""
    path = tmp_path / "layer_00.h5"
    with h5py.File(path, "w") as f:
        append_to_file(f, 0, torch.zeros(6, 4), torch.arange(6), torch.rand(6))
        append_to_file(f, 0, torch.ones(4, 4), torch.arange(6, 10), torch.rand(4))
    layer = load_layer_h5(tmp_path, 0, n_experts=4)
    assert layer[0]["activations"].shape == (10, 4)
    assert layer[0]["routing_weights"].shape == (10,)


def test_reconstruct_is_complete_over_real_tokens():
    """Every real (token, slot) pair is staged exactly once; padding is excluded."""
    torch.manual_seed(0)
    n_experts, K = _EXPERTS.num_experts, 2
    n_real = 5  # one padded row appended below -> N = 6
    n_tokens = n_real + 1

    hidden = torch.randn(n_tokens, _EXPERTS.config.hidden_size)
    top_k_index = torch.randint(0, n_experts, (n_tokens, K))
    top_k_weights = torch.rand(n_tokens, K)

    out = _reconstruct(
        torch.tensor([True] * n_real + [False]),  # last row is padding
        n_tokens=n_tokens,
        top_k_index=top_k_index,
        top_k_weights=top_k_weights,
        hidden=hidden,
    )

    total_rows = sum(rows[0].shape[0] for rows in out.values())
    assert total_rows == n_real * K  # exactly the real pairs, no padding
    assert all((ids != n_real).all() for _, ids, _ in out.values())


def test_reconstruct_last_token_selection():
    """real_mask restricts to the last real token of each row (the capture default)."""
    torch.manual_seed(0)
    K = 2
    # 2 rows × max_len 3 = 6 flat tokens; lengths 2 and 3 -> last real = pos 1 and pos 5
    hidden = torch.randn(6, _EXPERTS.config.hidden_size)
    top_k_index = torch.randint(0, _EXPERTS.num_experts, (6, K))
    top_k_weights = torch.rand(6, K)
    last_mask = torch.tensor([False, True, False, False, False, True])

    out = _reconstruct(
        last_mask,
        n_tokens=6,
        top_k_index=top_k_index,
        top_k_weights=top_k_weights,
        hidden=hidden,
    )

    total = sum(r[0].shape[0] for r in out.values())
    assert total == 2 * K  # only the 2 last-token rows, across their K experts
    kept_tokens = torch.cat([r[1] for r in out.values()]).tolist()
    assert set(kept_tokens) <= {1, 5}  # flat indices of the two last tokens
