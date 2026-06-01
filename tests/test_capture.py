"""Tests for capture-specific plumbing: row capping, token masking, and how
``reconstruct_expert_contributions`` selects/labels the rows it stages (real-token
masking, completeness, prompt-relative positions). Expert-math correctness is covered
in test_model_adapter.py."""

import h5py
import torch
from _helpers import build_experts, fake_model

from moe_interp.capture.cache import append_to_file
from moe_interp.capture.capture import token_real_mask
from moe_interp.capture.model_adapter import get_model_adapter

# A real OLMoE experts block + its adapter; values don't matter for the masking tests,
# only which (token, slot) rows get staged and how they're labelled.
_EXPERTS = build_experts("olmoe")
_ADAPTER = get_model_adapter(fake_model("olmoe"))


def _reconstruct(real_mask, *, n_tokens, max_len, top_k_index, top_k_weights, hidden):
    return _ADAPTER.reconstruct_expert_contributions(
        _EXPERTS,
        hidden,
        top_k_index,
        top_k_weights,
        real_mask=real_mask,
        second_moment=torch.ones(n_tokens),
        token_ids=torch.arange(n_tokens),
        max_len=max_len,
        norm_weight=torch.ones(_EXPERTS.config.hidden_size),
        norm_eps=0.0,
    )


def test_append_caps_rows_per_expert(tmp_path):
    """max_rows truncates the incoming batch and stops once the expert is full."""
    path = tmp_path / "layer.h5"
    with h5py.File(path, "w") as f:
        # first write of 6 rows, cap 10 -> all kept
        append_to_file(f, 0, torch.zeros(6, 4), torch.arange(6), max_rows=10)
        # second write of 8 rows -> only 4 fit (10 - 6)
        append_to_file(f, 0, torch.ones(8, 4), torch.arange(6, 14), max_rows=10)
        # third write -> already full, dropped
        append_to_file(f, 0, torch.ones(3, 4), torch.arange(14, 17), max_rows=10)
    with h5py.File(path, "r") as f:
        n = f["expert_000"]["activations"].shape[0]
    assert n == 10


def test_token_real_mask_drops_padding():
    # two rows, lengths 2 and 3, max_len 3 -> real positions 0,1 / 0,1,2
    mask = token_real_mask([2, 3], max_len=3)
    assert mask.tolist() == [True, True, False, True, True, True]


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
        max_len=n_tokens,
        top_k_index=top_k_index,
        top_k_weights=top_k_weights,
        hidden=hidden,
    )

    total_rows = sum(rows[0].shape[0] for rows in out.values())
    assert total_rows == n_real * K  # exactly the real pairs, no padding
    assert all((ids != n_real).all() for _, ids, _, _ in out.values())


def test_reconstruct_last_token_selection():
    """real_mask can restrict to the last real token of each row (token_selection=last)."""
    torch.manual_seed(0)
    K = 2
    # 2 rows × max_len 3 = 6 flat tokens; lengths 2 and 3 -> last real = pos 1 and pos 2
    hidden = torch.randn(6, _EXPERTS.config.hidden_size)
    top_k_index = torch.randint(0, _EXPERTS.num_experts, (6, K))
    top_k_weights = torch.rand(6, K)
    last_mask = torch.tensor([False, True, False, False, False, True])

    out = _reconstruct(
        last_mask,
        n_tokens=6,
        max_len=3,
        top_k_index=top_k_index,
        top_k_weights=top_k_weights,
        hidden=hidden,
    )

    total = sum(r[0].shape[0] for r in out.values())
    assert total == 2 * K  # only the 2 last-token rows, across their K experts
    kept_tokens = torch.cat([r[1] for r in out.values()]).tolist()
    assert set(kept_tokens) <= {1, 5}  # flat indices of the two last tokens
    # positions are prompt-relative (t % max_len): pos 1 and pos 2
    kept_pos = torch.cat([r[3] for r in out.values()]).tolist()
    assert set(kept_pos) <= {1, 2}
