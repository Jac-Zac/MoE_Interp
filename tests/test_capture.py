"""Tests for capture tensor staging."""

import torch

from moe_interp.capture.capture import build_pending_writes


def _layer_data(token_selection: str) -> list[dict]:
    return [
        {
            "active_experts": torch.tensor([1]),
            "token_indices": [torch.tensor([0, 1, 2, 3, 4, 5])],
            "down_projs": [torch.arange(12, dtype=torch.float32).reshape(6, 2)],
            "top_k_pos": [torch.tensor([0, 1, 0, 1, 0, 1])],
            "weights": torch.tensor(
                [
                    [0.9, 0.1],
                    [0.8, 0.2],
                    [0.7, 0.3],
                    [0.6, 0.4],
                    [0.5, 0.5],
                    [0.4, 0.6],
                ],
            ),
            "token_selection": token_selection,
        }
    ]


def test_build_pending_writes_keeps_only_last_tokens():
    pending = build_pending_writes(
        layer_data_list=_layer_data("last"),
        padded_token_ids=torch.tensor([[10, 11, -1], [20, 21, 22]]),
        last_positions=torch.tensor([1, 5]),
        prompt_lengths=torch.tensor([2, 3]),
        second_moment=torch.ones(6),
        max_len=3,
        norm_weight=torch.ones(2),
        norm_eps=0.0,
    )

    activations, tokens, weights, positions = pending[(0, 1)][0]

    assert activations.shape == (2, 2)
    assert tokens.tolist() == [11, 22]
    assert torch.allclose(weights, torch.tensor([0.2, 0.6]))
    assert positions.tolist() == [1, 2]


def test_build_pending_writes_can_keep_all_real_tokens():
    pending = build_pending_writes(
        layer_data_list=_layer_data("all"),
        padded_token_ids=torch.tensor([[10, 11, -1], [20, 21, 22]]),
        last_positions=torch.tensor([1, 5]),
        prompt_lengths=torch.tensor([2, 3]),
        second_moment=torch.ones(6),
        max_len=3,
        norm_weight=torch.ones(2),
        norm_eps=0.0,
    )

    activations, tokens, weights, positions = pending[(0, 1)][0]

    assert activations.shape == (5, 2)
    assert tokens.tolist() == [10, 11, 20, 21, 22]
    assert torch.allclose(weights, torch.tensor([0.9, 0.2, 0.4, 0.5, 0.6]))
    assert positions.tolist() == [0, 1, 0, 1, 2]
