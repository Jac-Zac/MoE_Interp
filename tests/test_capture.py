"""Tests for capture tensor staging."""

import torch
import torch.nn as nn

from moe_interp.capture.capture import reconstruct_expert_contributions, token_real_mask


class _TinyExperts(nn.Module):
    """Minimal stand-in for OlmoeExperts: same weight layout and forward math, so we can
    check reconstruct_expert_contributions against a true forward without a real model.
    """

    def __init__(self, n_experts, d_model, d_inter, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.gate_up_proj = nn.Parameter(
            torch.randn(n_experts, 2 * d_inter, d_model, generator=g)
        )
        self.down_proj = nn.Parameter(
            torch.randn(n_experts, d_model, d_inter, generator=g)
        )
        self.act_fn = nn.SiLU()
        self.num_experts = n_experts

    def forward(self, hidden_states, top_k_index, top_k_weights):
        out = torch.zeros_like(hidden_states)
        for e in range(self.num_experts):
            t_idx, k_idx = (top_k_index == e).nonzero(as_tuple=True)
            if t_idx.numel() == 0:
                continue
            gate, up = (hidden_states[t_idx] @ self.gate_up_proj[e].T).chunk(2, dim=-1)
            ch = self.act_fn(gate) * up
            contrib = (ch @ self.down_proj[e].T) * top_k_weights[t_idx, k_idx, None]
            out.index_add_(0, t_idx, contrib)
        return out


def test_token_real_mask_drops_padding():
    # two rows, lengths 2 and 3, max_len 3 -> real positions 0,1 / 0,1,2
    mask = token_real_mask([2, 3], max_len=3)
    assert mask.tolist() == [True, True, False, True, True, True]


def test_reconstruct_matches_true_forward_and_is_complete():
    torch.manual_seed(0)
    n_experts, d_model, d_inter, K = 6, 8, 16, 2
    n_real = 5  # one padded row appended below -> N = 6
    experts = _TinyExperts(n_experts, d_model, d_inter)

    hidden = torch.randn(n_real + 1, d_model)
    top_k_index = torch.randint(0, n_experts, (n_real + 1, K))
    top_k_weights = torch.rand(n_real + 1, K)

    # ground-truth summed output over the real tokens
    truth = experts(hidden, top_k_index, top_k_weights)

    out = reconstruct_expert_contributions(
        experts,
        hidden,
        top_k_index,
        top_k_weights,
        real_mask=torch.tensor([True] * n_real + [False]),  # last row is padding
        second_moment=torch.ones(n_real + 1),  # eps below makes RMSNorm a no-op
        token_ids=torch.arange(n_real + 1),
        max_len=n_real + 1,
        norm_weight=torch.ones(d_model),
        norm_eps=0.0,
    )

    # completeness: exactly n_real*K rows total, padding row excluded
    total_rows = sum(rows[0].shape[0] for rows in out.values())
    assert total_rows == n_real * K
    assert all(
        (ids != n_real).all() for _, ids, _, _ in out.values()
    )  # no padding token

    # correctness: summing reconstructed contributions back per token == true forward
    recon = torch.zeros_like(truth)
    for acts, ids, _, _ in out.values():
        recon.index_add_(0, ids.long(), acts.float())
    # activations are stored as float16, so compare at fp16 precision
    assert torch.allclose(recon[:n_real], truth[:n_real], rtol=1e-2, atol=1e-2)


def test_reconstruct_last_token_selection():
    """real_mask can restrict to the last real token of each row (token_selection=last)."""
    torch.manual_seed(0)
    experts = _TinyExperts(4, 8, 16)
    # 2 rows × max_len 3 = 6 flat tokens; lengths 2 and 3 -> last real = pos 1 and pos 2
    hidden = torch.randn(6, 8)
    top_k_index = torch.randint(0, 4, (6, 2))
    top_k_weights = torch.rand(6, 2)
    last_mask = torch.tensor([False, True, False, False, False, True])

    out = reconstruct_expert_contributions(
        experts,
        hidden,
        top_k_index,
        top_k_weights,
        real_mask=last_mask,
        second_moment=torch.ones(6),
        token_ids=torch.arange(6),
        max_len=3,
        norm_weight=torch.ones(8),
        norm_eps=0.0,
    )
    # only the 2 last-token rows kept, across their K experts
    total = sum(r[0].shape[0] for r in out.values())
    assert total == 2 * top_k_index.shape[1]
    kept_tokens = torch.cat([r[1] for r in out.values()]).tolist()
    assert set(kept_tokens) <= {1, 5}  # flat indices of the two last tokens
    # positions are prompt-relative (t % max_len): pos 1 and pos 2
    kept_pos = torch.cat([r[3] for r in out.values()]).tolist()
    assert set(kept_pos) <= {1, 2}
