"""Tests for MoE adapters: config extraction + per-expert reconstruction.

The reconstruction tests instantiate the *real* transformers expert modules with
random weights and run their forward (which sums every expert's weighted output into
the block output), then assert that summing the adapter's per-expert reconstruction
back per token reproduces that same tensor. Driving the production
``reconstruct_expert_contributions`` here is the gold-standard check that the captured
activations really are each expert's true contribution.
"""

from __future__ import annotations

import pytest
import torch
from _helpers import (
    D,
    E,
    K,
    N,
    build_experts,
    fake_model,
    no_op_norm_kwargs,
    random_routing,
)

from moe_interp.capture.model_adapter import (
    GptOssAdapter,
    SwiGLUMoEAdapter,
    get_model_adapter,
)


def _summed_reconstruction(adapter, experts, hidden, top_idx, weights) -> torch.Tensor:
    """Run the production reconstruction and sum every expert's contribution back into
    one (N, D) tensor. With the RMSNorm made an identity, this should equal the experts
    module's own (routing-weighted) forward output."""
    out = adapter.reconstruct_expert_contributions(
        experts, hidden, top_idx, weights, **no_op_norm_kwargs(N)
    )
    recon = torch.zeros(N, D)
    for acts, ids, _, _ in out.values():
        recon.index_add_(0, ids.long(), acts.float())
    return recon


# --- config / selection ------------------------------------------------------


def test_adapter_reads_config_fields():
    a = get_model_adapter(fake_model("olmoe"))
    assert (
        a.model_name,
        a.model_type,
        a.n_layers,
        a.n_experts,
        a.d_model,
        a.experts_per_tok,
    ) == ("dummy/olmoe", "olmoe", 2, E, D, K)


@pytest.mark.parametrize(
    "model_type,cls",
    [
        ("gpt_oss", GptOssAdapter),
        ("olmoe", SwiGLUMoEAdapter),
    ],
)
def test_get_model_adapter_picks_class(model_type, cls):
    assert isinstance(get_model_adapter(fake_model(model_type)), cls)


def test_get_model_adapter_rejects_unknown():
    with pytest.raises(ValueError, match="Unsupported model_type"):
        get_model_adapter(fake_model("llama"))


def test_repr_includes_model_info():
    assert "dummy/olmoe" in repr(get_model_adapter(fake_model("olmoe")))


# --- reconstruction correctness vs. the real modules -------------------------


@pytest.mark.parametrize("model_type", ["gpt_oss", "olmoe"])
def test_reconstruction_matches_module_forward(model_type):
    """Sum of captured per-expert contributions == the experts module's fused output."""
    experts = build_experts(model_type)
    hidden = torch.randn(N, D)
    top_idx, weights = random_routing()

    with torch.no_grad():
        true = experts(hidden, top_idx, weights)
    recon = _summed_reconstruction(
        get_model_adapter(fake_model(model_type)), experts, hidden, top_idx, weights
    )
    # Contributions are stored as float16, so compare at fp16 precision.
    assert torch.allclose(recon, true.float(), rtol=1e-2, atol=1e-3)
