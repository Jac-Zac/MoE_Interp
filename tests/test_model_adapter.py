"""Tests for MoE adapters: config extraction + per-expert reconstruction.

The reconstruction tests instantiate the *real* transformers expert modules with
random weights and run their forward (which sums every expert's weighted output into
the block output), then assert that summing the adapter's per-expert reconstruction
back per token reproduces that same tensor. Driving the production
``reconstruct_expert_contributions`` here is the gold-standard check that the captured
activations really are each expert's true contribution.
"""

from __future__ import annotations

import io

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
from rich.console import Console

from moe_interp.capture.model_adapter import (
    GptOssAdapter,
    MoEConfig,
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


def test_from_model_reads_config_fields():
    assert MoEConfig.from_model(fake_model("olmoe")) == MoEConfig(
        model_name="dummy/olmoe",
        model_type="olmoe",
        n_layers=2,
        n_experts=E,
        d_model=D,
        experts_per_tok=K,
    )


@pytest.mark.parametrize(
    "model_type,cls",
    [
        ("gpt_oss", GptOssAdapter),
        ("olmoe", SwiGLUMoEAdapter),
        ("mixtral", SwiGLUMoEAdapter),
    ],
)
def test_get_model_adapter_picks_class(model_type, cls):
    assert isinstance(get_model_adapter(fake_model(model_type)), cls)


def test_get_model_adapter_rejects_unknown():
    with pytest.raises(ValueError, match="Could not resolve"):
        get_model_adapter(fake_model("llama"))


def test_rich_rendering_includes_table():
    buf = io.StringIO()
    Console(file=buf, force_terminal=True, color_system="standard").print(
        get_model_adapter(fake_model("olmoe"))
    )
    out = buf.getvalue()
    assert "MoE Config" in out
    assert "dummy/olmoe" in out


# --- reconstruction correctness vs. the real modules -------------------------


@pytest.mark.parametrize("model_type", ["gpt_oss", "olmoe", "mixtral"])
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
