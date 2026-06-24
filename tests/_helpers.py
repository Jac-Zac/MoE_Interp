"""Shared test helpers: build the *real* transformers Experts modules with small
random weights so reconstruction can be verified against the model's own fused
forward (which sums every expert's weighted output into the block output)."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import torch

# Distinct hidden vs intermediate so any transpose/shape bug fails loudly.
D, I, E, K, N = 8, 12, 4, 2, 6

_EXPERTS_MODULE = {
    "gpt_oss": "transformers.models.gpt_oss.modeling_gpt_oss:GptOssExperts",
    "olmoe": "transformers.models.olmoe.modeling_olmoe:OlmoeExperts",
}


def fake_model(model_type: str) -> SimpleNamespace:
    """A stand-in for a loaded model exposing only the config fields adapters read."""
    cfg = SimpleNamespace(
        _name_or_path=f"dummy/{model_type}",
        model_type=model_type,
        num_hidden_layers=2,
        num_local_experts=E,
        hidden_size=D,
        intermediate_size=I,
        num_experts_per_tok=K,
        hidden_act="silu",
        # transformers wraps Experts.forward in a dispatcher that reads this.
        _experts_implementation="eager",
    )
    return SimpleNamespace(config=cfg)


def build_experts(model_type: str) -> torch.nn.Module:
    """Instantiate the real transformers Experts module with small random weights."""
    module_path, name = _EXPERTS_MODULE[model_type].split(":")
    Experts = getattr(importlib.import_module(module_path), name)
    model = fake_model(model_type)
    experts = Experts(model.config).eval()
    experts.config = model.config  # the forward dispatcher reads self.config
    with torch.no_grad():
        # std 0.2 (not the tiny real-init 0.02) drives activations into SiLU's nonlinear
        # range, so the reconstruction check actually exercises the gating math instead
        # of a near-linear regime where e.g. swapping gate/up would slip through.
        for p in experts.parameters():
            p.normal_(0.0, 0.2)
    return experts


def random_routing() -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(top_k_index, top_k_weights)`` of shape (N, K) from random logits."""
    logits = torch.randn(N, E)
    top_vals, top_idx = torch.topk(logits, K, dim=-1)
    return top_idx, torch.softmax(top_vals, dim=-1)


def no_op_norm_kwargs(n_tokens: int) -> dict:
    """Kwargs for ``reconstruct_expert_contributions`` that make the component RMSNorm an
    identity (unit second moment, unit weight, zero eps) so the summed reconstruction can
    be compared directly to the experts module's own forward output."""
    return dict(
        real_mask=torch.ones(n_tokens, dtype=torch.bool),
        second_moment=torch.ones(n_tokens),
        token_ids=torch.arange(n_tokens),
        max_len=n_tokens,
        norm_weight=torch.ones(D),
        norm_eps=0.0,
    )
