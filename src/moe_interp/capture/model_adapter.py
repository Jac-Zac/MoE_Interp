"""Model adapters for MoE expert-contribution reconstruction.

Capture taps each MoE block's boundary tensors once per forward
(``hidden_states``, ``top_k_index``, ``top_k_weights``) and re-derives every
expert's per-token contribution from the block's weight params — fused MoE
kernels never materialise the per-expert vectors, and nnsight 0.7's
``tracer.iter`` does not step the internal expert loop, so they cannot be traced
directly.

The boundary tap and the surrounding reconstruction loop (real-token masking,
routing-weight scaling, component RMSNorm) are shared. The only model-specific
piece is ``expert_forward`` — one expert's raw transform — which mirrors that
model's own expert math so the reconstruction is exact. Pick an adapter with
``get_model_adapter(model)``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn.functional as F


def apply_component_rmsnorm(
    hidden_states: torch.Tensor,
    second_moment: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Approximate component RMSNorm using the residual stream second moment.

    This keeps the expert output on the same scale as the final model norm while
    avoiding recomputing the full residual-stream normalization.
    """
    input_dtype = hidden_states.dtype
    # NOTE: Keep float32 here for stability (HF issue #33133).
    hidden_states = hidden_states.to(torch.float32)
    hidden_states = hidden_states * torch.rsqrt(second_moment.unsqueeze(-1) + eps)
    return weight * hidden_states.to(input_dtype)


class MoEAdapter(ABC):
    """MoE config + model-specific expert reconstruction.

    Reads the config fields capture needs straight off ``model.config``. Only
    ``expert_forward`` differs per model; everything else is shared.
    """

    def __init__(self, model: Any) -> None:
        cfg = model.config
        self.model_name: str = cfg._name_or_path
        self.model_type: str = cfg.model_type
        self.n_layers: int = cfg.num_hidden_layers
        self.d_model: int = cfg.hidden_size
        self.experts_per_tok: int = cfg.num_experts_per_tok
        # OLMoE/gpt-oss expose `num_local_experts`; some configs use `num_experts`.
        self.n_experts: int = getattr(cfg, "num_local_experts", None) or cfg.num_experts

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(model_name={self.model_name!r}, "
            f"model_type={self.model_type!r}, n_layers={self.n_layers}, "
            f"n_experts={self.n_experts}, d_model={self.d_model}, "
            f"experts_per_tok={self.experts_per_tok})"
        )

    # --- boundary tap (shared; override only if a model differs) -------------
    def tap_layer(self, layer: Any) -> Any:
        """Proxy to ``.save()`` inside the trace for one layer's MoE block.

        Fused experts modules (gpt-oss / OLMoE) are all called as
        ``experts(hidden_states, top_k_index, top_k_weights)``.
        """
        return layer.mlp.experts.inputs

    def unpack_boundary(self, saved: Any) -> tuple[Any, Any, Any]:
        """Return ``(hidden_states, top_k_index, top_k_weights)`` from a saved tap."""
        return saved[0]

    # --- model-specific expert math ------------------------------------------
    @abstractmethod
    def expert_forward(
        self, experts: Any, expert_id: int, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """One expert's raw output over a token subset.

        ``hidden_states`` is ``(n_tokens, d_model)`` float32. Returns
        ``(n_tokens, d_model)`` float32, BEFORE routing-weight scaling and the
        component RMSNorm (both applied by ``reconstruct_expert_contributions``).
        """

    # --- shared reconstruction loop ------------------------------------------
    def reconstruct_expert_contributions(
        self,
        experts: Any,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
        *,
        real_mask: torch.Tensor,
        second_moment: torch.Tensor,
        token_ids: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_eps: float,
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Recompute each expert's per-token contribution from one MoE block's inputs.

        Loops only over experts that fired; for each, batches all its (token, slot)
        pairs through ``expert_forward``, scales by the routing weight, then applies
        component RMSNorm. Returns ``{expert_id: (activations, token_ids,
        routing_weights)}`` over kept tokens only, CPU/float16 ready to write. Args
        mirror the flattened ``(b_size * max_len)`` token axis;
        ``real_mask``/``second_moment``/``token_ids`` are length N and are moved to
        the experts' device here.
        """
        dev = hidden_states.device
        hidden_states = hidden_states.float()
        top_k_weights = top_k_weights.float()
        real_mask = real_mask.to(dev)
        second_moment = second_moment.to(dev)
        token_ids = token_ids.to(dev)
        norm_weight = norm_weight.to(dev)

        out: dict[int, tuple] = {}
        for e in torch.unique(top_k_index).tolist():
            t_idx, k_idx = (top_k_index == e).nonzero(as_tuple=True)
            keep = real_mask[t_idx]
            t_idx, k_idx = t_idx[keep], k_idx[keep]
            if t_idx.numel() == 0:
                continue

            contrib = self.expert_forward(experts, e, hidden_states[t_idx])
            contrib = contrib * top_k_weights[t_idx, k_idx, None]
            contrib = apply_component_rmsnorm(
                contrib, second_moment[t_idx], norm_weight, norm_eps
            )
            out[e] = (
                contrib.half().cpu(),
                token_ids[t_idx].cpu(),
                top_k_weights[t_idx, k_idx].cpu(),
            )
        return out


class SwiGLUMoEAdapter(MoEAdapter):
    """Adapter for fused SwiGLU experts (OLMoE).

    Mirrors ``OlmoeExperts``: ``gate_up_proj`` is ``(E, 2I, D)`` and gate/up are
    contiguous halves; no biases, no clamp::

        gate, up = F.linear(h, gate_up_proj[e]).chunk(2)
        out = F.linear(act_fn(gate) * up, down_proj[e])
    """

    def expert_forward(
        self, experts: Any, expert_id: int, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        gate_up = experts.gate_up_proj[expert_id].detach().float()  # (2I, D)
        down = experts.down_proj[expert_id].detach().float()  # (D, I)
        gate, up = F.linear(hidden_states, gate_up).chunk(2, dim=-1)
        return F.linear(experts.act_fn(gate) * up, down)


class GptOssAdapter(MoEAdapter):
    """Adapter for gpt-oss fused experts.

    Mirrors ``GptOssExperts``: ``gate_up_proj`` is ``(E, D, 2I)`` (no transpose),
    gate/up are interleaved, with clamps, biases and the ``(up + 1) * gate *
    sigmoid(alpha * gate)`` gating. Reuses the module's own ``_apply_gate`` so the
    interleave/clamp/alpha details stay in lockstep with the model::

        gate_up = h @ gate_up_proj[e] + gate_up_proj_bias[e]
        out = experts._apply_gate(gate_up) @ down_proj[e] + down_proj_bias[e]
    """

    def expert_forward(
        self, experts: Any, expert_id: int, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        gate_up_w = experts.gate_up_proj[expert_id].detach().float()  # (D, 2I)
        gate_up_b = experts.gate_up_proj_bias[expert_id].detach().float()  # (2I,)
        down_w = experts.down_proj[expert_id].detach().float()  # (I, D)
        down_b = experts.down_proj_bias[expert_id].detach().float()  # (D,)
        gate_up = hidden_states @ gate_up_w + gate_up_b
        gated = experts._apply_gate(gate_up)
        return gated @ down_w + down_b


MODEL_TYPE_TO_ADAPTER: dict[str, type[MoEAdapter]] = {
    "gpt_oss": GptOssAdapter,
    "olmoe": SwiGLUMoEAdapter,
}


def get_model_adapter(model: Any) -> MoEAdapter:
    """Return the adapter matching ``model.config.model_type``."""
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    try:
        return MODEL_TYPE_TO_ADAPTER[model_type](model)
    except KeyError:
        raise ValueError(
            f"Unsupported model_type {model_type!r}. "
            f"Supported: {sorted(MODEL_TYPE_TO_ADAPTER)}."
        ) from None
