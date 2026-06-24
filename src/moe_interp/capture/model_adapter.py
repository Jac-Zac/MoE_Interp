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
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MoEConfig:
    """MoE model config fields needed by capture, extracted from ``model.config``."""

    model_name: str
    model_type: str
    n_layers: int
    n_experts: int
    d_model: int
    experts_per_tok: int

    @classmethod
    def from_model(cls, model: Any) -> "MoEConfig":
        cfg = model.config
        # OLMoE/Mixtral/gpt-oss expose `num_local_experts`; some configs use `num_experts`.
        n_experts = getattr(cfg, "num_local_experts", None)
        if n_experts is None:
            n_experts = cfg.num_experts
        return cls(
            model_name=cfg._name_or_path,
            model_type=cfg.model_type,
            n_layers=cfg.num_hidden_layers,
            n_experts=n_experts,
            d_model=cfg.hidden_size,
            experts_per_tok=cfg.num_experts_per_tok,
        )

    def _rich_table(self):
        from rich.table import Table

        t = Table(title="MoE Config")
        t.add_column("Property", style="cyan")
        t.add_column("Value", style="magenta")
        t.add_row("Model", self.model_name)
        t.add_row("Model Type", self.model_type)
        t.add_row("Layers", str(self.n_layers))
        t.add_row("Hidden Size", str(self.d_model))
        t.add_row("Experts", str(self.n_experts))
        t.add_row("Experts per Token", str(self.experts_per_tok))
        return t


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
    """Abstract adapter: model config + model-specific expert reconstruction."""

    def __init__(self, model: Any) -> None:
        self.config = MoEConfig.from_model(model)

    # --- config passthroughs -------------------------------------------------
    @property
    def model_name(self) -> str:
        return self.config.model_name

    @property
    def model_type(self) -> str:
        return self.config.model_type

    @property
    def n_layers(self) -> int:
        return self.config.n_layers

    @property
    def n_experts(self) -> int:
        return self.config.n_experts

    @property
    def d_model(self) -> int:
        return self.config.d_model

    @property
    def experts_per_tok(self) -> int:
        return self.config.experts_per_tok

    def __repr__(self) -> str:
        c = self.config
        return (
            f"{self.__class__.__name__}(model_name={c.model_name!r}, "
            f"model_type={c.model_type!r}, n_layers={c.n_layers}, "
            f"n_experts={c.n_experts}, d_model={c.d_model}, "
            f"experts_per_tok={c.experts_per_tok})"
        )

    def __rich_console__(self, console: Any, options: Any):
        yield self.config._rich_table()

    # --- boundary tap (shared; override only if a model differs) -------------
    def tap_layer(self, layer: Any) -> Any:
        """Proxy to ``.save()`` inside the trace for one layer's MoE block.

        Fused experts modules (gpt-oss / OLMoE / Mixtral) are all called as
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
        max_len: int,
        norm_weight: torch.Tensor,
        norm_eps: float,
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Recompute each expert's per-token contribution from one MoE block's inputs.

        Loops only over experts that fired; for each, batches all its (token, slot)
        pairs through ``expert_forward``, scales by the routing weight, then applies
        component RMSNorm. Returns ``{expert_id: (activations, token_ids,
        routing_weights, positions)}`` over real tokens only, CPU/float16 ready to
        write. Args mirror the flattened ``(b_size * max_len)`` token axis;
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
                (t_idx % max_len).cpu(),
            )
        return out


class SwiGLUMoEAdapter(MoEAdapter):
    """Adapter for fused SwiGLU experts (OLMoE, Mixtral).

    Mirrors ``OlmoeExperts``/``MixtralExperts``: ``gate_up_proj`` is
    ``(E, 2I, D)`` and gate/up are contiguous halves; no biases, no clamp::

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
    "mixtral": SwiGLUMoEAdapter,
}

KNOWN_MODEL_NAME_TO_TYPE: dict[str, str] = {
    "openai/gpt-oss-20b": "gpt_oss",
    "allenai/OLMoE-1B-7B-0924-Instruct": "olmoe",
    "mistralai/Mixtral-8x7B-Instruct-v0.1": "mixtral",
}


def get_model_adapter(
    model: Any | None = None,
    model_name: str | None = None,
    model_type: str | None = None,
) -> MoEAdapter:
    """Return the correct adapter, resolving the model type in order:

    1) ``model.config.model_type``  2) explicit ``model_type``
    3) exact lookup from known ``model_name``.
    """
    resolved_type: str | None = None
    if model is not None and getattr(
        getattr(model, "config", None), "model_type", None
    ):
        resolved_type = model.config.model_type
    elif model_type is not None:
        resolved_type = model_type
    elif model_name is not None:
        resolved_type = KNOWN_MODEL_NAME_TO_TYPE.get(model_name)

    if resolved_type in MODEL_TYPE_TO_ADAPTER:
        return MODEL_TYPE_TO_ADAPTER[resolved_type](model)

    raise ValueError(
        "Could not resolve model adapter. Supported model_type values: "
        f"{sorted(MODEL_TYPE_TO_ADAPTER)}. For model_name fallback, supported "
        f"names: {sorted(KNOWN_MODEL_NAME_TO_TYPE)}."
    )
