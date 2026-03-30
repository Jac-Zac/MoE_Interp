"""Model adapters for MoE trace access.

This module isolates model-specific nnsight trace node access so capture logic can
stay shared across MoE architectures.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MoEConfig:
    """Model configuration extracted once at adapter construction time."""

    model_name: str
    model_type: str
    n_layers: int
    n_experts: int
    d_model: int
    experts_per_tok: int


class MoEAdapter(ABC):
    """Abstract adapter for model-specific MoE trace access.

    Subclass properties expose model config (n_layers, n_experts, d_model, etc.)
    and __repr__ prints a rich config table.
    """

    def __init__(self, model: Any | None = None) -> None:
        self._config: MoEConfig | None = None
        if model is not None:
            self._config = self._extract_config(model)

    def _extract_config(self, model: Any) -> MoEConfig:
        cfg = model.config
        return MoEConfig(
            model_name=cfg._name_or_path,
            model_type=cfg.model_type,
            n_layers=cfg.num_hidden_layers,
            n_experts=self._get_n_experts(cfg),
            d_model=cfg.hidden_size,
            experts_per_tok=cfg.num_experts_per_tok,
        )

    def _get_n_experts(self, cfg: Any) -> int:
        return cfg.num_experts

    @property
    def model_name(self) -> str:
        return self._config.model_name

    @property
    def model_type(self) -> str:
        return self._config.model_type

    @property
    def n_layers(self) -> int:
        return self._config.n_layers

    @property
    def n_experts(self) -> int:
        return self._config.n_experts

    @property
    def d_model(self) -> int:
        return self._config.d_model

    @property
    def experts_per_tok(self) -> int:
        return self._config.experts_per_tok

    def __repr__(self) -> str:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        t = Table(title=f"Model Config [{self.__class__.__name__}]")
        t.add_column("Property", style="cyan")
        t.add_column("Value", style="magenta")
        t.add_row("Model", self.model_name)
        t.add_row("Model Type", self.model_type)
        t.add_row("Layers", str(self.n_layers))
        t.add_row("Hidden Size", str(self.d_model))
        t.add_row("Experts", str(self.n_experts))
        t.add_row("Experts per Token", str(self.experts_per_tok))
        console.print(t)
        return ""

    @abstractmethod
    def get_router_output(self, layer: Any) -> tuple[Any, Any, Any]:
        """Return router output tuple from the layer trace."""

    @abstractmethod
    def get_expert_hit(self, layer: Any) -> Any:
        """Return expert-hit tensor from the layer trace."""

    @abstractmethod
    def get_top_k_pos_token_idx(self, layer: Any) -> tuple[Any, Any]:
        """Return `(top_k_pos, token_idx)` for one active expert iteration."""

    @abstractmethod
    def get_expert_output(self, layer: Any) -> Any:
        """Return per-expert output tensor from the layer trace."""


class OLMoEAdapter(MoEAdapter):
    """Adapter for OLMoE-style MoE blocks."""

    def get_router_output(self, layer: Any) -> tuple[Any, Any, Any]:
        # top_k_weights: weight for each expert
        # top_k_indices: expert id active for each token
        # source.self_gate_0 outputs: (_, top_k_weights, top_k_indices)
        return layer.mlp.gate.output  # we can directly get gate output

    def get_expert_hit(self, layer: Any) -> Any:
        # NOTE: One must be very careful of what to get.
        # We need expert_hit after `nonzero_0`:
        # expert_mask_sum_0 -> expert_hit = torch.greater(...).nonzero()
        # torch_greater_0   -> + ...
        # nonzero_0         -> + ...
        return layer.mlp.experts.source.nonzero_0.output

    def get_top_k_pos_token_idx(self, layer: Any) -> tuple[Any, Any]:
        return layer.mlp.experts.source.torch_where_0.output

    def get_expert_output(self, layer: Any) -> Any:
        return layer.mlp.experts.source.nn_functional_linear_1.output


class GPTOSSAdapter(MoEAdapter):
    """Adapter for GPT-oss-style MoE blocks."""

    def _get_n_experts(self, cfg: Any) -> int:
        return cfg.num_local_experts

    def get_router_output(self, layer: Any) -> tuple[Any, Any, Any]:
        return layer.mlp.router.output

    def get_expert_hit(self, layer: Any) -> Any:
        return layer.mlp.experts.source.nonzero_0.output

    def get_top_k_pos_token_idx(self, layer: Any) -> tuple[Any, Any]:
        return layer.mlp.experts.source.torch_where_0.output

    def get_expert_output(self, layer: Any) -> Any:
        # For GPT-oss, this is the gated expert output before
        # `@ down_proj + down_proj_bias`.
        return layer.mlp.experts.source.self__apply_gate_0.output


class MixtralAdapter(MoEAdapter):
    """Adapter for Mixtral-style MoE blocks."""

    def _get_n_experts(self, cfg: Any) -> int:
        return cfg.num_local_experts

    def get_router_output(self, layer: Any) -> tuple[Any, Any, Any]:
        return layer.mlp.gate.output

    def get_expert_hit(self, layer: Any) -> Any:
        return layer.mlp.experts.source.nonzero_0.output

    def get_top_k_pos_token_idx(self, layer: Any) -> tuple[Any, Any]:
        return layer.mlp.experts.source.torch_where_0.output

    def get_expert_output(self, layer: Any) -> Any:
        return layer.mlp.experts.source.nn_functional_linear_1.output


MODEL_TYPE_TO_ADAPTER: dict[str, type[MoEAdapter]] = {
    "gpt_oss": GPTOSSAdapter,
    "mixtral": MixtralAdapter,
    "olmoe": OLMoEAdapter,
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
    """Return the correct adapter based on model metadata.

    Resolution order:
    1) `model.config.model_type`
    2) explicit `model_type`
    3) exact lookup from known `model_name`
    """
    resolved_type: str | None = None

    if (
        model is not None
        and hasattr(model, "config")
        and hasattr(model.config, "model_type")
    ):
        resolved_type = model.config.model_type
    elif model_type is not None:
        resolved_type = model_type
    elif model_name is not None:
        resolved_type = KNOWN_MODEL_NAME_TO_TYPE.get(model_name)

    if resolved_type in MODEL_TYPE_TO_ADAPTER:
        adapter_cls = MODEL_TYPE_TO_ADAPTER[resolved_type]
        return adapter_cls(model)

    raise ValueError(
        "Could not resolve model adapter. Supported model_type values: "
        f"{sorted(MODEL_TYPE_TO_ADAPTER.keys())}. "
        "For model_name fallback, supported names: "
        f"{sorted(KNOWN_MODEL_NAME_TO_TYPE.keys())}."
    )
