"""Tests for MoE model adapters."""

from __future__ import annotations

import io

from rich.console import Console

from moe_interp.capture.model_adapter import OLMoEAdapter


class _DummyConfig:
    _name_or_path = "allenai/OLMoE-1B-7B-0924-Instruct"
    model_type = "olmoe"
    num_hidden_layers = 16
    num_experts = 64
    hidden_size = 2048
    num_experts_per_tok = 8


class _DummyModel:
    config = _DummyConfig()


def test_adapter_repr_is_plain_text():
    adapter = OLMoEAdapter(_DummyModel())

    assert repr(adapter) == (
        "OLMoEAdapter(model_name='allenai/OLMoE-1B-7B-0924-Instruct', "
        "model_type='olmoe', n_layers=16, n_experts=64, d_model=2048, "
        "experts_per_tok=8)"
    )


def test_adapter_rich_rendering_includes_table_and_styles():
    adapter = OLMoEAdapter(_DummyModel())
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, color_system="standard")

    console.print(adapter)

    output = buf.getvalue()
    assert "Model Config [OLMoEAdapter]" in output
    assert "allenai/OLMoE-1B-7B-0924-Instruct" in output
    assert "\x1b[" in output
