"""Unit tests for the model-free pieces of the circuit code (no model needed)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from moe_interp.circuit import intervene
from moe_interp.circuit.attribution import attribution_grid_path
from moe_interp.circuit.downweight import _bootstrap_cell
from moe_interp.circuit.expert_sets import _causal_grid_set, _matched_random_set
from moe_interp.grids import top_experts


def test_concept_regex_matches_whole_words():
    from moe_interp.pursuit.concepts import CONCEPT_WORDS

    pat = intervene.concept_regex(CONCEPT_WORDS["offensive"])
    word = CONCEPT_WORDS["offensive"][0]
    assert pat.findall(f"there was {word} reported")  # whole concept word matches
    assert not pat.findall("xqzv nonsense filler text")  # nothing matches => empty


class _FakeModel:
    """Stub exposing just ``model.model.layers[l].mlp.experts.inputs[0]`` as ``(h, idx, w)``."""

    class _Experts:
        def __init__(self, idx, w):
            self.inputs = [(None, idx, w)]

    class _Layer:
        def __init__(self, idx, w):
            self.mlp = type("M", (), {"experts": _FakeModel._Experts(idx, w)})()

    def __init__(self, idx, w):
        self.model = type("Inner", (), {"layers": {0: _FakeModel._Layer(idx, w)}})()


def test_gate_scale_scales_only_selected_expert():
    # two tokens routed to experts 3 and 5; downweight expert 3 by 0.5, leave 5 untouched.
    idx = torch.tensor([[3], [5]])
    w = torch.tensor([[2.0], [4.0]])
    intervene.gate_scale_intervention([(0, 3)], 0.5)(_FakeModel(idx, w))
    assert w[0, 0] == 1.0  # expert 3's gate halved
    assert w[1, 0] == 4.0  # expert 5 untouched


def test_gate_scale_zero_is_knockout():
    idx = torch.tensor([[3], [5]])
    w = torch.tensor([[2.0], [4.0]])
    intervene.gate_scale_intervention([(0, 3)], 0.0)(_FakeModel(idx, w))
    assert w[0, 0] == 0.0  # scale=0 fully zeros expert 3's gate (knockout)
    assert w[1, 0] == 4.0  # expert 5 untouched


def test_matched_random_set_fails_instead_of_looping_when_layer_is_full():
    with pytest.raises(ValueError, match="all 2 experts"):
        _matched_random_set([(0, 0), (0, 1)], n_experts=2)


def test_causal_grid_drops_unsampled_nan_cells(tmp_path):
    path = tmp_path / "grid.npy"
    np.save(path, np.array([[np.nan, -1.0], [2.0, np.nan]]))
    assert _causal_grid_set(path, 4) == [(1, 0), (0, 1)]


def test_top_experts_validates_mode():
    with pytest.raises(ValueError, match="by must be"):
        top_experts(np.ones((2, 2)), by="largest")


def test_bootstrap_rejects_unpaired_lengths():
    cell = {"propensity": [1.0], "word_hit": [0.0], "distinct1": [1.0]}
    base = {"propensity": [1.0, 2.0], "word_hit": [0.0], "distinct1": [1.0]}

    with pytest.raises(ValueError, match="equal propensity lengths"):
        _bootstrap_cell(cell, base, n_boot=10, rng=np.random.default_rng(0))


def test_attribution_grid_path_prefers_canonical_then_legacy(tmp_path):
    legacy = tmp_path / "attribution" / "atp_grid_n100.npy"
    legacy.parent.mkdir()
    legacy.touch()

    assert (
        attribution_grid_path(
            tmp_path,
            concept="offensive",
            n_prompts=100,
        )
        == legacy
    )

    canonical = legacy.with_name("atp_grid_offensive_n100.npy")
    canonical.touch()
    assert (
        attribution_grid_path(
            tmp_path,
            concept="offensive",
            n_prompts=100,
        )
        == canonical
    )
    assert (
        attribution_grid_path(
            tmp_path,
            concept="countries",
            n_prompts=60,
            hi=0.8,
            challenging=True,
        ).name
        == "atp_grid_countries_n60_hi0.8_chal.npy"
    )
