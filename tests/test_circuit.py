"""Unit tests for the model-free pieces of the circuit code (no model needed)."""

from __future__ import annotations

import torch

from moe_interp.circuit import intervene


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
