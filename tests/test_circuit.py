"""Unit tests for the model-free pieces of the circuit comparison (no model needed)."""

from __future__ import annotations

import torch

from moe_interp.circuit import compare, intervene


def test_intervention_bar_builds():
    methods = {
        "baseline": {"eliciting_propensity": 2.0, "neutral_propensity": 1.0},
        "knockout": {"eliciting_propensity": 1.5, "neutral_propensity": 1.0},
    }
    fig = compare.intervention_bar(methods, title="t")
    assert len(fig.data) == 3  # eliciting, Δ, neutral bars


def test_concept_regex_matches_whole_words():
    from moe_interp.pursuit.concepts import CONCEPT_WORDS

    pat = intervene.concept_regex(CONCEPT_WORDS["offensive"])
    word = CONCEPT_WORDS["offensive"][0]
    assert pat.findall(f"there was {word} reported")  # whole concept word matches
    assert not pat.findall("xqzv nonsense filler text")  # nothing matches => empty
