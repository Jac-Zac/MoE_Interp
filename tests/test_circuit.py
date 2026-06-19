"""Unit tests for the model-free pieces of the circuit comparison (no model needed)."""

from __future__ import annotations

import torch

from moe_interp.circuit import compare, intervene


def test_pearson_perfect_and_anti():
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert compare._pearson(a, 2 * a + 1) > 0.999
    assert compare._pearson(a, -a) < -0.999


def test_faithfulness_recovers_known_correlation():
    patching = torch.zeros(2, 4)
    patching[0] = torch.tensor([1.0, 2.0, 3.0, 4.0])  # only layer 0 scored
    grids = {"good": patching.clone(), "flat": torch.ones(2, 4)}
    scores = compare.faithfulness(grids, patching)
    assert scores["good"]["pooled_r"] > 0.999
    assert abs(scores["flat"]["pooled_r"]) < 1e-6


def test_offensive_regex_matches_whole_words():
    from moe_interp.pursuit.concepts import CONCEPT_WORDS

    pat = intervene._offensive_regex()
    word = CONCEPT_WORDS["offensive"][0]
    assert pat.findall(f"there was {word} reported")  # whole offensive word matches
    assert not pat.findall("xqzv nonsense filler text")  # nothing offensive => no match
