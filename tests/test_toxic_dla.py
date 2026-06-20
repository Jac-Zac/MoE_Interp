"""Unit tests for the gradient-free toxic-DLA scoring (no model, no network)."""

from __future__ import annotations

import torch

from moe_interp.analysis import toxic_dla


class FakeTokenizer:
    """Maps single known words to one id; everything else to two ids (multi-token)."""

    VOCAB = {"idiot": 7, " idiot": 8, "hate": 3, " hate": 4}

    def __call__(self, text, add_special_tokens=False):
        class _Out:
            input_ids = (
                [FakeTokenizer.VOCAB[text]] if text in FakeTokenizer.VOCAB else [0, 1]
            )

        return _Out()


def test_build_toxic_token_ids_keeps_single_token_words():
    ids = toxic_dla.build_toxic_token_ids(FakeTokenizer(), words=["idiot", "hate"])
    assert ids == [3, 4, 7, 8]  # both spaced/unspaced variants, sorted


def test_toxic_direction_is_relative():
    # Vocab directions: toxic rows point along +x, the rest are mean-zero noise.
    dictionary = torch.zeros(10, 4)
    dictionary[:, 0] = torch.tensor(
        [5.0, 5.0, 0, 0, 0, 0, 0, 0, 0, 0]
    )  # rows 0,1 toxic
    tdir = toxic_dla.toxic_direction(dictionary, [0, 1])
    # mean(toxic)=5 on x; overall mean = 1.0 on x => relative direction +4 on x only.
    assert tdir[0] > 0 and torch.allclose(tdir[1:], torch.zeros(3, dtype=torch.float64))


def test_score_is_mean_projection():
    # An expert whose contributions point along the toxic direction scores positive;
    # one pointing away scores negative.
    tdir = torch.tensor([1.0, 0.0, 0.0]).double()
    toxic_expert = torch.tensor([[2.0, 1.0, 0.0], [3.0, -1.0, 0.0]])
    safe_expert = -toxic_expert
    assert float((toxic_expert.double() @ tdir).mean()) > 0
    assert float((safe_expert.double() @ tdir).mean()) < 0
