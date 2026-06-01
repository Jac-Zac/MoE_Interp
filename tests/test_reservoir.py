"""Tests for the bounded per-expert reservoir sampler."""

import numpy as np
import torch

from moe_interp.capture.reservoir import _ROW_FIELDS, ExpertReservoir, _Cell


def _block(rows: list[int], d: int = 4) -> dict[str, np.ndarray]:
    """A row-aligned block whose every field encodes the same global row ids, so a
    misaligned sample is detectable: activations[:,0], tokens, weights, positions all
    equal the row id."""
    ids = np.asarray(rows, dtype=np.int64)
    acts = np.tile(ids.reshape(-1, 1), (1, d)).astype(np.float16)
    return {
        "activations": acts,
        "tokens": ids.astype(np.int64),
        "routing_weights": ids.astype(np.float16),
        "positions": ids.astype(np.int64),
    }


def test_cell_keeps_all_when_under_capacity():
    cell = _Cell(capacity=100)
    cell.add(_block(list(range(30))), np.random.default_rng(0))
    v = cell.view()
    assert cell.filled == 30 and cell.seen == 30
    assert sorted(v["tokens"].tolist()) == list(range(30))


def test_cell_caps_at_capacity():
    cell = _Cell(capacity=50)
    rng = np.random.default_rng(0)
    for start in range(0, 1000, 10):  # stream 1000 rows in small blocks
        cell.add(_block(list(range(start, start + 10))), rng)
    assert cell.filled == 50 and cell.seen == 1000
    assert len(set(cell.view()["tokens"].tolist())) == 50  # no duplicate rows kept


def test_reservoir_rows_stay_aligned():
    """Every field must carry the same row id after sampling (no cross-field shuffle)."""
    cell = _Cell(capacity=40)
    rng = np.random.default_rng(1)
    for start in range(0, 500, 25):
        cell.add(_block(list(range(start, start + 25))), rng)
    v = cell.view()
    ids = v["tokens"]
    assert np.array_equal(v["activations"][:, 0].astype(np.int64), ids)
    assert np.array_equal(v["routing_weights"].astype(np.int64), ids)
    assert np.array_equal(v["positions"], ids)


def test_reservoir_is_approximately_uniform():
    """Each of N source rows should be kept with prob ~ capacity/N. Check the mean
    inclusion rate and that early vs late rows are sampled comparably."""
    N, cap, trials = 1000, 100, 300
    counts = np.zeros(N)
    for t in range(trials):
        cell = _Cell(capacity=cap)
        rng = np.random.default_rng(t)
        for start in range(0, N, 50):
            cell.add(_block(list(range(start, start + 50))), rng)
        counts[cell.view()["tokens"]] += 1
    rate = counts / trials
    assert abs(rate.mean() - cap / N) < 0.02
    # First and last deciles should be sampled at similar rates (no strong recency bias).
    assert abs(rate[:100].mean() - rate[-100:].mean()) < 0.05


def test_expert_reservoir_add_pending_and_stats():
    res = ExpertReservoir(capacity=20, seed=0)

    # mimic staged writes: (layer,expert) -> [(acts,tok,weight,pos), ...]
    def torch_block(rows):
        b = _block(rows)
        return (
            torch.from_numpy(b["activations"]),
            torch.from_numpy(b["tokens"]),
            torch.from_numpy(b["routing_weights"]),
            torch.from_numpy(b["positions"]),
        )

    for start in range(0, 200, 10):
        res.add_pending({(3, 7): [torch_block(list(range(start, start + 10)))]})
    stats = res.stats()
    assert stats[(3, 7)] == (20, 200)
    assert _ROW_FIELDS == ("activations", "tokens", "routing_weights", "positions")
