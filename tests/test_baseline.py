import numpy as np
import pandas as pd
import pytest

from mbt.baseline import build_random_entry_pool, index_buy_and_hold, random_entry_runs


def _bar(price: float, volume: float = 1000.0) -> dict:
    return {"open": price, "high": price * 1.01, "low": price * 0.99, "close": price, "volume": volume}


def _df(prices: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=len(prices), freq="D")
    return pd.DataFrame([_bar(p) for p in prices], index=idx)


def test_pool_excludes_halted_entry_day():
    df = _df([100, 101, 101, 103, 104])  # position 2 (101) is halted (flat O=H=L=C)
    halted_df = df.copy()
    halted_df.iloc[2] = {"open": 101, "high": 101, "low": 101, "close": 101, "volume": 1000}
    pool = build_random_entry_pool({"X": halted_df}, variant="A")
    assert (pool["entry_date"] == halted_df.index[2]).sum() == 0


def test_pool_variant_a_allows_every_row_as_an_entry_day():
    df = _df([100, 101, 102, 103, 104])
    pool = build_random_entry_pool({"X": df}, variant="A")
    assert len(pool) == len(df)  # intraday: no day-3 lookahead needed


def test_pool_variant_b_excludes_last_row_without_a_day3():
    df = _df([100, 101, 102, 103, 104])
    pool = build_random_entry_pool({"X": df}, variant="B")
    assert len(pool) == len(df) - 1


def test_random_entry_runs_reproducible_with_same_seed():
    df = _df(list(range(100, 150)))
    pool = build_random_entry_pool({"X": df}, variant="A")
    runs_a = random_entry_runs(pool, n_trades=5, n_runs=10, seed=42)
    runs_b = random_entry_runs(pool, n_trades=5, n_runs=10, seed=42)
    for a, b in zip(runs_a, runs_b):
        pd.testing.assert_frame_equal(a, b)


def test_random_entry_runs_different_seed_differs():
    df = _df(list(range(100, 150)))
    pool = build_random_entry_pool({"X": df}, variant="A")
    runs_a = random_entry_runs(pool, n_trades=5, n_runs=1, seed=1)
    runs_b = random_entry_runs(pool, n_trades=5, n_runs=1, seed=2)
    assert not runs_a[0]["entry_date"].equals(runs_b[0]["entry_date"])


def test_random_entry_runs_have_requested_trade_count():
    df = _df(list(range(100, 150)))
    pool = build_random_entry_pool({"X": df}, variant="A")
    runs = random_entry_runs(pool, n_trades=7, n_runs=3, seed=1)
    assert len(runs) == 3
    for run in runs:
        assert len(run) == 7


def test_index_buy_and_hold_normalizes_to_starting_capital():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    df = pd.DataFrame({"close": [100.0, 110.0, 99.0, 120.0, 130.0]}, index=idx)
    curve = index_buy_and_hold(df, idx[0], idx[-1], starting_capital=100_000.0)
    assert curve.iloc[0] == pytest.approx(100_000.0)
    assert curve.iloc[-1] == pytest.approx(130_000.0)
