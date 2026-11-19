import numpy as np
import pandas as pd
import pytest

from mbt.backtest import apply_position_limits, generate_candidate_trades, generate_trades
from mbt.config import FiltersConfig


def _df(rows: list[dict]) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=len(rows), freq="D")
    return pd.DataFrame(rows, index=idx)[["open", "high", "low", "close", "volume"]]


def _bar(price: float, volume: float = 1000.0) -> dict:
    return {"open": price, "high": price, "low": price, "close": price, "volume": volume}


def test_variant_a_exits_at_day2_close():
    # day0=100, day1=102 (signal, +2%), day2 open=103 close=108
    df = _df([_bar(100), _bar(102), {"open": 103, "high": 110, "low": 103, "close": 108, "volume": 1000}])
    trades = generate_candidate_trades(df, "X", threshold=0.01, variant="A", filters=FiltersConfig())
    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["entry_price"] == 103
    assert t["exit_price"] == 108
    assert t["entry_date"] == t["exit_date"]
    assert t["gross_ret"] == pytest.approx(108 / 103 - 1)


def test_variant_b_exits_at_day3_open():
    df = _df(
        [
            _bar(100),
            _bar(102),
            {"open": 103, "high": 105, "low": 102, "close": 104, "volume": 1000},
            {"open": 106, "high": 107, "low": 105, "close": 105.5, "volume": 1000},
        ]
    )
    trades = generate_candidate_trades(df, "X", threshold=0.01, variant="B", filters=FiltersConfig())
    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["entry_price"] == 103
    assert t["exit_price"] == 106
    assert t["entry_date"] != t["exit_date"]


def test_variant_d_exits_day2_close_when_day2_up():
    df = _df(
        [
            _bar(100),
            _bar(102),
            {"open": 103, "high": 110, "low": 103, "close": 109, "volume": 1000},  # day2 up vs entry 103
            {"open": 95, "high": 96, "low": 90, "close": 92, "volume": 1000},
        ]
    )
    trades = generate_candidate_trades(df, "X", threshold=0.01, variant="D", filters=FiltersConfig())
    t = trades.iloc[0]
    assert t["exit_price"] == 109
    assert t["entry_date"] == t["exit_date"]


def test_variant_d_exits_day3_close_when_day2_not_up():
    df = _df(
        [
            _bar(100),
            _bar(102),
            {"open": 103, "high": 104, "low": 98, "close": 99, "volume": 1000},  # day2 down vs entry 103
            {"open": 100, "high": 105, "low": 99, "close": 103, "volume": 1000},
        ]
    )
    trades = generate_candidate_trades(df, "X", threshold=0.01, variant="D", filters=FiltersConfig())
    t = trades.iloc[0]
    assert t["exit_price"] == 103
    assert t["exit_date"] != t["entry_date"]


def test_halted_day2_is_skipped():
    df = _df([_bar(100), _bar(102), _bar(102)])  # day2 O=H=L=C -> halted
    trades = generate_candidate_trades(df, "X", threshold=0.01, variant="A", filters=FiltersConfig())
    assert trades.empty


def test_gap_up_filter_skips_entry():
    df = _df(
        [
            _bar(100),
            _bar(102),
            {"open": 105, "high": 110, "low": 105, "close": 108, "volume": 1000},  # gap of ~2.94%
        ]
    )
    filters = FiltersConfig(skip_gap_up_pct=0.02)
    trades = generate_candidate_trades(df, "X", threshold=0.01, variant="A", filters=filters)
    assert trades.empty


def test_no_day2_yet_is_skipped():
    df = _df([_bar(100), _bar(102)])  # signal on last available day, no day-2 bar
    trades = generate_candidate_trades(df, "X", threshold=0.01, variant="A", filters=FiltersConfig())
    assert trades.empty


def test_position_limit_prefers_largest_day1_gain():
    candidates = pd.DataFrame(
        {
            "ticker": ["A", "B", "C"],
            "signal_date": pd.to_datetime(["2020-01-01"] * 3),
            "entry_date": pd.to_datetime(["2020-01-02"] * 3),
            "entry_price": [10.0, 10.0, 10.0],
            "exit_date": pd.to_datetime(["2020-01-02"] * 3),
            "exit_price": [11.0, 11.0, 11.0],
            "gross_ret": [0.1, 0.1, 0.1],
            "ret_d1": [0.01, 0.05, 0.03],
        }
    )
    accepted = apply_position_limits(candidates, max_open_positions=2)
    assert set(accepted["ticker"]) == {"B", "C"}


def test_position_limit_frees_slot_after_exit():
    # X and Y both want the single slot on day 2; Y wins (higher ret_d1) and
    # X is rejected outright (it is not requeued for later). Y's trade exits
    # same-day, so the slot is free again for Z's day-4 entry.
    candidates = pd.DataFrame(
        {
            "ticker": ["X", "Y", "Z"],
            "signal_date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-03"]),
            "entry_date": pd.to_datetime(["2020-01-02", "2020-01-02", "2020-01-04"]),
            "entry_price": [10.0, 10.0, 10.0],
            "exit_date": pd.to_datetime(["2020-01-04", "2020-01-02", "2020-01-04"]),
            "exit_price": [11.0, 11.0, 11.0],
            "gross_ret": [0.1, 0.1, 0.1],
            "ret_d1": [0.02, 0.05, 0.01],
        }
    )
    accepted = apply_position_limits(candidates, max_open_positions=1)
    assert set(accepted["ticker"]) == {"Y", "Z"}


def test_generate_trades_end_to_end_variant_a():
    data = {
        "X": _df([_bar(100), _bar(102), {"open": 103, "high": 105, "low": 102, "close": 104, "volume": 1000}]),
    }
    trades = generate_trades(data, variant="A", threshold=0.01, filters=FiltersConfig(), max_open_positions=5)
    assert len(trades) == 1
    assert list(trades.columns) == [
        "ticker",
        "signal_date",
        "entry_date",
        "entry_price",
        "exit_date",
        "exit_price",
        "gross_ret",
    ]
