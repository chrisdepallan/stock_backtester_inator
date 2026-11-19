import numpy as np
import pandas as pd
import pytest

from mbt.metrics import (
    build_equity_curve,
    percentile_rank,
    portfolio_metrics,
    split_in_out_of_sample,
    trade_metrics,
)


def _trades(net_rets: list[float], margin_pnls: list[float] | None = None) -> pd.DataFrame:
    n = len(net_rets)
    margin_pnls = margin_pnls if margin_pnls is not None else net_rets
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "ticker": ["X"] * n,
            "signal_date": idx,
            "entry_date": idx,
            "exit_date": idx,
            "gross_ret": net_rets,
            "net_ret": net_rets,
            "margin_pnl": margin_pnls,
            "wiped_out": [p <= -1.0 for p in margin_pnls],
            "margin_call": [p <= -0.5 for p in margin_pnls],
        }
    )


def test_trade_metrics_basic():
    trades = _trades([0.02, -0.01, 0.03, -0.02])
    m = trade_metrics(trades)
    assert m.n_trades == 4
    assert m.win_rate == pytest.approx(0.5)
    assert m.avg_net_ret == pytest.approx(0.005)
    assert m.n_wipeouts == 0


def test_trade_metrics_counts_wipeouts_and_margin_calls():
    trades = _trades([0.1, -1.2, -0.6, 0.05], margin_pnls=[0.1, -1.2, -0.6, 0.05])
    m = trade_metrics(trades)
    assert m.n_wipeouts == 1
    assert m.n_margin_calls == 2


def test_trade_metrics_empty():
    m = trade_metrics(pd.DataFrame(columns=["net_ret", "gross_ret", "margin_pnl", "wiped_out", "margin_call"]))
    assert m.n_trades == 0
    assert m.n_wipeouts == 0


def test_equity_curve_compounds_realized_pnl_at_exit():
    trades = _trades([0.10, -0.05])
    equity = build_equity_curve(trades, capital_per_trade=20_000.0, starting_capital=100_000.0)
    assert equity.iloc[0] == pytest.approx(100_000.0)
    assert equity.iloc[-1] == pytest.approx(100_000.0 + 20_000 * 0.10 - 20_000 * 0.05)


def test_portfolio_metrics_final_equity_and_drawdown():
    idx = pd.to_datetime(["2024-01-01", "2024-07-01", "2025-01-01"])
    equity = pd.Series([100_000.0, 80_000.0, 120_000.0], index=idx)
    pm = portfolio_metrics(equity, risk_free_rate_annual=0.07)
    assert pm.final_equity == pytest.approx(120_000.0)
    assert pm.total_return == pytest.approx(0.2)
    assert pm.max_drawdown == pytest.approx(-0.2)


def test_portfolio_metrics_handles_negative_final_equity_without_crashing():
    # Leveraged losses can push equity to or past zero; a fractional power of
    # a negative base would otherwise raise/produce a complex number.
    idx = pd.to_datetime(["2024-01-01", "2024-07-01", "2025-01-01"])
    equity = pd.Series([100_000.0, 50_000.0, -20_000.0], index=idx)
    pm = portfolio_metrics(equity, risk_free_rate_annual=0.07)
    assert pm.final_equity == pytest.approx(-20_000.0)
    assert pm.cagr == pytest.approx(-1.0)


def test_percentile_rank():
    dist = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert percentile_rank(10, dist) == pytest.approx(100.0)
    assert percentile_rank(0, dist) == pytest.approx(0.0)
    assert percentile_rank(5, dist) == pytest.approx(50.0)


def test_split_in_out_of_sample():
    idx = pd.to_datetime(["2020-01-01", "2021-12-31", "2022-01-01", "2023-06-01"])
    trades = pd.DataFrame({"signal_date": idx, "net_ret": [0.01, 0.02, 0.03, 0.04]})
    in_sample, out_of_sample = split_in_out_of_sample(trades, "2021-12-31", "2022-01-01")
    assert len(in_sample) == 2
    assert len(out_of_sample) == 2
