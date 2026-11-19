"""Baselines run alongside every variant: a random-entry null distribution
and a buy-and-hold index curve.

The random-entry baseline reuses `backtest.resolve_trade_exit` so it applies
the exact same variant exit rule (and halt handling) as the real strategy —
the only difference is that entries are not gated by the day-1 signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mbt.backtest import TRADE_COLUMNS, resolve_trade_exit


def build_random_entry_pool(data: dict[str, pd.DataFrame], variant: str) -> pd.DataFrame:
    """Every (ticker, day) in the universe that is a valid entry for `variant`,
    with no signal condition at all — the sampling pool for the baseline.
    """
    rows = []
    for ticker, df in data.items():
        for pos in range(len(df)):
            resolved = resolve_trade_exit(df, pos, variant)
            if resolved is None:
                continue
            entry_date, entry_price, exit_date, exit_price = resolved
            rows.append(
                {
                    "ticker": ticker,
                    "signal_date": entry_date,
                    "entry_date": entry_date,
                    "entry_price": entry_price,
                    "exit_date": exit_date,
                    "exit_price": exit_price,
                    "gross_ret": exit_price / entry_price - 1,
                }
            )
    return pd.DataFrame(rows, columns=TRADE_COLUMNS)


def random_entry_runs(
    pool: pd.DataFrame,
    n_trades: int,
    n_runs: int,
    seed: int,
) -> list[pd.DataFrame]:
    """Sample `n_runs` independent sets of `n_trades` random entries (same
    holding-period rule as the strategy, via `pool`), each without
    replacement, from a single seeded generator so results are reproducible.
    """
    if pool.empty or n_trades <= 0:
        return [pd.DataFrame(columns=TRADE_COLUMNS) for _ in range(n_runs)]

    n_trades = min(n_trades, len(pool))
    rng = np.random.default_rng(seed)
    runs = []
    for _ in range(n_runs):
        chosen = rng.choice(len(pool), size=n_trades, replace=False)
        run_trades = pool.iloc[chosen].sort_values("entry_date").reset_index(drop=True)
        runs.append(run_trades)
    return runs


def index_buy_and_hold(
    benchmark_df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    starting_capital: float,
) -> pd.Series:
    """Equity curve of buying the benchmark's close at `start` and holding to `end`."""
    window = benchmark_df.loc[(benchmark_df.index >= start) & (benchmark_df.index <= end)]
    if window.empty:
        return pd.Series(dtype=float)
    return window["close"] / window["close"].iloc[0] * starting_capital
