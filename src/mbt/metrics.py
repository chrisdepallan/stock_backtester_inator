"""Per-trade metrics and portfolio metrics (CAGR, drawdown, Sharpe, equity curve).

H1/H2 pattern statistics live in `signals.pattern_stats` since they are
computed directly from the day-1 signal, not from a trades table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TradeMetrics:
    n_trades: int
    win_rate: float
    avg_gross_ret: float
    avg_net_ret: float
    expectancy: float
    profit_factor: float
    worst_trade_margin_pnl: float
    n_wipeouts: int
    n_margin_calls: int


def trade_metrics(trades: pd.DataFrame) -> TradeMetrics:
    """Per-trade metrics for one variant x threshold x leverage combination."""
    if trades.empty:
        return TradeMetrics(
            n_trades=0,
            win_rate=float("nan"),
            avg_gross_ret=float("nan"),
            avg_net_ret=float("nan"),
            expectancy=float("nan"),
            profit_factor=float("nan"),
            worst_trade_margin_pnl=float("nan"),
            n_wipeouts=0,
            n_margin_calls=0,
        )

    wins = trades["net_ret"] > 0
    losses = trades["net_ret"] < 0
    win_rate = float(wins.mean())
    loss_rate = float(losses.mean())

    avg_win = float(trades.loc[wins, "net_ret"].mean()) if wins.any() else 0.0
    avg_loss = float(-trades.loc[losses, "net_ret"].mean()) if losses.any() else 0.0
    expectancy = avg_win * win_rate - avg_loss * loss_rate

    gross_wins = float(trades.loc[wins, "net_ret"].sum())
    gross_losses = float(-trades.loc[losses, "net_ret"].sum())
    profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else float("inf")

    return TradeMetrics(
        n_trades=len(trades),
        win_rate=win_rate,
        avg_gross_ret=float(trades["gross_ret"].mean()),
        avg_net_ret=float(trades["net_ret"].mean()),
        expectancy=float(expectancy),
        profit_factor=float(profit_factor),
        worst_trade_margin_pnl=float(trades["margin_pnl"].min()),
        n_wipeouts=int(trades["wiped_out"].sum()),
        n_margin_calls=int(trades["margin_call"].sum()),
    )


def build_equity_curve(trades: pd.DataFrame, capital_per_trade: float, starting_capital: float) -> pd.Series:
    """Equity over time: `starting_capital` plus cumulative realized trade P&L,
    booked on each trade's exit date (trades are sized with equal capital per
    the position-sizing rule, so P&L is not compounded into bigger stakes).
    """
    if trades.empty:
        return pd.Series([starting_capital], index=[pd.Timestamp("1970-01-01")])

    pnl_rupees = trades["margin_pnl"] * capital_per_trade
    realized = pnl_rupees.groupby(trades["exit_date"]).sum().sort_index()
    cum_equity = starting_capital + realized.cumsum()

    lead_in_date = cum_equity.index[0] - pd.Timedelta(days=1)
    return pd.concat([pd.Series([starting_capital], index=[lead_in_date]), cum_equity])


@dataclass
class PortfolioMetrics:
    final_equity: float
    total_return: float
    cagr: float
    max_drawdown: float
    sharpe: float


def portfolio_metrics(equity: pd.Series, risk_free_rate_annual: float) -> PortfolioMetrics:
    """CAGR, max drawdown, annualised Sharpe, and final equity from an equity curve.

    The curve only moves on trade-exit dates, so Sharpe annualises using the
    curve's own implied steps-per-year rather than assuming daily steps.
    """
    if len(equity) < 2:
        final = float(equity.iloc[-1]) if len(equity) else float("nan")
        return PortfolioMetrics(final, 0.0, float("nan"), 0.0, float("nan"))

    starting = float(equity.iloc[0])
    final = float(equity.iloc[-1])
    total_return = final / starting - 1

    span_days = (equity.index[-1] - equity.index[0]).days
    years = span_days / 365.25 if span_days > 0 else float("nan")
    if not years or years <= 0:
        cagr = float("nan")
    elif final <= 0:
        cagr = -1.0  # wiped out: leveraged losses can take equity to or past zero
    else:
        cagr = (final / starting) ** (1 / years) - 1

    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_drawdown = float(drawdown.min())

    step_rets = equity.pct_change().dropna()
    if len(step_rets) > 1 and years and years > 0:
        periods_per_year = len(step_rets) / years
        rf_per_period = risk_free_rate_annual / periods_per_year
        std = step_rets.std(ddof=1)
        sharpe = float((step_rets.mean() - rf_per_period) / std * np.sqrt(periods_per_year)) if std > 0 else float("nan")
    else:
        sharpe = float("nan")

    return PortfolioMetrics(final, float(total_return), float(cagr), max_drawdown, sharpe)


def percentile_rank(value: float, distribution: np.ndarray) -> float:
    """Percentage of `distribution` values at or below `value`."""
    if len(distribution) == 0 or np.isnan(value):
        return float("nan")
    return float((np.asarray(distribution) <= value).mean() * 100)


def split_in_out_of_sample(
    trades: pd.DataFrame, in_sample_end: str, out_of_sample_start: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split trades into (in-sample, out-of-sample) by signal date."""
    in_sample_end_ts = pd.Timestamp(in_sample_end)
    out_of_sample_start_ts = pd.Timestamp(out_of_sample_start)
    in_sample = trades.loc[trades["signal_date"] <= in_sample_end_ts]
    out_of_sample = trades.loc[trades["signal_date"] >= out_of_sample_start_ts]
    return in_sample, out_of_sample
