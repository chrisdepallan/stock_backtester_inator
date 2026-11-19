"""Margin, costs, wipe-outs and margin calls for already-generated trades.

Margin = position size / leverage, i.e. position size (the full leveraged
market exposure) = margin * leverage. Costs are charged on the full
position value; P&L is expressed as a return on margin:

    margin_pnl = leverage * gross_ret - costs / margin
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mbt.config import CostsConfig
from mbt.costs import intraday_costs, overnight_costs

TRADE_OUTPUT_COLUMNS = [
    "ticker",
    "signal_date",
    "entry_date",
    "entry_price",
    "exit_date",
    "exit_price",
    "gross_ret",
    "costs",
    "net_ret",
    "leverage",
    "margin_pnl",
    "wiped_out",
    "margin_call",
]


def apply_leverage_and_costs(
    trades: pd.DataFrame,
    leverage: float,
    capital_per_trade: float,
    costs_cfg: CostsConfig,
    margin_call_threshold: float,
) -> pd.DataFrame:
    """Add cost, margin P&L, wipe-out and margin-call columns to raw `trades`.

    `capital_per_trade` is the margin committed to each trade (equal capital
    per the position-sizing rule); position value = capital_per_trade * leverage.
    A wipe-out is a margin_pnl <= -100%; a margin call is margin_pnl below
    -margin_call_threshold. Losses are never capped, since overnight gaps can
    push a trade's loss past 100% of its margin.
    """
    if trades.empty:
        return pd.DataFrame(columns=TRADE_OUTPUT_COLUMNS)

    margin = capital_per_trade
    position_value = margin * leverage

    # capital_per_trade and leverage are fixed for the whole call, so
    # position_value/margin are scalars: every intraday trade shares one
    # cost total, and every overnight trade shares one base cost total plus
    # an MTF-interest term that depends only on that trade's nights held.
    # This lets the whole trades table be costed without a per-row loop.
    out = trades.copy()
    entry_date = pd.to_datetime(out["entry_date"])
    exit_date = pd.to_datetime(out["exit_date"])
    held_overnight = (exit_date > entry_date).to_numpy()
    nights_held = (exit_date - entry_date).dt.days.to_numpy()

    intraday_total = intraday_costs(position_value, costs_cfg.intraday).total
    overnight_base = overnight_costs(position_value, margin, nights_held=0, cfg=costs_cfg.overnight)
    borrowed = max(position_value - margin, 0.0)
    daily_rate = costs_cfg.overnight.mtf_annual_interest_pct / 365
    overnight_total = overnight_base.total + borrowed * daily_rate * nights_held

    out["costs"] = np.where(held_overnight, overnight_total, intraday_total)
    out["net_ret"] = out["gross_ret"] - out["costs"] / position_value
    out["leverage"] = leverage
    out["margin_pnl"] = leverage * out["gross_ret"] - out["costs"] / margin
    out["wiped_out"] = out["margin_pnl"] <= -1.0
    out["margin_call"] = out["margin_pnl"] <= -margin_call_threshold

    return out[TRADE_OUTPUT_COLUMNS]
