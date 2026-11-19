"""Indian equity transaction costs: intraday (variant A / same-day variant D)
vs MTF overnight (variant B, C, or variant D held past day 2).

All costs are charged on the full leveraged position value, not just the
margin put up — per the project spec's leverage model.
"""

from __future__ import annotations

from dataclasses import dataclass

from mbt.config import CostsConfig, IntradayCostConfig, OvernightCostConfig

CRORE = 10_000_000


@dataclass
class TradeCosts:
    brokerage: float
    stt: float
    exchange_txn: float
    sebi_fee: float
    stamp_duty: float
    gst: float
    slippage: float
    mtf_interest: float

    @property
    def total(self) -> float:
        return (
            self.brokerage
            + self.stt
            + self.exchange_txn
            + self.sebi_fee
            + self.stamp_duty
            + self.gst
            + self.slippage
            + self.mtf_interest
        )


def _brokerage_per_order(position_value: float, pct: float, flat: float) -> float:
    return min(position_value * pct, flat)


def intraday_costs(position_value: float, cfg: IntradayCostConfig) -> TradeCosts:
    """Round-trip (buy + sell) cost of a same-day trade on `position_value`."""
    brokerage = 2 * _brokerage_per_order(position_value, cfg.brokerage_pct, cfg.brokerage_flat)
    stt = position_value * cfg.stt_sell_pct  # sell side only, intraday
    exchange_txn = 2 * position_value * cfg.exchange_txn_pct
    sebi_fee = (2 * position_value / CRORE) * cfg.sebi_fee_per_crore
    stamp_duty = position_value * cfg.stamp_duty_buy_pct  # buy side only
    gst = cfg.gst_pct * (brokerage + exchange_txn + sebi_fee)
    slippage = 2 * position_value * cfg.slippage_pct
    return TradeCosts(brokerage, stt, exchange_txn, sebi_fee, stamp_duty, gst, slippage, mtf_interest=0.0)


def overnight_costs(
    position_value: float,
    margin: float,
    nights_held: int,
    cfg: OvernightCostConfig,
) -> TradeCosts:
    """Round-trip cost of an MTF trade held `nights_held` calendar nights."""
    brokerage = 2 * _brokerage_per_order(position_value, cfg.brokerage_pct, cfg.brokerage_flat)
    stt = 2 * position_value * cfg.stt_pct  # buy and sell side, MTF
    exchange_txn = 2 * position_value * cfg.exchange_txn_pct
    sebi_fee = (2 * position_value / CRORE) * cfg.sebi_fee_per_crore
    stamp_duty = position_value * cfg.stamp_duty_buy_pct  # buy side only
    gst = cfg.gst_pct * (brokerage + exchange_txn + sebi_fee)
    slippage = 2 * position_value * cfg.slippage_pct

    borrowed = max(position_value - margin, 0.0)
    daily_rate = cfg.mtf_annual_interest_pct / 365
    mtf_interest = borrowed * daily_rate * max(nights_held, 0)

    return TradeCosts(brokerage, stt, exchange_txn, sebi_fee, stamp_duty, gst, slippage, mtf_interest)


def trade_costs(
    position_value: float,
    margin: float,
    entry_date,
    exit_date,
    cfg: CostsConfig,
) -> TradeCosts:
    """Dispatch to intraday or overnight costs based on whether exit is same-day as entry."""
    held_overnight = exit_date > entry_date
    if not held_overnight:
        return intraday_costs(position_value, cfg.intraday)
    nights_held = (exit_date - entry_date).days
    return overnight_costs(position_value, margin, nights_held, cfg.overnight)
