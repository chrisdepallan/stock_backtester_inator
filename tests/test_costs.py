import pandas as pd
import pytest

from mbt.config import IntradayCostConfig, OvernightCostConfig
from mbt.costs import intraday_costs, overnight_costs, trade_costs

INTRADAY_CFG = IntradayCostConfig(
    brokerage_pct=0.0003,
    brokerage_flat=20,
    stt_sell_pct=0.00025,
    exchange_txn_pct=0.00003,
    sebi_fee_per_crore=10,
    stamp_duty_buy_pct=0.00003,
    gst_pct=0.18,
    slippage_pct=0.0005,
)

OVERNIGHT_CFG = OvernightCostConfig(
    brokerage_pct=0.0003,
    brokerage_flat=20,
    stt_pct=0.001,
    exchange_txn_pct=0.00003,
    sebi_fee_per_crore=10,
    stamp_duty_buy_pct=0.00015,
    gst_pct=0.18,
    slippage_pct=0.0005,
    mtf_annual_interest_pct=0.18,
)


def test_intraday_cost_matches_known_trade_within_one_rupee():
    """A round-trip intraday trade of Rs 1,00,000 (e.g. 100 shares at Rs 1,000),
    worked out by hand against Zerodha's published intraday-equity formula:

    - Brokerage: min(0.03% of turnover, Rs 20) per executed order, 2 orders
      -> min(30, 20) * 2 = Rs 40.00
    - STT: 0.025% on the sell side only -> Rs 25.00
    - Exchange transaction charges: 0.003% both sides -> Rs 6.00
    - SEBI fee: Rs 10 per crore of (buy + sell) turnover -> Rs 0.20
    - Stamp duty: 0.003% on the buy side only -> Rs 3.00
    - GST: 18% on (brokerage + exchange + SEBI) = 18% of 46.20 -> Rs 8.316
    - Slippage (assumed, not a real broker charge): 0.05% both sides -> Rs 100.00

    Total: 40 + 25 + 6 + 0.2 + 3 + 8.316 + 100 = Rs 182.516

    NOTE: rates are taken from config.yaml, which the project spec flags as
    approximate until cross-checked against Zerodha's live calculator.
    """
    costs = intraday_costs(position_value=100_000.0, cfg=INTRADAY_CFG)
    expected_total = 182.516
    assert costs.total == pytest.approx(expected_total, abs=1.0)
    assert costs.brokerage == pytest.approx(40.0)
    assert costs.stt == pytest.approx(25.0)
    assert costs.exchange_txn == pytest.approx(6.0)
    assert costs.sebi_fee == pytest.approx(0.2)
    assert costs.stamp_duty == pytest.approx(3.0)
    assert costs.slippage == pytest.approx(100.0)


def test_brokerage_is_capped_at_flat_fee_for_large_trades():
    # Rs 10,00,000 position: 0.03% = Rs 300, which is above the Rs 20 flat cap.
    costs = intraday_costs(position_value=1_000_000.0, cfg=INTRADAY_CFG)
    assert costs.brokerage == pytest.approx(40.0)  # 2 orders x Rs 20 cap


def test_overnight_costs_charge_stt_on_both_sides():
    costs = overnight_costs(position_value=100_000.0, margin=20_000.0, nights_held=1, cfg=OVERNIGHT_CFG)
    assert costs.stt == pytest.approx(2 * 100_000.0 * 0.001)


def test_overnight_costs_charge_interest_only_on_borrowed_amount():
    costs_1x = overnight_costs(position_value=100_000.0, margin=100_000.0, nights_held=1, cfg=OVERNIGHT_CFG)
    assert costs_1x.mtf_interest == pytest.approx(0.0)

    costs_5x = overnight_costs(position_value=100_000.0, margin=20_000.0, nights_held=1, cfg=OVERNIGHT_CFG)
    borrowed = 100_000.0 - 20_000.0
    assert costs_5x.mtf_interest == pytest.approx(borrowed * (0.18 / 365) * 1)


def test_trade_costs_dispatches_on_same_day_vs_overnight():
    from mbt.config import CostsConfig

    cfg = CostsConfig(intraday=INTRADAY_CFG, overnight=OVERNIGHT_CFG)
    same_day = pd.Timestamp("2024-01-02")
    next_day = pd.Timestamp("2024-01-03")

    same_day_costs = trade_costs(100_000.0, 100_000.0, same_day, same_day, cfg)
    assert same_day_costs.mtf_interest == 0.0
    assert same_day_costs.stt == pytest.approx(25.0)  # intraday STT formula

    overnight_trade_costs = trade_costs(100_000.0, 20_000.0, same_day, next_day, cfg)
    assert overnight_trade_costs.mtf_interest > 0.0
    assert overnight_trade_costs.stt == pytest.approx(200.0)  # overnight STT formula
