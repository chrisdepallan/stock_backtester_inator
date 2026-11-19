import pandas as pd
import pytest

from mbt.config import CostsConfig, IntradayCostConfig, OvernightCostConfig
from mbt.leverage import apply_leverage_and_costs

ZERO_INTRADAY = IntradayCostConfig(
    brokerage_pct=0, brokerage_flat=0, stt_sell_pct=0, exchange_txn_pct=0,
    sebi_fee_per_crore=0, stamp_duty_buy_pct=0, gst_pct=0, slippage_pct=0,
)
ZERO_OVERNIGHT = OvernightCostConfig(
    brokerage_pct=0, brokerage_flat=0, stt_pct=0, exchange_txn_pct=0,
    sebi_fee_per_crore=0, stamp_duty_buy_pct=0, gst_pct=0, slippage_pct=0,
    mtf_annual_interest_pct=0,
)
ZERO_COST_CFG = CostsConfig(intraday=ZERO_INTRADAY, overnight=ZERO_OVERNIGHT)


def _trade(gross_ret: float, entry_date="2024-01-02", exit_date="2024-01-02") -> pd.DataFrame:
    entry_price = 100.0
    return pd.DataFrame(
        [
            {
                "ticker": "X",
                "signal_date": pd.Timestamp("2024-01-01"),
                "entry_date": pd.Timestamp(entry_date),
                "entry_price": entry_price,
                "exit_date": pd.Timestamp(exit_date),
                "exit_price": entry_price * (1 + gross_ret),
                "gross_ret": gross_ret,
            }
        ]
    )


def test_minus_21_pct_move_at_5x_is_a_wipeout():
    trades = _trade(gross_ret=-0.21)
    out = apply_leverage_and_costs(
        trades, leverage=5.0, capital_per_trade=20_000.0, costs_cfg=ZERO_COST_CFG, margin_call_threshold=0.5
    )
    row = out.iloc[0]
    assert row["margin_pnl"] == pytest.approx(-1.05)
    assert bool(row["wiped_out"]) is True
    assert bool(row["margin_call"]) is True


def test_small_loss_at_1x_is_neither_wipeout_nor_margin_call():
    trades = _trade(gross_ret=-0.05)
    out = apply_leverage_and_costs(
        trades, leverage=1.0, capital_per_trade=20_000.0, costs_cfg=ZERO_COST_CFG, margin_call_threshold=0.5
    )
    row = out.iloc[0]
    assert row["margin_pnl"] == pytest.approx(-0.05)
    assert bool(row["wiped_out"]) is False
    assert bool(row["margin_call"]) is False


def test_margin_loss_is_not_capped_at_negative_100_pct():
    # A -30% overnight gap at 5x is a -150% margin loss; record it in full.
    trades = _trade(gross_ret=-0.30, entry_date="2024-01-02", exit_date="2024-01-03")
    out = apply_leverage_and_costs(
        trades, leverage=5.0, capital_per_trade=20_000.0, costs_cfg=ZERO_COST_CFG, margin_call_threshold=0.5
    )
    assert out.iloc[0]["margin_pnl"] == pytest.approx(-1.50)


def test_costs_drag_scales_with_leverage_when_brokerage_cap_does_not_bind():
    # Same gross_ret and costs_cfg, large enough position that brokerage is
    # pct-based (not flat-capped) at both leverage levels, so margin_pnl
    # drag from costs scales exactly with leverage: margin_pnl = L * net_ret.
    from mbt.config import CostsConfig as _CostsConfig
    from mbt.config import IntradayCostConfig as _IntradayCostConfig
    from mbt.config import OvernightCostConfig as _OvernightCostConfig

    cfg = _CostsConfig(
        intraday=_IntradayCostConfig(
            brokerage_pct=0.0003, brokerage_flat=20, stt_sell_pct=0.00025, exchange_txn_pct=0.00003,
            sebi_fee_per_crore=10, stamp_duty_buy_pct=0.00003, gst_pct=0.18, slippage_pct=0.0005,
        ),
        overnight=_OvernightCostConfig(
            brokerage_pct=0.0003, brokerage_flat=20, stt_pct=0.001, exchange_txn_pct=0.00003,
            sebi_fee_per_crore=10, stamp_duty_buy_pct=0.00015, gst_pct=0.18, slippage_pct=0.0005,
            mtf_annual_interest_pct=0.18,
        ),
    )
    # Small enough that position_value * brokerage_pct stays under the Rs 20
    # flat cap at both 1x (Rs 5,000) and 5x (Rs 25,000) exposure.
    trades = _trade(gross_ret=0.0)
    out_1x = apply_leverage_and_costs(trades, leverage=1.0, capital_per_trade=5_000.0, costs_cfg=cfg, margin_call_threshold=0.5)
    out_5x = apply_leverage_and_costs(trades, leverage=5.0, capital_per_trade=5_000.0, costs_cfg=cfg, margin_call_threshold=0.5)
    assert out_5x.iloc[0]["margin_pnl"] == pytest.approx(5.0 * out_1x.iloc[0]["margin_pnl"])


def test_brokerage_flat_cap_makes_cost_drag_less_than_leverage_proportional():
    # When the Rs 20 flat cap binds at higher leverage (bigger position
    # value) but not at 1x, cost drag is sub-proportional to leverage.
    from mbt.config import CostsConfig as _CostsConfig
    from mbt.config import IntradayCostConfig as _IntradayCostConfig
    from mbt.config import OvernightCostConfig as _OvernightCostConfig

    cfg = _CostsConfig(
        intraday=_IntradayCostConfig(
            brokerage_pct=0.0003, brokerage_flat=20, stt_sell_pct=0.00025, exchange_txn_pct=0.00003,
            sebi_fee_per_crore=10, stamp_duty_buy_pct=0.00003, gst_pct=0.18, slippage_pct=0.0005,
        ),
        overnight=_OvernightCostConfig(
            brokerage_pct=0.0003, brokerage_flat=20, stt_pct=0.001, exchange_txn_pct=0.00003,
            sebi_fee_per_crore=10, stamp_duty_buy_pct=0.00015, gst_pct=0.18, slippage_pct=0.0005,
            mtf_annual_interest_pct=0.18,
        ),
    )
    trades = _trade(gross_ret=0.0)
    out_1x = apply_leverage_and_costs(trades, leverage=1.0, capital_per_trade=20_000.0, costs_cfg=cfg, margin_call_threshold=0.5)
    out_5x = apply_leverage_and_costs(trades, leverage=5.0, capital_per_trade=20_000.0, costs_cfg=cfg, margin_call_threshold=0.5)
    # Both are pure cost drag since gross_ret=0; 5x drag is worse but less
    # than 5x the 1x drag, because the brokerage cap binds at 5x only.
    drag_1x = -out_1x.iloc[0]["margin_pnl"]
    drag_5x = -out_5x.iloc[0]["margin_pnl"]
    assert drag_5x > drag_1x
    assert drag_5x < 5.0 * drag_1x


def test_empty_trades_returns_empty_frame_with_output_columns():
    out = apply_leverage_and_costs(
        pd.DataFrame(columns=["ticker", "signal_date", "entry_date", "entry_price", "exit_date", "exit_price", "gross_ret"]),
        leverage=1.0,
        capital_per_trade=20_000.0,
        costs_cfg=ZERO_COST_CFG,
        margin_call_threshold=0.5,
    )
    assert out.empty
    assert list(out.columns) == [
        "ticker", "signal_date", "entry_date", "entry_price", "exit_date", "exit_price",
        "gross_ret", "costs", "net_ret", "leverage", "margin_pnl", "wiped_out", "margin_call",
    ]
