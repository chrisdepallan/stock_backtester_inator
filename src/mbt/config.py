"""Load and validate config.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel


class UniverseConfig(BaseModel):
    tickers: list[str]
    benchmark: str


class DataConfig(BaseModel):
    start_date: str
    end_date: Optional[str] = None
    auto_adjust: bool = True
    cache_dir: str = "data/raw"
    source: Literal["yfinance", "kite"] = "yfinance"


class FiltersConfig(BaseModel):
    volume_above_20d_avg: bool = False
    top_quartile_range: bool = False
    skip_gap_up_pct: Optional[float] = None


class SignalsConfig(BaseModel):
    thresholds: list[float]
    filters: FiltersConfig


class LeverageConfig(BaseModel):
    levels: list[float]
    margin_call_threshold: float = 0.50


class PositionSizingConfig(BaseModel):
    starting_capital: float = 100000
    max_open_positions: int = 5


class IntradayCostConfig(BaseModel):
    brokerage_pct: float
    brokerage_flat: float
    stt_sell_pct: float
    exchange_txn_pct: float
    sebi_fee_per_crore: float
    stamp_duty_buy_pct: float
    gst_pct: float
    slippage_pct: float


class OvernightCostConfig(BaseModel):
    brokerage_pct: float
    brokerage_flat: float
    stt_pct: float
    exchange_txn_pct: float
    sebi_fee_per_crore: float
    stamp_duty_buy_pct: float
    gst_pct: float
    slippage_pct: float
    mtf_annual_interest_pct: float


class CostsConfig(BaseModel):
    intraday: IntradayCostConfig
    overnight: OvernightCostConfig


class ValidationConfig(BaseModel):
    in_sample_end: str
    out_of_sample_start: str
    random_baseline_runs: int = 1000
    random_seed: int = 42


class PortfolioConfig(BaseModel):
    risk_free_rate_annual: float = 0.07


class Config(BaseModel):
    universe: UniverseConfig
    data: DataConfig
    signals: SignalsConfig
    variants: list[str]
    leverage: LeverageConfig
    position_sizing: PositionSizingConfig
    costs: CostsConfig
    validation: ValidationConfig
    portfolio: PortfolioConfig


def load_config(path: str | Path = "config.yaml") -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config.model_validate(raw)
