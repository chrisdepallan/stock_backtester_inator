"""Turn day-1 signals into trades per variant, with a position-limit constraint
applied across the whole universe at once.

Entry is always day-2 open. Exit depends on the variant:

- A: day-2 close (intraday)
- B: day-3 open (1 night)
- C: day-3 close (1 night + 1 day)
- D: day-2 close if day 2 is up (close above the entry price), else day-3 close

Produces gross, no-cost, no-leverage trades. `costs.py` and `leverage.py`
turn `gross_ret` into `net_ret` / `margin_pnl` as a later pipeline stage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mbt.config import FiltersConfig
from mbt.signals import compute_day1_return, compute_signal

TRADE_COLUMNS = [
    "ticker",
    "signal_date",
    "entry_date",
    "entry_price",
    "exit_date",
    "exit_price",
    "gross_ret",
]

VALID_VARIANTS = ("A", "B", "C", "D")


def _is_halted(row: pd.Series) -> bool:
    """True if open == high == low == close (a circuit limit / halt)."""
    return row["open"] == row["high"] == row["low"] == row["close"]


def generate_candidate_trades(
    df: pd.DataFrame,
    ticker: str,
    threshold: float,
    variant: str,
    filters: FiltersConfig,
) -> pd.DataFrame:
    """Candidate trades for one ticker, before the cross-universe position limit."""
    if variant not in VALID_VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")

    signal = compute_signal(df, threshold, filters)
    ret_d1 = compute_day1_return(df)
    idx = df.index
    n = len(df)
    sig_positions = np.flatnonzero(signal.to_numpy())

    rows = []
    for i in sig_positions:
        day2_pos = i + 1
        if day2_pos >= n:
            continue  # no day-2 bar yet
        day2 = df.iloc[day2_pos]
        if _is_halted(day2):
            continue

        entry_price = day2["open"]
        if filters.skip_gap_up_pct is not None:
            day1_close = df.iloc[i]["close"]
            gap = entry_price / day1_close - 1
            if gap > filters.skip_gap_up_pct:
                continue
        entry_date = idx[day2_pos]

        if variant == "A":
            exit_date, exit_price = entry_date, day2["close"]
        else:
            day3_pos = i + 2
            if day3_pos >= n:
                continue  # no day-3 bar yet
            day3 = df.iloc[day3_pos]
            if variant == "B":
                exit_date, exit_price = idx[day3_pos], day3["open"]
            elif variant == "C":
                exit_date, exit_price = idx[day3_pos], day3["close"]
            else:  # D
                day2_is_up = day2["close"] > entry_price
                if day2_is_up:
                    exit_date, exit_price = entry_date, day2["close"]
                else:
                    if _is_halted(day3):
                        continue
                    exit_date, exit_price = idx[day3_pos], day3["close"]

        gross_ret = exit_price / entry_price - 1
        rows.append(
            {
                "ticker": ticker,
                "signal_date": idx[i],
                "entry_date": entry_date,
                "entry_price": float(entry_price),
                "exit_date": exit_date,
                "exit_price": float(exit_price),
                "gross_ret": float(gross_ret),
                "ret_d1": float(ret_d1.iloc[i]),
            }
        )

    return pd.DataFrame(rows, columns=TRADE_COLUMNS + ["ret_d1"])


def apply_position_limits(candidates: pd.DataFrame, max_open_positions: int) -> pd.DataFrame:
    """Keep at most `max_open_positions` concurrently open trades.

    Processes entry dates in order; when more candidates signal on the same
    day than there is free capacity for, the largest day-1 gains win.
    """
    if candidates.empty:
        return candidates.drop(columns=["ret_d1"], errors="ignore")

    ordered = candidates.sort_values(["entry_date", "ret_d1"], ascending=[True, False])

    open_until: dict[str, pd.Timestamp] = {}  # ticker -> exit_date of its open trade
    accepted: list[int] = []

    for entry_date, group in ordered.groupby("entry_date", sort=True):
        # A trade that exits on or before this entry date has already freed its slot.
        open_until = {t: d for t, d in open_until.items() if d > entry_date}
        available = max_open_positions - len(open_until)
        if available <= 0:
            continue
        for row in group.itertuples():
            if available <= 0:
                break
            if row.ticker in open_until:
                continue
            accepted.append(row.Index)
            open_until[row.ticker] = row.exit_date
            available -= 1

    return ordered.loc[accepted].sort_values("entry_date").drop(columns=["ret_d1"]).reset_index(drop=True)


def generate_trades(
    data: dict[str, pd.DataFrame],
    variant: str,
    threshold: float,
    filters: FiltersConfig,
    max_open_positions: int,
) -> pd.DataFrame:
    """Generate the trades for one variant x threshold, across the whole universe."""
    candidate_frames = [
        generate_candidate_trades(df, ticker, threshold, variant, filters) for ticker, df in data.items()
    ]
    candidate_frames = [c for c in candidate_frames if not c.empty]
    if not candidate_frames:
        return pd.DataFrame(columns=TRADE_COLUMNS)

    candidates = pd.concat(candidate_frames, ignore_index=True)
    return apply_position_limits(candidates, max_open_positions)
