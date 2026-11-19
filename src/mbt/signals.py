"""Day-1 return signal, thresholds, optional filters, and H1/H2 pattern statistics.

Every function here must only look at data up to and including the signal
day (day 1). Entry happens at day-2 open, so nothing in this module may
depend on day-2 or later prices — that is what the look-ahead test checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from mbt.config import FiltersConfig


def compute_day1_return(df: pd.DataFrame) -> pd.Series:
    """Close-to-close return realised on each row, i.e. close[t] / close[t-1] - 1."""
    return df["close"].pct_change()


def compute_signal(df: pd.DataFrame, threshold: float, filters: FiltersConfig) -> pd.Series:
    """Boolean Series, aligned on day 1, True where the day-1 up-move signal fires.

    Uses only `df` rows up to and including each day's own index position.
    """
    ret_d1 = compute_day1_return(df)
    mask = ret_d1 > threshold

    if filters.volume_above_20d_avg:
        vol_avg_20d = df["volume"].rolling(20).mean()
        mask &= df["volume"] > vol_avg_20d

    if filters.top_quartile_range:
        day_range = (df["high"] - df["low"]).replace(0, np.nan)
        close_pos_in_range = (df["close"] - df["low"]) / day_range
        mask &= close_pos_in_range >= 0.75

    return mask.fillna(False)


@dataclass
class PatternStats:
    n_days: int
    p_up_unconditional: float

    n_day1_up: int
    p_day2up_given_day1up: float
    p_value_h1: float

    n_day1up_day2up: int
    p_day3down_given_day1up_day2up: float
    p_value_h2: float

    mean_day2_ret_after_day1up: float
    t_stat_day2: float
    p_value_day2_ret: float

    mean_day3_ret_after_day1up_day2up: float
    t_stat_day3: float
    p_value_day3_ret: float


def pattern_stats(data: dict[str, pd.DataFrame]) -> PatternStats:
    """Compute H1/H2 pattern statistics pooled across every ticker in `data`."""
    up_list: list[pd.Series] = []
    day2_given_day1up_list: list[np.ndarray] = []
    day3_given_day1up_day2up_list: list[np.ndarray] = []
    day2_ret_after_day1up_list: list[np.ndarray] = []
    day3_ret_after_day1up_day2up_list: list[np.ndarray] = []

    for df in data.values():
        if len(df) < 4:
            continue
        ret = df["close"].pct_change()
        up = (ret > 0).to_numpy()
        valid = ret.notna().to_numpy()

        up_list.append(pd.Series(up[valid]))

        day1up = up[:-1] & valid[:-1]
        day2up = up[1:]
        day2_given_day1up_list.append(day2up[day1up])
        day2_ret_after_day1up_list.append(ret.to_numpy()[1:][day1up])

        if len(df) >= 3:
            day1up2 = up[:-2] & up[1:-1] & valid[:-2] & valid[1:-1]
            day3up = up[2:]
            day3_given_day1up_day2up_list.append((~day3up)[day1up2])
            day3_ret_after_day1up_day2up_list.append(ret.to_numpy()[2:][day1up2])

    all_up = pd.concat(up_list) if up_list else pd.Series(dtype=bool)
    n_days = len(all_up)
    p_up_unconditional = float(all_up.mean()) if n_days else float("nan")

    day2_given_day1up = np.concatenate(day2_given_day1up_list) if day2_given_day1up_list else np.array([])
    n_day1_up = len(day2_given_day1up)
    p_day2up_given_day1up = float(day2_given_day1up.mean()) if n_day1_up else float("nan")
    k_day2up = int(day2_given_day1up.sum())
    p_value_h1 = (
        stats.binomtest(k_day2up, n_day1_up, p_up_unconditional).pvalue if n_day1_up else float("nan")
    )

    day3down_given_day1up_day2up = (
        np.concatenate(day3_given_day1up_day2up_list) if day3_given_day1up_day2up_list else np.array([])
    )
    n_day1up_day2up = len(day3down_given_day1up_day2up)
    p_day3down_given_day1up_day2up = (
        float(day3down_given_day1up_day2up.mean()) if n_day1up_day2up else float("nan")
    )
    p_down_unconditional = 1.0 - p_up_unconditional
    k_day3down = int(day3down_given_day1up_day2up.sum())
    p_value_h2 = (
        stats.binomtest(k_day3down, n_day1up_day2up, p_down_unconditional).pvalue
        if n_day1up_day2up
        else float("nan")
    )

    day2_rets = (
        np.concatenate(day2_ret_after_day1up_list) if day2_ret_after_day1up_list else np.array([])
    )
    mean_day2_ret = float(day2_rets.mean()) if len(day2_rets) else float("nan")
    t_day2, p_day2 = stats.ttest_1samp(day2_rets, 0.0) if len(day2_rets) > 1 else (float("nan"), float("nan"))

    day3_rets = (
        np.concatenate(day3_ret_after_day1up_day2up_list) if day3_ret_after_day1up_day2up_list else np.array([])
    )
    mean_day3_ret = float(day3_rets.mean()) if len(day3_rets) else float("nan")
    t_day3, p_day3 = stats.ttest_1samp(day3_rets, 0.0) if len(day3_rets) > 1 else (float("nan"), float("nan"))

    return PatternStats(
        n_days=n_days,
        p_up_unconditional=p_up_unconditional,
        n_day1_up=n_day1_up,
        p_day2up_given_day1up=p_day2up_given_day1up,
        p_value_h1=float(p_value_h1),
        n_day1up_day2up=n_day1up_day2up,
        p_day3down_given_day1up_day2up=p_day3down_given_day1up_day2up,
        p_value_h2=float(p_value_h2),
        mean_day2_ret_after_day1up=mean_day2_ret,
        t_stat_day2=float(t_day2),
        p_value_day2_ret=float(p_day2),
        mean_day3_ret_after_day1up_day2up=mean_day3_ret,
        t_stat_day3=float(t_day3),
        p_value_day3_ret=float(p_day3),
    )
