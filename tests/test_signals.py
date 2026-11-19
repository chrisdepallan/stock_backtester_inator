import numpy as np
import pandas as pd
import pytest

from mbt.config import FiltersConfig
from mbt.signals import compute_day1_return, compute_signal, pattern_stats


def _make_df(closes: list[float], volume: float = 1000.0) -> pd.DataFrame:
    n = len(closes)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    closes = np.array(closes)
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "volume": [volume] * n,
        },
        index=idx,
    )


def test_day1_return_is_close_to_close():
    df = _make_df([100, 101, 99, 105])
    ret = compute_day1_return(df)
    assert ret.iloc[0] != ret.iloc[0]  # NaN
    assert ret.iloc[1] == pytest.approx(0.01)
    assert ret.iloc[2] == pytest.approx(99 / 101 - 1)


def test_signal_fires_above_threshold():
    df = _make_df([100, 102, 102.5, 90])
    signal = compute_signal(df, threshold=0.01, filters=FiltersConfig())
    assert signal.iloc[1] == True  # noqa: E712  (102/100 - 1 = 2% > 1%)
    assert signal.iloc[2] == False  # noqa: E712 (0.49% < 1%)


def test_signal_is_not_affected_by_prices_after_day1_close():
    """Changing any price after day-1 close must not change that day's signal.

    This is the project's explicit look-ahead guard: the signal on a given
    day may only depend on data up to and including that day's own close.
    """
    df = _make_df([100, 102, 103, 104, 95, 200])
    filters = FiltersConfig(volume_above_20d_avg=True, top_quartile_range=True)
    original = compute_signal(df, threshold=0.0, filters=filters)

    mutated = df.copy()
    # Day-1 is at index 1 (ret 2%); mutate every row strictly after it.
    mutated.loc[mutated.index[2:], ["open", "high", "low", "close", "volume"]] *= 7.0
    mutated_signal = compute_signal(mutated, threshold=0.0, filters=filters)

    assert original.iloc[1] == mutated_signal.iloc[1]
    assert original.iloc[0] == mutated_signal.iloc[0]


def test_pattern_stats_on_perfectly_continuing_series():
    # Monotonically increasing closes: every day is "up", so P(up)=1 and
    # the conditional probabilities collapse to 1 (and 0 for "day3 down").
    df = _make_df([100, 101, 102, 103, 104, 105, 106])
    result = pattern_stats({"X": df})
    assert result.p_up_unconditional == pytest.approx(1.0)
    assert result.p_day2up_given_day1up == pytest.approx(1.0)
    assert result.p_day3down_given_day1up_day2up == pytest.approx(0.0)
