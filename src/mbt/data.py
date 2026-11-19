"""DataSource interface, YFinanceSource, and Parquet-backed local caching."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

# A single-day move bigger than this (as a fraction) is treated as bad data
# unless it is the first row for a ticker (no prior close to compare against).
MAX_SANE_DAILY_MOVE = 0.25


class DataSource(ABC):
    """Interface for a source of daily OHLCV bars, one ticker at a time."""

    @abstractmethod
    def fetch(self, ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
        """Return a DataFrame indexed by date with columns open, high, low, close, volume."""
        raise NotImplementedError


class YFinanceSource(DataSource):
    """Fetches daily OHLCV bars from Yahoo Finance via yfinance.

    Yahoo silently returns an empty frame (logged by yfinance as "possibly
    delisted") on transient failures such as rate limiting, not just on
    genuinely unknown tickers. Retries with backoff before giving up.
    """

    def __init__(self, auto_adjust: bool = True, max_retries: int = 3, retry_backoff: float = 2.0) -> None:
        self.auto_adjust = auto_adjust
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

    def fetch(self, ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
        import yfinance as yf

        df = None
        for attempt in range(self.max_retries):
            df = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=self.auto_adjust,
                progress=False,
            )
            if df is not None and not df.empty:
                break
            if attempt < self.max_retries - 1:
                wait = self.retry_backoff * (attempt + 1)
                logger.warning("Empty response for %s (attempt %d/%d), retrying in %.1fs",
                               ticker, attempt + 1, self.max_retries, wait)
                time.sleep(wait)

        if df is None or df.empty:
            logger.warning("No data for %s after %d attempts", ticker, self.max_retries)
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df = df.rename(columns=str.lower)
        df.index.name = "date"
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df[OHLCV_COLUMNS].copy()


class ParquetCache:
    """Wraps a DataSource with a per-ticker Parquet cache in `cache_dir`.

    On each `get`, only the date range missing from the cache is fetched
    from the underlying source; the cache file is then rewritten in full.
    """

    def __init__(self, source: DataSource, cache_dir: str | Path, request_delay: float = 0.0) -> None:
        self.source = source
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.request_delay = request_delay
        self._made_a_request = False

    def _path(self, ticker: str) -> Path:
        safe = ticker.replace("^", "_").replace("&", "_and_")
        return self.cache_dir / f"{safe}.parquet"

    def _read_cache(self, ticker: str) -> pd.DataFrame:
        path = self._path(ticker)
        if not path.exists():
            return pd.DataFrame(columns=OHLCV_COLUMNS)
        cached = pd.read_parquet(path)
        cached.index = pd.to_datetime(cached.index)
        return cached

    def get(self, ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
        cached = self._read_cache(ticker)

        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end) if end else pd.Timestamp(datetime.now().date())

        fetch_start = start_ts
        if not cached.empty:
            fetch_start = max(start_ts, cached.index.max() + pd.Timedelta(days=1))

        if fetch_start <= end_ts:
            if self.request_delay and self._made_a_request:
                time.sleep(self.request_delay)
            self._made_a_request = True
            fresh = self.source.fetch(
                ticker,
                fetch_start.strftime("%Y-%m-%d"),
                (end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            )
            if not fresh.empty:
                combined = pd.concat([cached, fresh])
                combined = combined[~combined.index.duplicated(keep="last")].sort_index()
                combined.to_parquet(self._path(ticker))
                cached = combined

        if cached.empty:
            return cached
        mask = (cached.index >= start_ts) & (cached.index <= end_ts)
        return cached.loc[mask].copy()


def clean_ohlcv(df: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
    """Drop rows with zero/negative volume or an unexplained >25% single-day move.

    Logs what was dropped so corporate-action false positives can be spotted.
    """
    if df.empty:
        return df

    zero_volume = df["volume"] <= 0
    daily_move = df["close"].pct_change().abs()
    huge_move = daily_move > MAX_SANE_DAILY_MOVE
    bad = zero_volume | huge_move.fillna(False)

    n_bad = int(bad.sum())
    if n_bad:
        logger.info(
            "Dropping %d bad row(s) for %s: %s",
            n_bad,
            ticker or "<unknown>",
            [str(d.date()) for d in df.index[bad]],
        )
    return df.loc[~bad].copy()


def fetch_universe(
    tickers: list[str],
    start: str,
    end: str | None,
    cache_dir: str | Path,
    auto_adjust: bool = True,
    request_delay: float = 0.5,
    on_ticker_done: Callable[[str], None] | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch (cached) and clean OHLCV data for every ticker in `tickers`.

    `request_delay` pauses between tickers that actually hit the network,
    to stay under Yahoo Finance's rate limit; fully cached tickers incur
    no delay. `on_ticker_done`, if given, is called after each ticker so
    callers can drive a progress bar.
    """
    cache = ParquetCache(YFinanceSource(auto_adjust=auto_adjust), cache_dir, request_delay=request_delay)
    out: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        raw = cache.get(ticker, start, end)
        out[ticker] = clean_ohlcv(raw, ticker)
        if on_ticker_done is not None:
            on_ticker_done(ticker)
    return out
