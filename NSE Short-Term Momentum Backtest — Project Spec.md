    # NSE Short-Term Momentum Backtest — Project Spec

Nov 19, 2026 · @Chris Dennis

## Goal and hypothesis

Build a Python backtest that gives a clear yes/no on one question: after a stock closes up on day 1, does buying it and selling on day 2 or day 3 make money after costs? It should also show what 5x leverage does to that result.

Hypotheses to test:

- **H1 (continuation):** if a stock closes up on day 1, it is more likely than usual to close up on day 2.
- **H2 (reversal):** if a stock closes up on both day 1 and day 2, it is more likely than usual to close down on day 3.
- **H3 (tradeable):** a strategy built on H1/H2 has positive expected return per trade after all costs, out of sample, and beats random entries with the same holding period.

If H3 fails at 1x leverage, the strategy is rejected. Leverage cannot turn a negative edge positive.

## Scope and data

Use daily OHLCV data for NIFTY 50 stocks from January 2015 to the latest available date, fetched with `yfinance` and cached locally.

| Item | Choice |
| --- | --- |
| Universe | NIFTY 50 constituents, NSE tickers with `.NS` suffix (e.g. `RELIANCE.NS`) |
| Benchmark | NIFTY 50 index (`^NSEI`) |
| Period | 2015-01-01 to latest |
| Frequency | Daily: open, high, low, close, volume |
| Prices | `auto_adjust=True` so splits and bonuses don't create fake jumps |
| Cache | One Parquet file per ticker in `data/raw/`; re-download only missing dates |
| Source fallback | Zerodha Kite Connect historical API (paid add-on), behind the same `DataSource` interface |

The ticker list lives in `config.yaml`, so you can switch to NIFTY 100 or NIFTY 200 without code changes.

Known limitation: today's NIFTY 50 list excludes stocks that dropped out since 2015 (survivorship bias). Results will look slightly better than reality, and the report must say so.

## Strategy rules

The signal is a day-1 close-to-close gain above a threshold. Entry is always at day-2 open, because the signal is only known after day-1 close.

**Signal:** `ret_d1 = close[d1] / close[d0] - 1`, and `ret_d1 > threshold`. Sweep threshold over 0%, 1%, 2% and 3%.

**Variants** (each is run separately):

| Variant | Entry | Exit | Holding | Leverage type |
| --- | --- | --- | --- | --- |
| A | Day-2 open | Day-2 close | Intraday | Intraday margin (up to 5x) |
| B | Day-2 open | Day-3 open | 1 night | MTF |
| C | Day-2 open | Day-3 close | 1 night + 1 day | MTF |
| D | Day-2 open | Day-2 close if day 2 is up, else day-3 close | 1–2 days | MTF |

**Optional filters** (off by default, toggled in config):

- Day-1 volume above its 20-day average
- Day-1 close in the top 25% of that day's high–low range
- Skip day 2 if it gaps up more than 2% at open

**Position sizing:** equal capital per trade, at most N open positions at once (default N = 5). If more stocks signal on the same day, take the largest day-1 gains first.

**Baselines** (run alongside every variant):

1. Random entry: same number of trades, same holding period, random dates and stocks. Run 1,000 times with a fixed seed.
2. Buy and hold the NIFTY 50 index.

## Costs, leverage and margin

All costs are charged on the full leveraged position, not just your margin. Every rate lives in `config.yaml` so you can update it.

Default cost rates are approximate and **must be checked against Zerodha's brokerage calculator before trusting results**:

| Charge | Intraday (variant A) | Overnight / MTF (B, C, D) |
| --- | --- | --- |
| Brokerage | 0.03% or ₹20 per order, whichever is lower | Check Zerodha's current MTF brokerage |
| STT | 0.025% on sell side | 0.1% on buy and sell |
| Exchange transaction charge | \~0.003% both sides | \~0.003% both sides |
| SEBI fee | ₹10 per crore | ₹10 per crore |
| Stamp duty | 0.003% on buy side | 0.015% on buy side |
| GST | 18% on brokerage + exchange + SEBI fees | Same |
| Slippage | 0.05% per side (assumed) | 0.05% per side (assumed) |
| MTF interest | None | Daily rate on the borrowed amount (set from Zerodha's current rate) |

**Leverage model** (`leverage.py`):

- Margin = position size / L. Run every variant at L = 1, 2 and 5.
- P&L on margin = L × price return − costs ÷ margin.
- If a single trade's loss reaches 100% of its margin, mark it **wiped out**. Overnight gaps can cause losses larger than the margin; record the full loss and do not cap it.
- Count how many trades would have triggered a margin call (loss above a configurable share of margin, default 50%).
- Variant A must exit at day-2 close, matching broker auto square-off.

## Metrics and outputs

The deliverable is one `reports/report.md`, plus supporting files, that ends with a verdict.

**Pattern statistics** (answer H1 and H2 directly):

- P(day 2 up | day 1 up) vs unconditional P(day up), with a binomial test p-value
- P(day 3 down | day 1 up and day 2 up) vs unconditional P(day down), with p-value
- Mean day-2 and day-3 return after a day-1 up move, with t-stat

**Per-trade metrics** (for each variant × threshold × leverage):

| Metric | Notes |
| --- | --- |
| Trades | Count |
| Win rate | % of trades with net P&L > 0 |
| Avg return per trade | Gross and net of costs, % |
| Expectancy | Avg win × win rate − avg loss × loss rate |
| Profit factor | Gross wins ÷ gross losses |
| Worst trade | % of margin |
| Wipe-outs / margin calls | Count |

**Portfolio metrics:** CAGR, max drawdown, Sharpe ratio (annualised, risk-free rate in config), and final equity starting from ₹1,00,000.

**Validation:**

- In-sample: 2015–2021. Out-of-sample: 2022 to latest. Pick parameters only on in-sample data; report both.
- Percentile of the strategy's result within the 1,000 random-entry runs.

**Files written:**

- `reports/summary.csv`: one row per variant × threshold × leverage
- `reports/trades.csv`: every trade with dates, prices, costs, P&L
- `reports/equity_curve.png`: strategy vs NIFTY 50 vs random median
- `reports/report.md`: tables, charts, and the verdict

**Verdict rule:** "worth paper trading" only if out-of-sample net expectancy at 1x is above zero, the p-value is below 0.05, and the result beats 95% of random runs. Otherwise "rejected".

## Project structure and CLI

It's a Python 3.11+ package named `mbt` (momentum backtest) with a small CLI. Data flows one way: fetch → signals → trades → costs and leverage → metrics → report.

```text
momentum-backtest/
├── pyproject.toml
├── config.yaml
├── README.md
├── src/mbt/
│   ├── config.py      # load + validate config.yaml (pydantic)
│   ├── data.py        # DataSource interface, YFinanceSource, Parquet cache
│   ├── signals.py     # day-1 return, thresholds, filters
│   ├── backtest.py    # turns signals into trades per variant, position limits
│   ├── costs.py       # Indian equity charges, intraday vs MTF
│   ├── leverage.py    # margin, wipe-outs, margin calls, MTF interest
│   ├── baseline.py    # random-entry simulation, index buy-and-hold
│   ├── metrics.py     # pattern stats, per-trade and portfolio metrics
│   ├── report.py      # CSVs, PNG chart, report.md with verdict
│   └── cli.py         # typer app
├── tests/
├── data/              # gitignored cache
└── reports/           # gitignored outputs
```

**Dependencies:** pandas, numpy, yfinance, pyarrow, scipy, matplotlib, pydantic, pyyaml, typer, pytest.

**CLI commands:**

| Command | What it does |
| --- | --- |
| `mbt fetch` | Download or update cached data for the universe |
| `mbt stats` | Print the H1/H2 pattern statistics only |
| `mbt run --variant A --threshold 0.01 --leverage 5` | One backtest, prints summary |
| `mbt sweep` | All variants × thresholds × leverage levels, writes summary.csv |
| `mbt report` | Builds report.md and the chart from the latest sweep |

Trades are stored as a DataFrame with columns: `ticker, signal_date, entry_date, entry_price, exit_date, exit_price, gross_ret, costs, net_ret, leverage, margin_pnl, wiped_out, margin_call`.

## Pitfalls, acceptance criteria and build order

The most common way a backtest like this lies is look-ahead bias, so it gets its own test.

**Pitfalls to guard against:**

- **Look-ahead:** signals may use data up to day-1 close only. Entry is never at day-1 close.
- **Variant D:** the "day 2 is up" check compares day-2 close with entry price. Treat it as a decision made just before close and note this in the report.
- **Survivorship:** see Scope and data; state it in the report.
- **Overfitting:** with 4 variants × 4 thresholds × 3 filters, something will look good by chance. Trust only the out-of-sample period.
- **Missing days and holidays:** use the next trading day in the data, never calendar days.
- **Circuit limits and halts:** if open equals high equals low equals close, skip that trade.
- **Bad data:** drop rows with zero volume or a single-day move above 25% unless a corporate action explains it; log what was dropped.

**Acceptance criteria:**

- [ ] `pytest` passes, including a cost test that matches Zerodha's calculator for one known trade to within ₹1
- [ ] Look-ahead test: changing any price after day-1 close does not change that day's signal
- [ ] Leverage test: a −21% move at 5x gives a wipe-out with loss above 100% of margin
- [ ] Same config and seed always give identical summary.csv
- [ ] `mbt fetch && mbt sweep && mbt report` runs end to end and writes report.md with a verdict

**Build order for Claude Code** (one step per session or commit):

1. Scaffold the package, `pyproject.toml`, `config.yaml`, and the typer CLI with empty commands.
2. `data.py` with Parquet caching; `mbt fetch` working for 3 tickers.
3. `signals.py` and `mbt stats`, plus the look-ahead test. Check the H1/H2 numbers before going further.
4. `backtest.py` for variant A at 1x, no costs.
5. `costs.py` with the cost test.
6. Variants B, C and D, and `leverage.py` with its tests.
7. `baseline.py` (random entries and index).
8. `metrics.py`, `mbt sweep`, in-sample/out-of-sample split.
9. `report.py` and the verdict rule.
10. Run on the full NIFTY 50 and read the report.
