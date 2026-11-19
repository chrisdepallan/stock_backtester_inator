"""Sweep orchestration plus the output files: summary.csv, trades.csv,
equity_curve.png, and report.md with the verdict.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mbt.backtest import generate_trades
from mbt.baseline import build_random_entry_pool, index_buy_and_hold, random_entry_runs
from mbt.config import Config
from mbt.leverage import apply_leverage_and_costs
from mbt.metrics import build_equity_curve, percentile_rank, portfolio_metrics, split_in_out_of_sample, trade_metrics
from mbt.signals import pattern_stats

SUMMARY_COLUMNS = [
    "variant", "threshold", "leverage",
    "is_trades", "is_win_rate", "is_avg_gross_ret", "is_avg_net_ret", "is_expectancy",
    "is_profit_factor", "is_worst_trade_margin_pnl", "is_wipeouts", "is_margin_calls",
    "is_final_equity", "is_cagr", "is_sharpe", "is_max_drawdown",
    "oos_trades", "oos_win_rate", "oos_avg_gross_ret", "oos_avg_net_ret", "oos_expectancy",
    "oos_profit_factor", "oos_worst_trade_margin_pnl", "oos_wipeouts", "oos_margin_calls",
    "oos_final_equity", "oos_cagr", "oos_sharpe", "oos_max_drawdown",
    "oos_percentile_vs_random",
]


def capital_per_trade(cfg: Config) -> float:
    return cfg.position_sizing.starting_capital / cfg.position_sizing.max_open_positions


def _split_metrics_row(trades_priced: pd.DataFrame, cfg: Config) -> dict:
    cap = capital_per_trade(cfg)
    in_sample, out_of_sample = split_in_out_of_sample(
        trades_priced, cfg.validation.in_sample_end, cfg.validation.out_of_sample_start
    )
    is_tm = trade_metrics(in_sample)
    oos_tm = trade_metrics(out_of_sample)
    is_pm = portfolio_metrics(
        build_equity_curve(in_sample, cap, cfg.position_sizing.starting_capital),
        cfg.portfolio.risk_free_rate_annual,
    )
    oos_pm = portfolio_metrics(
        build_equity_curve(out_of_sample, cap, cfg.position_sizing.starting_capital),
        cfg.portfolio.risk_free_rate_annual,
    )
    return {
        "is_trades": is_tm.n_trades, "is_win_rate": is_tm.win_rate,
        "is_avg_gross_ret": is_tm.avg_gross_ret, "is_avg_net_ret": is_tm.avg_net_ret,
        "is_expectancy": is_tm.expectancy, "is_profit_factor": is_tm.profit_factor,
        "is_worst_trade_margin_pnl": is_tm.worst_trade_margin_pnl,
        "is_wipeouts": is_tm.n_wipeouts, "is_margin_calls": is_tm.n_margin_calls,
        "is_final_equity": is_pm.final_equity, "is_cagr": is_pm.cagr,
        "is_sharpe": is_pm.sharpe, "is_max_drawdown": is_pm.max_drawdown,
        "oos_trades": oos_tm.n_trades, "oos_win_rate": oos_tm.win_rate,
        "oos_avg_gross_ret": oos_tm.avg_gross_ret, "oos_avg_net_ret": oos_tm.avg_net_ret,
        "oos_expectancy": oos_tm.expectancy, "oos_profit_factor": oos_tm.profit_factor,
        "oos_worst_trade_margin_pnl": oos_tm.worst_trade_margin_pnl,
        "oos_wipeouts": oos_tm.n_wipeouts, "oos_margin_calls": oos_tm.n_margin_calls,
        "oos_final_equity": oos_pm.final_equity, "oos_cagr": oos_pm.cagr,
        "oos_sharpe": oos_pm.sharpe, "oos_max_drawdown": oos_pm.max_drawdown,
    }


def run_sweep(data: dict[str, pd.DataFrame], cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every variant x threshold x leverage combination.

    Returns (summary_df, trades_df) — one summary row and a slice of priced
    trades per combination. Random-entry baselines are drawn once per
    (variant, threshold) from a shared seeded generator and re-priced (cheaply,
    vectorized) at every leverage level for the percentile-vs-random column.
    """
    cap = capital_per_trade(cfg)
    rows: list[dict] = []
    trades_frames: list[pd.DataFrame] = []

    for variant in cfg.variants:
        pool = build_random_entry_pool(data, variant)

        for threshold in cfg.signals.thresholds:
            trades_raw = generate_trades(
                data, variant, threshold, cfg.signals.filters, cfg.position_sizing.max_open_positions
            )
            n_trades = len(trades_raw)
            baseline_runs = (
                random_entry_runs(pool, n_trades, cfg.validation.random_baseline_runs, cfg.validation.random_seed)
                if n_trades > 0
                else []
            )

            for leverage in cfg.leverage.levels:
                priced = apply_leverage_and_costs(
                    trades_raw, leverage, cap, cfg.costs, cfg.leverage.margin_call_threshold
                )
                tagged = priced.copy()
                tagged.insert(0, "variant", variant)
                tagged.insert(1, "threshold", threshold)
                trades_frames.append(tagged)

                row = {"variant": variant, "threshold": threshold, "leverage": leverage}
                row.update(_split_metrics_row(priced, cfg))

                oos_percentile = float("nan")
                if baseline_runs:
                    _, real_oos = split_in_out_of_sample(
                        priced, cfg.validation.in_sample_end, cfg.validation.out_of_sample_start
                    )
                    real_oos_expectancy = trade_metrics(real_oos).expectancy
                    oos_expectancies = []
                    for run in baseline_runs:
                        run_priced = apply_leverage_and_costs(
                            run, leverage, cap, cfg.costs, cfg.leverage.margin_call_threshold
                        )
                        _, run_oos = split_in_out_of_sample(
                            run_priced, cfg.validation.in_sample_end, cfg.validation.out_of_sample_start
                        )
                        oos_expectancies.append(trade_metrics(run_oos).expectancy)
                    oos_percentile = percentile_rank(real_oos_expectancy, np.array(oos_expectancies))
                row["oos_percentile_vs_random"] = oos_percentile

                rows.append(row)

    summary_df = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    trades_df = pd.concat(trades_frames, ignore_index=True) if trades_frames else pd.DataFrame()
    return summary_df, trades_df


def write_summary_csv(summary_df: pd.DataFrame, path: str | Path) -> None:
    summary_df.to_csv(path, index=False)


def write_trades_csv(trades_df: pd.DataFrame, path: str | Path) -> None:
    trades_df.to_csv(path, index=False)


def pick_best_config(summary_df: pd.DataFrame) -> pd.Series | None:
    """Pick the variant x threshold combo with the best in-sample expectancy at 1x.

    Parameters must be picked on in-sample data only (never out-of-sample) to
    avoid the overfitting pitfall the spec calls out.
    """
    lev1 = summary_df.loc[summary_df["leverage"] == 1].dropna(subset=["is_expectancy"])
    if lev1.empty:
        return None
    return lev1.loc[lev1["is_expectancy"].idxmax()]


def determine_verdict(selected: pd.Series, oos_pattern_p_value: float) -> tuple[str, list[str]]:
    """Apply the spec's verdict rule to the in-sample-selected config's OOS results."""
    checks = [
        (selected["oos_expectancy"] > 0, f"out-of-sample net expectancy at 1x: {selected['oos_expectancy']:.4%}"),
        (oos_pattern_p_value < 0.05, f"out-of-sample H1 continuation p-value: {oos_pattern_p_value:.4g}"),
        (
            selected["oos_percentile_vs_random"] >= 95,
            f"beat {selected['oos_percentile_vs_random']:.1f}% of random-entry runs out-of-sample",
        ),
    ]
    passed = all(ok for ok, _ in checks)
    verdict = "worth paper trading" if passed else "rejected"
    return verdict, [msg for _, msg in checks]


def _reindex_ffill(series: pd.Series, calendar: pd.DatetimeIndex, fill_start_value: float) -> pd.Series:
    combined = series.reindex(series.index.union(calendar)).sort_index().ffill()
    return combined.reindex(calendar).fillna(fill_start_value)


def build_equity_chart(
    data: dict[str, pd.DataFrame],
    cfg: Config,
    selected: pd.Series,
    trades_df: pd.DataFrame,
    path: str | Path,
) -> None:
    """equity_curve.png: strategy (selected config, 1x) vs NIFTY 50 buy & hold
    vs the median of the random-entry baseline runs.

    `data` must include the benchmark ticker (for the buy-and-hold curve);
    it is excluded from the random-entry sampling pool, which should only
    draw from the same stock universe the strategy trades.
    """
    cap = capital_per_trade(cfg)
    starting_capital = cfg.position_sizing.starting_capital
    stock_data = {t: df for t, df in data.items() if t != cfg.universe.benchmark}

    selected_trades = trades_df.loc[
        (trades_df["variant"] == selected["variant"])
        & (trades_df["threshold"] == selected["threshold"])
        & (trades_df["leverage"] == 1)
    ]
    if selected_trades.empty:
        return

    strategy_equity = build_equity_curve(selected_trades, cap, starting_capital)
    start, end = strategy_equity.index[0], strategy_equity.index[-1]

    benchmark_df = data[cfg.universe.benchmark]
    benchmark_curve = index_buy_and_hold(benchmark_df, start, end, starting_capital)
    if benchmark_curve.empty:
        return
    calendar = benchmark_curve.index

    pool = build_random_entry_pool(stock_data, selected["variant"])
    baseline_runs = random_entry_runs(
        pool, len(selected_trades), cfg.validation.random_baseline_runs, cfg.validation.random_seed
    )
    baseline_curves = []
    for run in baseline_runs:
        priced = apply_leverage_and_costs(run, 1.0, cap, cfg.costs, cfg.leverage.margin_call_threshold)
        eq = build_equity_curve(priced, cap, starting_capital)
        baseline_curves.append(_reindex_ffill(eq, calendar, starting_capital))
    random_median = pd.concat(baseline_curves, axis=1).median(axis=1) if baseline_curves else None

    strategy_on_cal = _reindex_ffill(strategy_equity, calendar, starting_capital)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(strategy_on_cal.index, strategy_on_cal.values, label=f"Strategy ({selected['variant']}, thr={selected['threshold']:.0%})")
    ax.plot(benchmark_curve.index, benchmark_curve.values, label="NIFTY 50 buy & hold")
    if random_median is not None:
        ax.plot(random_median.index, random_median.values, label="Random entry (median)", linestyle="--")
    ax.set_ylabel("Equity (Rs)")
    ax.set_title("Equity curve: strategy vs benchmark vs random")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_report_markdown(
    data: dict[str, pd.DataFrame],
    cfg: Config,
    summary_df: pd.DataFrame,
    chart_path: str,
) -> str:
    """report.md: pattern stats, summary tables, verdict, and known limitations.

    `data` must include the benchmark ticker; it is excluded from the H1/H2
    pattern statistics, which are about the traded stock universe only.
    """
    stock_data = {t: df for t, df in data.items() if t != cfg.universe.benchmark}
    full_pattern = pattern_stats(stock_data)

    oos_start = pd.Timestamp(cfg.validation.out_of_sample_start)
    oos_data = {t: df.loc[df.index >= oos_start] for t, df in stock_data.items()}
    oos_pattern = pattern_stats(oos_data)

    selected = pick_best_config(summary_df)
    lines = ["# NSE Short-Term Momentum Backtest — Report", ""]

    lines += [
        "## H1/H2 pattern statistics (full period)",
        "",
        f"- P(day up), unconditional: {full_pattern.p_up_unconditional:.4f}",
        f"- H1 — P(day 2 up | day 1 up): {full_pattern.p_day2up_given_day1up:.4f} "
        f"(n={full_pattern.n_day1_up}, p={full_pattern.p_value_h1:.4g})",
        f"- H2 — P(day 3 down | day 1 up and day 2 up): {full_pattern.p_day3down_given_day1up_day2up:.4f} "
        f"(n={full_pattern.n_day1up_day2up}, p={full_pattern.p_value_h2:.4g})",
        f"- Mean day-2 return after day-1 up: {full_pattern.mean_day2_ret_after_day1up:.4%} "
        f"(t={full_pattern.t_stat_day2:.2f}, p={full_pattern.p_value_day2_ret:.4g})",
        f"- Mean day-3 return after day-1/day-2 up: {full_pattern.mean_day3_ret_after_day1up_day2up:.4%} "
        f"(t={full_pattern.t_stat_day3:.2f}, p={full_pattern.p_value_day3_ret:.4g})",
        "",
    ]

    if selected is not None:
        verdict, checks = determine_verdict(selected, oos_pattern.p_value_h1)
        lines += [
            "## Verdict",
            "",
            f"Best in-sample config: variant **{selected['variant']}**, "
            f"threshold **{selected['threshold']:.0%}**, at 1x leverage.",
            "",
            f"**{verdict.upper()}**",
            "",
        ]
        for ok_msg in checks:
            lines.append(f"- {ok_msg}")
        lines.append("")

        lines += ["### What leverage does to this result (same variant/threshold)", ""]
        lines.append("| Leverage | OOS trades | OOS win rate | OOS avg net ret | OOS expectancy | OOS CAGR | OOS max drawdown | OOS wipe-outs | OOS margin calls |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        lev_rows = summary_df.loc[
            (summary_df["variant"] == selected["variant"]) & (summary_df["threshold"] == selected["threshold"])
        ].sort_values("leverage")
        for _, r in lev_rows.iterrows():
            lines.append(
                f"| {r['leverage']:.0f}x | {r['oos_trades']:.0f} | {r['oos_win_rate']:.2%} | "
                f"{r['oos_avg_net_ret']:.4%} | {r['oos_expectancy']:.4%} | {r['oos_cagr']:.2%} | "
                f"{r['oos_max_drawdown']:.2%} | {r['oos_wipeouts']:.0f} | {r['oos_margin_calls']:.0f} |"
            )
        lines.append("")
        lines.append(f"![Equity curve]({chart_path})")
        lines.append("")
    else:
        lines += ["## Verdict", "", "**REJECTED** — no variant/threshold produced any in-sample trades.", ""]

    lines += ["## Full sweep summary", "", summary_df.to_markdown(index=False), ""]

    lines += [
        "## Known limitations",
        "",
        "- **Survivorship bias:** the universe is today's NIFTY 50 constituent list, "
        "which excludes stocks that dropped out since 2015. Results will look slightly "
        "better than reality.",
        "- **Variant D:** the \"day 2 is up\" check compares day-2 close with the entry "
        "price; treat it as a decision made just before close, not one with perfect "
        "end-of-day information.",
        "- **Overfitting:** 4 variants x 4 thresholds x 3 leverage levels means some "
        "combinations will look good by chance. Only the out-of-sample period and the "
        "verdict above should be trusted.",
        "- **Cost rates:** brokerage, STT, exchange, SEBI, stamp duty and MTF interest "
        "rates in config.yaml are taken from the published Zerodha formula but have not "
        "been cross-checked against Zerodha's live brokerage calculator — do so before "
        "trusting absolute cost or P&L figures.",
        "- **Data gaps:** some tickers had corporate actions during the backtest period "
        "(e.g. TATAMOTORS.NS's 2025 demerger) that can leave gaps or symbol changes in "
        "the cached history; check `mbt fetch` output for 0-row tickers.",
        "",
    ]

    return "\n".join(lines)
