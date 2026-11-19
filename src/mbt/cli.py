from pathlib import Path

import pandas as pd
import typer

from mbt.backtest import generate_trades
from mbt.config import Config, load_config
from mbt.data import fetch_universe
from mbt.leverage import apply_leverage_and_costs
from mbt.metrics import build_equity_curve, portfolio_metrics, trade_metrics
from mbt.report import build_equity_chart, build_report_markdown, capital_per_trade, run_sweep, write_summary_csv, write_trades_csv
from mbt.signals import pattern_stats

app = typer.Typer(help="NSE short-term momentum backtest")


def _load_universe(cfg: Config) -> dict[str, pd.DataFrame]:
    tickers = cfg.universe.tickers + [cfg.universe.benchmark]
    with typer.progressbar(length=len(tickers), label="Fetching data") as progress:
        return fetch_universe(
            tickers,
            start=cfg.data.start_date,
            end=cfg.data.end_date,
            cache_dir=cfg.data.cache_dir,
            auto_adjust=cfg.data.auto_adjust,
            on_ticker_done=lambda _ticker: progress.update(1),
        )


@app.command()
def fetch(
    config_path: str = typer.Option("config.yaml", "--config", help="Path to config.yaml"),
    limit: int = typer.Option(0, help="Only fetch the first N tickers (0 = all)"),
) -> None:
    """Download or update cached data for the universe."""
    cfg = load_config(config_path)
    if limit:
        cfg = cfg.model_copy(update={"universe": cfg.universe.model_copy(update={"tickers": cfg.universe.tickers[:limit]})})

    data = _load_universe(cfg)
    for ticker, df in data.items():
        span = f"{df.index.min().date()} to {df.index.max().date()}" if not df.empty else "no data"
        typer.echo(f"{ticker}: {len(df)} rows ({span})")


@app.command()
def stats(
    config_path: str = typer.Option("config.yaml", "--config", help="Path to config.yaml"),
) -> None:
    """Print the H1/H2 pattern statistics only."""
    cfg = load_config(config_path)
    with typer.progressbar(length=len(cfg.universe.tickers), label="Fetching data") as progress:
        data = fetch_universe(
            cfg.universe.tickers,
            start=cfg.data.start_date,
            end=cfg.data.end_date,
            cache_dir=cfg.data.cache_dir,
            auto_adjust=cfg.data.auto_adjust,
            on_ticker_done=lambda _ticker: progress.update(1),
        )
    result = pattern_stats(data)

    typer.echo(f"Days analysed: {result.n_days}")
    typer.echo(f"P(day up), unconditional: {result.p_up_unconditional:.4f}")
    typer.echo("")
    typer.echo("H1 (continuation): P(day 2 up | day 1 up)")
    typer.echo(f"  n = {result.n_day1_up}, P = {result.p_day2up_given_day1up:.4f}, p-value = {result.p_value_h1:.4g}")
    typer.echo(f"  mean day-2 return after day-1 up: {result.mean_day2_ret_after_day1up:.4%}"
               f" (t = {result.t_stat_day2:.2f}, p = {result.p_value_day2_ret:.4g})")
    typer.echo("")
    typer.echo("H2 (reversal): P(day 3 down | day 1 up and day 2 up)")
    typer.echo(f"  n = {result.n_day1up_day2up}, P = {result.p_day3down_given_day1up_day2up:.4f}, "
               f"p-value = {result.p_value_h2:.4g}")
    typer.echo(f"  mean day-3 return after day-1/day-2 up: {result.mean_day3_ret_after_day1up_day2up:.4%}"
               f" (t = {result.t_stat_day3:.2f}, p = {result.p_value_day3_ret:.4g})")


@app.command()
def run(
    variant: str = typer.Option(..., help="Variant to run: A, B, C or D"),
    threshold: float = typer.Option(..., help="Day-1 return threshold, e.g. 0.01"),
    leverage: float = typer.Option(1.0, help="Leverage level, e.g. 1, 2 or 5"),
    config_path: str = typer.Option("config.yaml", "--config", help="Path to config.yaml"),
) -> None:
    """Run one backtest and print the summary."""
    cfg = load_config(config_path)
    data = _load_universe(cfg)
    stock_data = {t: data[t] for t in cfg.universe.tickers}

    trades_raw = generate_trades(stock_data, variant, threshold, cfg.signals.filters, cfg.position_sizing.max_open_positions)
    cap = capital_per_trade(cfg)
    trades = apply_leverage_and_costs(trades_raw, leverage, cap, cfg.costs, cfg.leverage.margin_call_threshold)

    tm = trade_metrics(trades)
    pm = portfolio_metrics(build_equity_curve(trades, cap, cfg.position_sizing.starting_capital), cfg.portfolio.risk_free_rate_annual)

    typer.echo(f"Variant {variant}, threshold {threshold:.2%}, leverage {leverage:.0f}x")
    typer.echo(f"Trades: {tm.n_trades}")
    typer.echo(f"Win rate: {tm.win_rate:.2%}")
    typer.echo(f"Avg gross / net return per trade: {tm.avg_gross_ret:.4%} / {tm.avg_net_ret:.4%}")
    typer.echo(f"Expectancy: {tm.expectancy:.4%}")
    typer.echo(f"Profit factor: {tm.profit_factor:.2f}")
    typer.echo(f"Worst trade (% of margin): {tm.worst_trade_margin_pnl:.2%}")
    typer.echo(f"Wipe-outs: {tm.n_wipeouts}, margin calls: {tm.n_margin_calls}")
    typer.echo(f"Final equity: Rs {pm.final_equity:,.0f} (started at Rs {cfg.position_sizing.starting_capital:,.0f})")
    typer.echo(f"CAGR: {pm.cagr:.2%}, max drawdown: {pm.max_drawdown:.2%}, Sharpe: {pm.sharpe:.2f}")


@app.command()
def sweep(
    config_path: str = typer.Option("config.yaml", "--config", help="Path to config.yaml"),
    reports_dir: str = typer.Option("reports", help="Directory to write summary.csv and trades.csv into"),
) -> None:
    """Run all variants x thresholds x leverage levels and write summary.csv + trades.csv."""
    cfg = load_config(config_path)
    data = _load_universe(cfg)
    stock_data = {t: data[t] for t in cfg.universe.tickers}

    total_combos = len(cfg.variants) * len(cfg.signals.thresholds) * len(cfg.leverage.levels)
    with typer.progressbar(length=total_combos, label="Running sweep") as progress:
        summary_df, trades_df = run_sweep(stock_data, cfg, on_combo_done=lambda: progress.update(1))

    out_dir = Path(reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_summary_csv(summary_df, out_dir / "summary.csv")
    write_trades_csv(trades_df, out_dir / "trades.csv")

    typer.echo(f"Wrote {len(summary_df)} summary rows to {out_dir / 'summary.csv'}")
    typer.echo(f"Wrote {len(trades_df)} trades to {out_dir / 'trades.csv'}")


@app.command()
def report(
    config_path: str = typer.Option("config.yaml", "--config", help="Path to config.yaml"),
    reports_dir: str = typer.Option("reports", help="Directory with summary.csv/trades.csv, and where report.md is written"),
) -> None:
    """Build report.md and the equity curve chart from the latest sweep."""
    cfg = load_config(config_path)
    data = _load_universe(cfg)  # includes the benchmark, needed for the buy-and-hold curve

    out_dir = Path(reports_dir)
    summary_df = pd.read_csv(out_dir / "summary.csv")
    trades_df = pd.read_csv(out_dir / "trades.csv", parse_dates=["signal_date", "entry_date", "exit_date"])

    chart_filename = "equity_curve.png"
    selected = summary_df.loc[summary_df["leverage"] == 1].dropna(subset=["is_expectancy"])
    if not selected.empty:
        best = selected.loc[selected["is_expectancy"].idxmax()]
        build_equity_chart(data, cfg, best, trades_df, out_dir / chart_filename)

    report_text = build_report_markdown(data, cfg, summary_df, chart_filename)
    (out_dir / "report.md").write_text(report_text, encoding="utf-8")
    typer.echo(f"Wrote {out_dir / 'report.md'}")


if __name__ == "__main__":
    app()
