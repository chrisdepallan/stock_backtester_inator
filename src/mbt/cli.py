import typer

from mbt.config import load_config
from mbt.data import fetch_universe
from mbt.signals import pattern_stats

app = typer.Typer(help="NSE short-term momentum backtest")


@app.command()
def fetch(
    config_path: str = typer.Option("config.yaml", "--config", help="Path to config.yaml"),
    limit: int = typer.Option(0, help="Only fetch the first N tickers (0 = all)"),
) -> None:
    """Download or update cached data for the universe."""
    cfg = load_config(config_path)
    tickers = cfg.universe.tickers + [cfg.universe.benchmark]
    if limit:
        tickers = tickers[:limit]

    data = fetch_universe(
        tickers,
        start=cfg.data.start_date,
        end=cfg.data.end_date,
        cache_dir=cfg.data.cache_dir,
        auto_adjust=cfg.data.auto_adjust,
    )
    for ticker, df in data.items():
        span = f"{df.index.min().date()} to {df.index.max().date()}" if not df.empty else "no data"
        typer.echo(f"{ticker}: {len(df)} rows ({span})")


@app.command()
def stats(
    config_path: str = typer.Option("config.yaml", "--config", help="Path to config.yaml"),
) -> None:
    """Print the H1/H2 pattern statistics only."""
    cfg = load_config(config_path)
    data = fetch_universe(
        cfg.universe.tickers,
        start=cfg.data.start_date,
        end=cfg.data.end_date,
        cache_dir=cfg.data.cache_dir,
        auto_adjust=cfg.data.auto_adjust,
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
) -> None:
    """Run one backtest and print the summary."""
    raise NotImplementedError


@app.command()
def sweep() -> None:
    """Run all variants x thresholds x leverage levels and write summary.csv."""
    raise NotImplementedError


@app.command()
def report() -> None:
    """Build report.md and the equity curve chart from the latest sweep."""
    raise NotImplementedError


if __name__ == "__main__":
    app()
