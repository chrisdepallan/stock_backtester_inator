import typer

app = typer.Typer(help="NSE short-term momentum backtest")


@app.command()
def fetch() -> None:
    """Download or update cached data for the universe."""
    raise NotImplementedError


@app.command()
def stats() -> None:
    """Print the H1/H2 pattern statistics only."""
    raise NotImplementedError


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
