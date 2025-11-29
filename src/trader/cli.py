"""Command-line interface for the trading platform."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import typer
from rich.console import Console
from rich.table import Table

from trader.config.settings import get_settings
from trader.core.models import TimeFrame
from trader.data.fetcher import get_data_fetcher
from trader.engine.backtest import BacktestEngine, BacktestResult
from trader.strategies.registry import get_strategy, list_strategies

app = typer.Typer(
    name="trader",
    help="Automated stock trading platform with backtesting",
    add_completion=False,
)
console = Console()


# Common stock groups for quick testing
STOCK_GROUPS = {
    "mag7": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
    "faang": ["META", "AAPL", "AMZN", "NFLX", "GOOGL"],
    "tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "INTC", "CRM"],
    "finance": ["JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW"],
    "healthcare": ["JNJ", "UNH", "PFE", "ABBV", "MRK", "TMO", "ABT", "LLY"],
}


@app.command()
def backtest(
    symbol: str = typer.Argument(..., help="Stock symbol to backtest (e.g., AAPL)"),
    strategy: str = typer.Option("sma", "--strategy", "-S", help="Strategy: sma, rsi, macd, momentum"),
    days: int = typer.Option(365, "--days", "-d", help="Number of days of history"),
    fast_period: int = typer.Option(10, "--fast", "-f", help="Fast SMA/MACD period"),
    slow_period: int = typer.Option(50, "--slow", "-s", help="Slow SMA/MACD period"),
    capital: float = typer.Option(100000.0, "--capital", "-c", help="Initial capital"),
    commission: float = typer.Option(0.0, "--commission", help="Commission per trade"),
) -> None:
    """Run a backtest with a trading strategy."""
    asyncio.run(
        _run_backtest(
            symbol=symbol,
            strategy_name=strategy,
            days=days,
            fast_period=fast_period,
            slow_period=slow_period,
            capital=capital,
            commission=commission,
        )
    )


@app.command("scan")
def scan_stocks(
    symbols: str = typer.Argument(
        ...,
        help="Comma-separated symbols or group name (mag7, faang, tech, finance, healthcare)",
    ),
    strategy: str = typer.Option("sma", "--strategy", "-S", help="Strategy to test"),
    days: int = typer.Option(365, "--days", "-d", help="Number of days of history"),
    sort_by: str = typer.Option("sharpe", "--sort", help="Sort by: sharpe, return, drawdown"),
) -> None:
    """Backtest a strategy across multiple stocks and show comparison."""
    # Parse symbols - could be a group name or comma-separated list
    if symbols.lower() in STOCK_GROUPS:
        symbol_list = STOCK_GROUPS[symbols.lower()]
        console.print(f"\n[bold blue]Scanning {symbols.upper()} stocks[/bold blue]")
    else:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        console.print(f"\n[bold blue]Scanning {len(symbol_list)} stocks[/bold blue]")

    asyncio.run(
        _run_scan(
            symbols=symbol_list,
            strategy_name=strategy,
            days=days,
            sort_by=sort_by,
        )
    )


async def _run_scan(
    symbols: list[str],
    strategy_name: str,
    days: int,
    sort_by: str,
) -> None:
    """Run backtests on multiple symbols and display comparison."""
    settings = get_settings()
    fetcher = get_data_fetcher(settings)

    # Create strategy
    try:
        strategy = get_strategy(strategy_name)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from None

    console.print(f"Strategy: {strategy.description}")
    console.print(f"Period: {days} days\n")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    results: list[dict] = []

    with console.status("[bold green]Running backtests...") as status:
        for symbol in symbols:
            status.update(f"[bold green]Testing {symbol}...")

            try:
                # Fetch data
                data = await fetcher.fetch_bars_df(
                    symbol=symbol,
                    timeframe=TimeFrame.DAY,
                    start=start_date,
                    end=end_date,
                )

                if len(data) < strategy.min_bars_required:
                    results.append({
                        "symbol": symbol,
                        "status": "skip",
                        "reason": "Not enough data",
                    })
                    continue

                # Run backtest
                engine = BacktestEngine(initial_capital=100000.0)
                result = await engine.run(
                    strategy=strategy,
                    data=data,
                    symbol=symbol,
                )

                summary = result.summary()
                results.append({
                    "symbol": symbol,
                    "status": "ok",
                    "return_pct": summary["total_return_pct"],
                    "sharpe": summary["sharpe_ratio"],
                    "max_dd": summary["max_drawdown_pct"],
                    "trades": summary["num_trades"],
                    "win_rate": summary["win_rate"],
                })

            except Exception as e:
                results.append({
                    "symbol": symbol,
                    "status": "error",
                    "reason": str(e)[:30],
                })

    # Filter successful results
    ok_results = [r for r in results if r["status"] == "ok"]
    failed_results = [r for r in results if r["status"] != "ok"]

    if not ok_results:
        console.print("[red]No successful backtests[/red]")
        return

    # Sort results
    sort_key = {
        "sharpe": lambda x: x["sharpe"],
        "return": lambda x: x["return_pct"],
        "drawdown": lambda x: -x["max_dd"],  # Lower is better
    }.get(sort_by, lambda x: x["sharpe"])

    ok_results.sort(key=sort_key, reverse=True)

    # Display results table
    table = Table(
        title=f"Strategy: {strategy_name.upper()} | {days} days",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Symbol", style="bold")
    table.add_column("Return", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("Max DD", justify="right")
    table.add_column("Trades", justify="right")
    table.add_column("Win Rate", justify="right")

    for r in ok_results:
        # Color code return
        ret = r["return_pct"]
        if ret >= 10:
            ret_str = f"[green]+{ret:.1f}%[/green]"
        elif ret >= 0:
            ret_str = f"[dim]+{ret:.1f}%[/dim]"
        else:
            ret_str = f"[red]{ret:.1f}%[/red]"

        # Color code Sharpe
        sharpe = r["sharpe"]
        if sharpe >= 1.0:
            sharpe_str = f"[green]{sharpe:.2f}[/green]"
        elif sharpe >= 0.5:
            sharpe_str = f"[dim]{sharpe:.2f}[/dim]"
        else:
            sharpe_str = f"[red]{sharpe:.2f}[/red]"

        # Color code drawdown
        dd = r["max_dd"]
        if dd <= 10:
            dd_str = f"[green]{dd:.1f}%[/green]"
        elif dd <= 20:
            dd_str = f"[dim]{dd:.1f}%[/dim]"
        else:
            dd_str = f"[red]{dd:.1f}%[/red]"

        table.add_row(
            r["symbol"],
            ret_str,
            sharpe_str,
            dd_str,
            str(r["trades"]),
            f"{r['win_rate']:.0f}%",
        )

    console.print(table)

    # Summary stats
    avg_return = sum(r["return_pct"] for r in ok_results) / len(ok_results)
    avg_sharpe = sum(r["sharpe"] for r in ok_results) / len(ok_results)
    winners = sum(1 for r in ok_results if r["return_pct"] > 0)

    console.print("\n[bold]Summary:[/bold]")
    console.print(f"  Stocks tested: {len(ok_results)}")
    console.print(f"  Profitable: {winners}/{len(ok_results)} ({100*winners/len(ok_results):.0f}%)")
    console.print(f"  Avg Return: {avg_return:+.1f}%")
    console.print(f"  Avg Sharpe: {avg_sharpe:.2f}")

    if failed_results:
        console.print(f"\n[dim]Skipped: {', '.join(r['symbol'] for r in failed_results)}[/dim]")

    console.print()


async def _run_backtest(
    symbol: str,
    strategy_name: str,
    days: int,
    fast_period: int,
    slow_period: int,
    capital: float,
    commission: float,
) -> None:
    """Async implementation of backtest command."""
    settings = get_settings()

    # Create strategy based on name
    try:
        if strategy_name.lower() == "sma":
            strategy = get_strategy("sma", fast_period=fast_period, slow_period=slow_period)
        elif strategy_name.lower() == "macd":
            strategy = get_strategy("macd", fast_period=fast_period, slow_period=slow_period)
        elif strategy_name.lower() == "rsi":
            strategy = get_strategy("rsi")
        elif strategy_name.lower() == "momentum":
            strategy = get_strategy("momentum")
        else:
            strategy = get_strategy(strategy_name)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from None

    console.print("\n[bold blue]Trader Backtest[/bold blue]")
    console.print(f"Symbol: {symbol}")
    console.print(f"Strategy: {strategy.description}")
    console.print(f"Period: {days} days")
    console.print(f"Initial Capital: ${capital:,.2f}")

    if not settings.has_alpaca_credentials:
        console.print(
            "\n[yellow]Note: No Alpaca credentials found. Using mock data.[/yellow]"
        )

    # Get data fetcher
    fetcher = get_data_fetcher(settings)

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    # Fetch data
    with console.status(f"[bold green]Fetching data for {symbol}..."):
        data = await fetcher.fetch_bars_df(
            symbol=symbol,
            timeframe=TimeFrame.DAY,
            start=start_date,
            end=end_date,
        )

    if len(data) == 0:
        console.print("[red]Error: No data returned[/red]")
        raise typer.Exit(1)

    console.print(f"Fetched {len(data)} bars")

    # Run backtest
    engine = BacktestEngine(
        initial_capital=capital,
        commission=commission,
    )

    with console.status("[bold green]Running backtest..."):
        result = await engine.run(
            strategy=strategy,
            data=data,
            symbol=symbol,
        )

    # Display results
    _print_results(result)


def _print_results(result: BacktestResult) -> None:
    """Print backtest results using rich formatting."""
    console.print("\n")

    # Summary table
    table = Table(title="Backtest Results", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")

    summary = result.summary()

    table.add_row("Strategy", summary["strategy"])
    table.add_row("Symbol", summary["symbol"])
    table.add_row(
        "Period",
        f"{summary['start_date'][:10]} to {summary['end_date'][:10]}",
    )
    table.add_row("", "")  # Spacer

    table.add_row("Initial Capital", f"${summary['initial_capital']:,.2f}")
    table.add_row("Final Capital", f"${summary['final_capital']:,.2f}")

    # Color code the return
    return_pct = summary["total_return_pct"]
    if return_pct >= 0:
        return_str = f"[green]+{return_pct:.2f}%[/green]"
    else:
        return_str = f"[red]{return_pct:.2f}%[/red]"
    table.add_row("Total Return", return_str)

    table.add_row("", "")  # Spacer
    table.add_row("Total Trades", str(summary["num_trades"]))
    table.add_row("Winning Trades", str(summary["winning_trades"]))
    table.add_row("Losing Trades", str(summary["losing_trades"]))
    table.add_row("Win Rate", f"{summary['win_rate']:.1f}%")
    table.add_row("Profit Factor", f"{summary['profit_factor']:.2f}")

    table.add_row("", "")  # Spacer
    table.add_row("Max Drawdown", f"{summary['max_drawdown_pct']:.2f}%")
    table.add_row("Sharpe Ratio", f"{summary['sharpe_ratio']:.2f}")

    console.print(table)

    # Trade list if any
    if result.trades:
        console.print("\n")
        trades_table = Table(title="Trades", show_header=True, header_style="bold cyan")
        trades_table.add_column("#", style="dim")
        trades_table.add_column("Entry Date")
        trades_table.add_column("Exit Date")
        trades_table.add_column("Qty", justify="right")
        trades_table.add_column("Entry $", justify="right")
        trades_table.add_column("Exit $", justify="right")
        trades_table.add_column("P&L", justify="right")

        for i, trade in enumerate(result.trades[:10], 1):  # Show first 10
            pnl = float(trade.pnl)
            if pnl >= 0:
                pnl_str = f"[green]+${pnl:,.2f}[/green]"
            else:
                pnl_str = f"[red]-${abs(pnl):,.2f}[/red]"

            trades_table.add_row(
                str(i),
                trade.entry_time.strftime("%Y-%m-%d"),
                trade.exit_time.strftime("%Y-%m-%d"),
                str(trade.quantity),
                f"${float(trade.entry_price):,.2f}",
                f"${float(trade.exit_price):,.2f}",
                pnl_str,
            )

        if len(result.trades) > 10:
            trades_table.add_row(
                "...",
                f"({len(result.trades) - 10} more trades)",
                "",
                "",
                "",
                "",
                "",
            )

        console.print(trades_table)

    console.print("\n")


@app.command()
def paper(
    symbols: str = typer.Argument(
        "AAPL", help="Comma-separated symbols to trade (e.g., AAPL,MSFT)"
    ),
    fast_period: int = typer.Option(10, "--fast", "-f", help="Fast SMA period"),
    slow_period: int = typer.Option(50, "--slow", "-s", help="Slow SMA period"),
    capital: float = typer.Option(100000.0, "--capital", "-c", help="Initial capital"),
    interval: int = typer.Option(
        60, "--interval", "-i", help="Check interval in seconds"
    ),
    day_trade: bool = typer.Option(False, "--day-trade", help="Close positions at EOD"),
) -> None:
    """Run paper trading with mock data (no API key needed)."""
    symbol_list = [s.strip().upper() for s in symbols.split(",")]
    asyncio.run(
        _run_paper_trading(
            symbols=symbol_list,
            fast_period=fast_period,
            slow_period=slow_period,
            capital=capital,
            interval=interval,
            day_trade=day_trade,
        )
    )


async def _run_paper_trading(
    symbols: list[str],
    fast_period: int,
    slow_period: int,
    capital: float,
    interval: int,
    day_trade: bool,
) -> None:
    """Run paper trading simulation."""
    from trader.broker.paper import PaperBroker
    from trader.data.fetcher import MockDataFetcher
    from trader.engine.live import EngineConfig, LiveTradingEngine, TradingMode
    from trader.risk.manager import RiskConfig, RiskManager

    console.print("\n[bold blue]Trader Paper Trading[/bold blue]")
    console.print(f"Symbols: {', '.join(symbols)}")
    console.print(f"Strategy: SMA Crossover ({fast_period}/{slow_period})")
    console.print(f"Initial Capital: ${capital:,.2f}")
    console.print(f"Check Interval: {interval}s")
    console.print(f"Mode: {'Day Trading' if day_trade else 'Swing Trading'}")
    console.print("\n[yellow]Using mock data (no API key required)[/yellow]")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")

    # Create components
    broker = PaperBroker(initial_capital=capital)
    data_fetcher = MockDataFetcher(base_price=150.0, seed=42)
    strategy = get_strategy("sma", fast_period=fast_period, slow_period=slow_period)
    risk_manager = RiskManager(
        RiskConfig(
            max_position_size_pct=0.20,  # 20% per position for demo
            max_open_positions=len(symbols),
        )
    )

    config = EngineConfig(
        symbols=symbols,
        trading_mode=TradingMode.DAY if day_trade else TradingMode.SWING,
        check_interval_seconds=interval,
    )

    engine = LiveTradingEngine(
        broker=broker,
        data_fetcher=data_fetcher,
        strategy=strategy,
        risk_manager=risk_manager,
        config=config,
    )

    # Set initial prices for paper broker
    for symbol in symbols:
        broker.set_price(symbol, 150.0)

    try:
        # Run for a limited time in demo mode
        console.print("[green]Starting paper trading engine...[/green]")

        # Run one iteration for demo
        await broker.connect()

        account_value = await broker.get_account_value()
        console.print(f"Account Value: ${account_value:,.2f}")
        console.print(f"Cash: ${await broker.get_cash():,.2f}")

        # Simulate a few trading cycles
        for i in range(3):
            console.print(f"\n[bold]--- Cycle {i + 1} ---[/bold]")

            for symbol in symbols:
                # Update price with some movement
                import random

                current = float(await broker.get_latest_price(symbol))
                new_price = current * (1 + random.uniform(-0.02, 0.02))
                broker.set_price(symbol, new_price)
                console.print(f"{symbol}: ${new_price:.2f}")

            # Would normally call engine._trading_iteration() here
            # For demo, just show status
            positions = await broker.get_positions()
            if positions:
                console.print("Positions:")
                for pos in positions:
                    pnl = pos.unrealized_pnl or 0
                    console.print(
                        f"  {pos.symbol}: {pos.quantity} shares @ ${pos.avg_entry_price:.2f} "
                        f"(P&L: ${pnl:.2f})"
                    )
            else:
                console.print("No open positions")

            await asyncio.sleep(1)

        await broker.disconnect()
        console.print("\n[green]Paper trading demo complete![/green]")
        console.print(
            "\nTo run continuous paper trading with real Alpaca data:\n"
            "1. Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env\n"
            "2. Run: trader live AAPL,MSFT"
        )

    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping...[/yellow]")
        await engine.stop()


@app.command()
def strategies() -> None:
    """List all available trading strategies."""
    console.print("\n[bold blue]Available Strategies[/bold blue]\n")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Name", style="green")
    table.add_column("Description")

    for strat in list_strategies():
        table.add_row(strat["name"], strat["description"])

    console.print(table)
    console.print("\n[dim]Use --strategy/-S to select a strategy for backtesting[/dim]")
    console.print("[dim]Example: trader backtest AAPL --strategy rsi[/dim]\n")


@app.command()
def version() -> None:
    """Show version information."""
    from trader import __version__

    console.print(f"Trader version {__version__}")


if __name__ == "__main__":
    app()
