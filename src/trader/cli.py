"""Command-line interface for the trading platform."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from trader.config.settings import get_settings, setup_logging
from trader.core.models import TimeFrame
from trader.data.fetcher import get_data_fetcher
from trader.engine.backtest import BacktestEngine, BacktestResult
from trader.storage import get_trade_store
from trader.strategies.registry import get_strategy, list_strategies

app = typer.Typer(
    name="trader",
    help="Automated stock trading platform with backtesting",
    add_completion=False,
)
console = Console()


@app.callback()
def main_callback() -> None:
    """Initialize logging on startup."""
    settings = get_settings()
    setup_logging(settings)


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
    save: bool = typer.Option(True, "--save/--no-save", help="Save results to database"),
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
            save_results=save,
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
    save_results: bool = True,
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

    # Save to database
    if save_results:
        store = get_trade_store(settings.database_path)
        run_id = store.save_backtest(result)
        console.print(f"[dim]Saved to database: {run_id}[/dim]")

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


# Pre-defined screener filters for common momentum criteria
SCREEN_PRESETS = {
    "momentum": {
        "description": "Stocks above SMA20 and SMA50 with volume",
        "filters": {
            "Average Volume": "Over 500K",
            "Price": "Over $10",
            "20-Day Simple Moving Average": "Price above SMA20",
            "50-Day Simple Moving Average": "Price above SMA50",
        },
    },
    "oversold": {
        "description": "RSI oversold stocks (potential bounce)",
        "filters": {
            "Average Volume": "Over 500K",
            "Price": "Over $10",
            "RSI (14)": "Oversold (30)",
        },
    },
    "overbought": {
        "description": "RSI overbought stocks (potential reversal)",
        "filters": {
            "Average Volume": "Over 500K",
            "Price": "Over $10",
            "RSI (14)": "Overbought (70)",
        },
    },
    "breakout": {
        "description": "Stocks at new highs with volume",
        "filters": {
            "Average Volume": "Over 500K",
            "Price": "Over $10",
            "52-Week High/Low": "New High",
            "Relative Volume": "Over 1.5",
        },
    },
    "value": {
        "description": "Undervalued stocks with strong fundamentals",
        "filters": {
            "Average Volume": "Over 500K",
            "Price": "Over $10",
            "P/E": "Under 15",
            "PEG": "Under 1",
        },
    },
}


@app.command("screen")
def screen_stocks(
    preset: str = typer.Argument(
        ...,
        help="Screener preset: momentum, oversold, overbought, breakout, value",
    ),
    limit: int = typer.Option(20, "--limit", "-l", help="Max stocks to return"),
    scan_strategy: str | None = typer.Option(
        None, "--scan", "-S", help="Run backtest scan with this strategy"
    ),
    scan_days: int = typer.Option(365, "--days", "-d", help="Days for backtest scan"),
) -> None:
    """
    Screen stocks using Finviz filters and optionally backtest them.

    Presets:
      - momentum: Stocks above SMA20/SMA50 with volume
      - oversold: RSI < 30 (potential bounce)
      - overbought: RSI > 70 (potential reversal)
      - breakout: New 52-week highs with volume surge
      - value: Low P/E and PEG ratio

    Examples:
      trader screen momentum
      trader screen oversold --limit 10
      trader screen momentum --scan rsi --days 180
    """
    if preset.lower() not in SCREEN_PRESETS:
        console.print(f"[red]Unknown preset: {preset}[/red]")
        console.print(f"Available: {', '.join(SCREEN_PRESETS.keys())}")
        raise typer.Exit(1)

    preset_config = SCREEN_PRESETS[preset.lower()]
    console.print(f"\n[bold blue]Stock Screener: {preset.upper()}[/bold blue]")
    console.print(f"[dim]{preset_config['description']}[/dim]\n")

    asyncio.run(
        _run_screen(
            filters=preset_config["filters"],
            limit=limit,
            scan_strategy=scan_strategy,
            scan_days=scan_days,
        )
    )


async def _run_screen(
    filters: dict[str, str],
    limit: int,
    scan_strategy: str | None,
    scan_days: int,
) -> None:
    """Run the stock screener and optionally backtest results."""
    try:
        from finvizfinance.screener.overview import Overview
    except ImportError:
        console.print("[red]finvizfinance not installed. Run: uv add finvizfinance[/red]")
        raise typer.Exit(1) from None

    with console.status("[bold green]Screening stocks (this may take a minute)..."):
        foverview = Overview()
        foverview.set_filter(filters_dict=filters)
        df = foverview.screener_view()

    if df is None or len(df) == 0:
        console.print("[yellow]No stocks found matching criteria[/yellow]")
        return

    console.print(f"Found [green]{len(df)}[/green] stocks matching criteria\n")

    # Sort by change (momentum) and limit
    df_sorted = df.sort_values("Change", ascending=False)
    df_limited = df_sorted.head(limit)

    # Display screener results
    table = Table(title=f"Top {limit} Screener Results", show_header=True, header_style="bold cyan")
    table.add_column("Ticker", style="bold")
    table.add_column("Price", justify="right")
    table.add_column("Change", justify="right")
    table.add_column("Volume", justify="right")
    table.add_column("Sector")

    for _, row in df_limited.iterrows():
        change = row.get("Change", 0)
        if change is None:
            change = 0
        if isinstance(change, str):
            change = float(change.replace("%", "")) / 100 if "%" in change else float(change)

        if change >= 0.05:
            change_str = f"[green]+{change:.1%}[/green]"
        elif change >= 0:
            change_str = f"[dim]+{change:.1%}[/dim]"
        else:
            change_str = f"[red]{change:.1%}[/red]"

        volume = row.get("Volume", 0)
        if volume >= 1_000_000:
            vol_str = f"{volume/1_000_000:.1f}M"
        elif volume >= 1_000:
            vol_str = f"{volume/1_000:.0f}K"
        else:
            vol_str = str(int(volume))

        table.add_row(
            row["Ticker"],
            f"${row['Price']:.2f}",
            change_str,
            vol_str,
            row.get("Sector", "N/A")[:15],
        )

    console.print(table)

    # Get list of tickers
    tickers = df_limited["Ticker"].tolist()
    console.print(f"\n[dim]Tickers: {', '.join(tickers)}[/dim]")

    # If scan strategy requested, run backtests
    if scan_strategy:
        console.print(f"\n[bold]Running {scan_strategy.upper()} backtest on screened stocks...[/bold]\n")
        await _run_scan(
            symbols=tickers,
            strategy_name=scan_strategy,
            days=scan_days,
            sort_by="sharpe",
        )


@app.command("screen-list")
def list_screen_presets() -> None:
    """List available screener presets."""
    console.print("\n[bold blue]Available Screener Presets[/bold blue]\n")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Preset", style="green")
    table.add_column("Description")
    table.add_column("Key Filters")

    for name, config in SCREEN_PRESETS.items():
        filters_str = ", ".join(list(config["filters"].keys())[:3])
        if len(config["filters"]) > 3:
            filters_str += "..."
        table.add_row(name, config["description"], filters_str)

    console.print(table)
    console.print("\n[dim]Usage: trader screen <preset> [--limit N] [--scan STRATEGY][/dim]")
    console.print("[dim]Example: trader screen momentum --scan rsi --days 180[/dim]\n")


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


@app.command("runs")
def list_runs(
    strategy: str | None = typer.Option(None, "--strategy", "-S", help="Filter by strategy"),
    symbol: str | None = typer.Option(None, "--symbol", help="Filter by symbol"),
    limit: int = typer.Option(20, "--limit", "-l", help="Number of runs to show"),
) -> None:
    """List recent backtest runs from database."""
    settings = get_settings()
    store = get_trade_store(settings.database_path)

    runs = store.get_backtest_runs(strategy=strategy, symbol=symbol, limit=limit)

    if not runs:
        console.print("[yellow]No backtest runs found.[/yellow]")
        console.print("[dim]Run a backtest first: trader backtest AAPL[/dim]")
        return

    console.print(f"\n[bold blue]Recent Backtest Runs[/bold blue] ({len(runs)} results)\n")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Run ID", style="dim")
    table.add_column("Strategy")
    table.add_column("Symbol")
    table.add_column("Return", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("Trades", justify="right")
    table.add_column("Win Rate", justify="right")
    table.add_column("Date")

    for run in runs:
        ret = run["total_return_pct"]
        if ret >= 10:
            ret_str = f"[green]+{ret:.1f}%[/green]"
        elif ret >= 0:
            ret_str = f"+{ret:.1f}%"
        else:
            ret_str = f"[red]{ret:.1f}%[/red]"

        sharpe = run["sharpe_ratio"]
        if sharpe >= 1.0:
            sharpe_str = f"[green]{sharpe:.2f}[/green]"
        elif sharpe >= 0:
            sharpe_str = f"{sharpe:.2f}"
        else:
            sharpe_str = f"[red]{sharpe:.2f}[/red]"

        # Truncate run_id for display
        run_id_short = run["run_id"][:25] + "..." if len(run["run_id"]) > 28 else run["run_id"]

        table.add_row(
            run_id_short,
            run["strategy_name"],
            run["symbol"],
            ret_str,
            sharpe_str,
            str(run["num_trades"]),
            f"{run['win_rate']:.0f}%",
            run["created_at"][:10],
        )

    console.print(table)
    console.print(f"\n[dim]Database: {settings.database_path}[/dim]\n")


@app.command("trades")
def list_trades(
    run_id: str | None = typer.Option(None, "--run", "-r", help="Filter by run ID"),
    symbol: str | None = typer.Option(None, "--symbol", help="Filter by symbol"),
    limit: int = typer.Option(20, "--limit", "-l", help="Number of trades to show"),
) -> None:
    """List trades from database."""
    settings = get_settings()
    store = get_trade_store(settings.database_path)

    trades = store.get_trades(run_id=run_id, symbol=symbol, limit=limit)

    if not trades:
        console.print("[yellow]No trades found.[/yellow]")
        return

    console.print(f"\n[bold blue]Trade History[/bold blue] ({len(trades)} results)\n")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Symbol", style="bold")
    table.add_column("Entry")
    table.add_column("Exit")
    table.add_column("Qty", justify="right")
    table.add_column("Entry $", justify="right")
    table.add_column("Exit $", justify="right")
    table.add_column("P&L", justify="right")
    table.add_column("P&L %", justify="right")

    for trade in trades:
        pnl = trade["pnl"]
        pnl_pct = trade["pnl_pct"] * 100

        if pnl >= 0:
            pnl_str = f"[green]+${pnl:,.2f}[/green]"
            pnl_pct_str = f"[green]+{pnl_pct:.1f}%[/green]"
        else:
            pnl_str = f"[red]-${abs(pnl):,.2f}[/red]"
            pnl_pct_str = f"[red]{pnl_pct:.1f}%[/red]"

        table.add_row(
            trade["symbol"],
            trade["entry_time"][:10],
            trade["exit_time"][:10],
            str(trade["quantity"]),
            f"${trade['entry_price']:,.2f}",
            f"${trade['exit_price']:,.2f}",
            pnl_str,
            pnl_pct_str,
        )

    console.print(table)

    # Show aggregate stats
    stats = store.get_trade_stats(symbol=symbol)
    if stats["total_trades"] > 0:
        console.print("\n[bold]Aggregate Stats:[/bold]")
        console.print(f"  Total trades: {stats['total_trades']}")
        console.print(f"  Win rate: {stats['win_rate']:.1f}%")
        console.print(f"  Total P&L: ${stats['total_pnl']:,.2f}")
        console.print(f"  Avg P&L: ${stats['avg_pnl']:,.2f} ({stats['avg_pnl_pct']:.1f}%)")
        console.print(f"  Best trade: ${stats['best_trade']:,.2f}")
        console.print(f"  Worst trade: ${stats['worst_trade']:,.2f}")

    console.print()


@app.command("stats")
def show_stats(
    symbol: str | None = typer.Option(None, "--symbol", help="Filter by symbol"),
) -> None:
    """Show aggregate trading statistics."""
    settings = get_settings()
    store = get_trade_store(settings.database_path)

    stats = store.get_trade_stats(symbol=symbol)

    if stats["total_trades"] == 0:
        console.print("[yellow]No trades found.[/yellow]")
        return

    title = f"Trading Stats: {symbol}" if symbol else "Trading Stats: All Symbols"
    console.print(f"\n[bold blue]{title}[/bold blue]\n")

    table = Table(show_header=False)
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")

    table.add_row("Total Trades", str(stats["total_trades"]))
    table.add_row("Winning Trades", str(stats["winning_trades"]))
    table.add_row("Losing Trades", str(stats["losing_trades"]))
    table.add_row("Win Rate", f"{stats['win_rate']:.1f}%")
    table.add_row("", "")

    total_pnl = stats["total_pnl"]
    if total_pnl >= 0:
        table.add_row("Total P&L", f"[green]+${total_pnl:,.2f}[/green]")
    else:
        table.add_row("Total P&L", f"[red]-${abs(total_pnl):,.2f}[/red]")

    table.add_row("Avg P&L per Trade", f"${stats['avg_pnl']:,.2f}")
    table.add_row("Avg Return", f"{stats['avg_pnl_pct']:.2f}%")
    table.add_row("", "")
    table.add_row("Best Trade", f"[green]+${stats['best_trade']:,.2f}[/green]")
    table.add_row("Worst Trade", f"[red]${stats['worst_trade']:,.2f}[/red]")

    console.print(table)
    console.print(f"\n[dim]Database: {settings.database_path}[/dim]\n")


@app.command("intraday-scan")
def intraday_scan(
    symbols: str = typer.Argument(
        ...,
        help="Comma-separated symbols or group name (mag7, faang, tech, finance, healthcare)",
    ),
    hours: float = typer.Option(3.0, "--hours", "-H", help="Lookback period in hours"),
    date: str | None = typer.Option(None, "--date", "-D", help="Date to scan (YYYY-MM-DD), default: today"),
    start_time: str = typer.Option("09:30", "--start", help="Start time in ET (HH:MM)"),
    end_time: str | None = typer.Option(None, "--end", help="End time in ET (HH:MM), default: start + hours"),
    min_change: float = typer.Option(0.0, "--min-change", "-m", help="Minimum % change to show"),
    sort_by: str = typer.Option("change", "--sort", help="Sort by: change, volume"),
    limit: int = typer.Option(20, "--limit", "-l", help="Max results to show"),
) -> None:
    """
    Scan stocks for intraday momentum.

    Fetches intraday bars and calculates % change over the specified period.
    Requires Alpaca API credentials.

    Examples:
      trader intraday-scan mag7 --hours 3
      trader intraday-scan tech --date 2025-11-25 --hours 6
      trader intraday-scan mag7 --date 2025-11-25 --start 09:30 --end 12:00
      trader intraday-scan AAPL,MSFT --date 2025-11-25 --start 14:00 --hours 2
    """
    # Parse symbols
    if symbols.lower() in STOCK_GROUPS:
        symbol_list = STOCK_GROUPS[symbols.lower()]
        console.print(f"\n[bold blue]Intraday Scan: {symbols.upper()}[/bold blue]")
    else:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        console.print(f"\n[bold blue]Intraday Scan: {len(symbol_list)} stocks[/bold blue]")

    asyncio.run(
        _run_intraday_scan(
            symbols=symbol_list,
            hours=hours,
            scan_date=date,
            start_time_str=start_time,
            end_time_str=end_time,
            min_change=min_change,
            sort_by=sort_by,
            limit=limit,
        )
    )


async def _run_intraday_scan(
    symbols: list[str],
    hours: float,
    scan_date: str | None,
    start_time_str: str,
    end_time_str: str | None,
    min_change: float,
    sort_by: str,
    limit: int,
) -> None:
    """Run intraday momentum scan."""
    from zoneinfo import ZoneInfo

    settings = get_settings()

    if not settings.has_alpaca_credentials:
        console.print("[red]Error: Intraday scanning requires Alpaca API credentials.[/red]")
        console.print("[dim]Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env[/dim]")
        raise typer.Exit(1)

    fetcher = get_data_fetcher(settings)

    # Use timezone-aware datetimes
    utc = ZoneInfo("UTC")
    et = ZoneInfo("America/New_York")

    # Parse date
    if scan_date:
        try:
            base_date = datetime.strptime(scan_date, "%Y-%m-%d")
        except ValueError:
            console.print(f"[red]Invalid date format: {scan_date}. Use YYYY-MM-DD[/red]")
            raise typer.Exit(1) from None
    else:
        base_date = datetime.now()

    # Parse start time
    try:
        start_hour, start_min = map(int, start_time_str.split(":"))
    except ValueError:
        console.print(f"[red]Invalid start time: {start_time_str}. Use HH:MM[/red]")
        raise typer.Exit(1) from None

    # Build start datetime in ET, then convert to UTC
    start_et = datetime(
        base_date.year, base_date.month, base_date.day,
        start_hour, start_min, tzinfo=et
    )

    # Calculate end time
    if end_time_str:
        try:
            end_hour, end_min = map(int, end_time_str.split(":"))
            end_et = datetime(
                base_date.year, base_date.month, base_date.day,
                end_hour, end_min, tzinfo=et
            )
        except ValueError:
            console.print(f"[red]Invalid end time: {end_time_str}. Use HH:MM[/red]")
            raise typer.Exit(1) from None
    else:
        end_et = start_et + timedelta(hours=hours)

    # Convert to UTC for API
    start_time = start_et.astimezone(utc)
    end_time = end_et.astimezone(utc)

    console.print(f"Period: {start_et.strftime('%Y-%m-%d %H:%M')} to {end_et.strftime('%H:%M')} ET")
    console.print(f"Min change: {min_change}%\n")

    results: list[dict] = []

    with console.status("[bold green]Fetching intraday data...") as status:
        for symbol in symbols:
            status.update(f"[bold green]Fetching {symbol}...")

            try:
                data = await fetcher.fetch_bars_df(
                    symbol=symbol,
                    timeframe=TimeFrame.MINUTE_5,  # 5-minute bars
                    start=start_time,
                    end=end_time,
                )

                if len(data) < 2:
                    continue

                # Calculate change
                first_price = data["open"].iloc[0]
                last_price = data["close"].iloc[-1]
                change_pct = ((last_price - first_price) / first_price) * 100

                # Calculate volume
                total_volume = data["volume"].sum()

                # Calculate high/low range
                high = data["high"].max()
                low = data["low"].min()
                range_pct = ((high - low) / low) * 100

                results.append({
                    "symbol": symbol,
                    "change_pct": change_pct,
                    "first_price": first_price,
                    "last_price": last_price,
                    "volume": total_volume,
                    "range_pct": range_pct,
                    "bars": len(data),
                })

            except Exception as e:
                logger.debug(f"Error fetching {symbol}: {e}")
                continue

    if not results:
        console.print("[yellow]No data available. Market may be closed or holiday.[/yellow]")
        console.print(f"[dim]Requested: {start_et.strftime('%Y-%m-%d %H:%M')} to {end_et.strftime('%H:%M')} ET[/dim]")
        console.print("[dim]US market hours: 9:30 AM - 4:00 PM ET, Mon-Fri[/dim]")
        return

    # Filter by minimum change
    if min_change > 0:
        results = [r for r in results if abs(r["change_pct"]) >= min_change]

    if not results:
        console.print(f"[yellow]No stocks with >= {min_change}% change[/yellow]")
        return

    # Sort results
    if sort_by == "volume":
        results.sort(key=lambda x: x["volume"], reverse=True)
    else:  # change (default)
        results.sort(key=lambda x: x["change_pct"], reverse=True)

    # Limit results
    results = results[:limit]

    # Display results
    table = Table(
        title=f"Intraday Movers ({hours}h lookback)",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Symbol", style="bold")
    table.add_column("Change", justify="right")
    table.add_column("Price", justify="right")
    table.add_column("Range", justify="right")
    table.add_column("Volume", justify="right")

    for r in results:
        # Color code change
        change = r["change_pct"]
        if change >= 2.0:
            change_str = f"[green bold]+{change:.2f}%[/green bold]"
        elif change >= 0:
            change_str = f"[green]+{change:.2f}%[/green]"
        elif change >= -2.0:
            change_str = f"[red]{change:.2f}%[/red]"
        else:
            change_str = f"[red bold]{change:.2f}%[/red bold]"

        # Format volume
        vol = r["volume"]
        if vol >= 1_000_000:
            vol_str = f"{vol/1_000_000:.1f}M"
        elif vol >= 1_000:
            vol_str = f"{vol/1_000:.0f}K"
        else:
            vol_str = str(int(vol))

        table.add_row(
            r["symbol"],
            change_str,
            f"${r['last_price']:.2f}",
            f"{r['range_pct']:.1f}%",
            vol_str,
        )

    console.print(table)

    # Summary
    gainers = sum(1 for r in results if r["change_pct"] > 0)
    losers = len(results) - gainers
    avg_change = sum(r["change_pct"] for r in results) / len(results)

    console.print(f"\n[dim]Gainers: {gainers} | Losers: {losers} | Avg: {avg_change:+.2f}%[/dim]")
    console.print(f"[dim]Data from {start_et.strftime('%H:%M')} to {end_et.strftime('%H:%M')} ET[/dim]\n")


@app.command()
def version() -> None:
    """Show version information."""
    from trader import __version__

    console.print(f"Trader version {__version__}")


if __name__ == "__main__":
    app()
