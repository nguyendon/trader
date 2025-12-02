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
    "tech": [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "NVDA",
        "META",
        "TSLA",
        "AMD",
        "INTC",
        "CRM",
    ],
    "finance": ["JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW"],
    "healthcare": ["JNJ", "UNH", "PFE", "ABBV", "MRK", "TMO", "ABT", "LLY"],
}


@app.command()
def backtest(
    symbol: str = typer.Argument(..., help="Stock symbol to backtest (e.g., AAPL)"),
    strategy: str = typer.Option(
        "sma", "--strategy", "-S", help="Strategy: sma, rsi, macd, momentum"
    ),
    days: int = typer.Option(365, "--days", "-d", help="Number of days of history"),
    fast_period: int = typer.Option(10, "--fast", "-f", help="Fast SMA/MACD period"),
    slow_period: int = typer.Option(50, "--slow", "-s", help="Slow SMA/MACD period"),
    capital: float = typer.Option(100000.0, "--capital", "-c", help="Initial capital"),
    commission: float = typer.Option(0.0, "--commission", help="Commission per trade"),
    save: bool = typer.Option(
        True, "--save/--no-save", help="Save results to database"
    ),
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
    sort_by: str = typer.Option(
        "sharpe", "--sort", help="Sort by: sharpe, return, drawdown"
    ),
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
                    results.append(
                        {
                            "symbol": symbol,
                            "status": "skip",
                            "reason": "Not enough data",
                        }
                    )
                    continue

                # Run backtest
                engine = BacktestEngine(initial_capital=100000.0)
                result = await engine.run(
                    strategy=strategy,
                    data=data,
                    symbol=symbol,
                )

                summary = result.summary()
                results.append(
                    {
                        "symbol": symbol,
                        "status": "ok",
                        "return_pct": summary["total_return_pct"],
                        "sharpe": summary["sharpe_ratio"],
                        "max_dd": summary["max_drawdown_pct"],
                        "trades": summary["num_trades"],
                        "win_rate": summary["win_rate"],
                    }
                )

            except Exception as e:
                results.append(
                    {
                        "symbol": symbol,
                        "status": "error",
                        "reason": str(e)[:30],
                    }
                )

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
    console.print(
        f"  Profitable: {winners}/{len(ok_results)} ({100 * winners / len(ok_results):.0f}%)"
    )
    console.print(f"  Avg Return: {avg_return:+.1f}%")
    console.print(f"  Avg Sharpe: {avg_sharpe:.2f}")

    if failed_results:
        console.print(
            f"\n[dim]Skipped: {', '.join(r['symbol'] for r in failed_results)}[/dim]"
        )

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
            strategy = get_strategy(
                "sma", fast_period=fast_period, slow_period=slow_period
            )
        elif strategy_name.lower() == "macd":
            strategy = get_strategy(
                "macd", fast_period=fast_period, slow_period=slow_period
            )
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
        console.print(
            "[red]finvizfinance not installed. Run: uv add finvizfinance[/red]"
        )
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
    table = Table(
        title=f"Top {limit} Screener Results",
        show_header=True,
        header_style="bold cyan",
    )
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
            change = (
                float(change.replace("%", "")) / 100 if "%" in change else float(change)
            )

        if change >= 0.05:
            change_str = f"[green]+{change:.1%}[/green]"
        elif change >= 0:
            change_str = f"[dim]+{change:.1%}[/dim]"
        else:
            change_str = f"[red]{change:.1%}[/red]"

        volume = row.get("Volume", 0)
        if volume >= 1_000_000:
            vol_str = f"{volume / 1_000_000:.1f}M"
        elif volume >= 1_000:
            vol_str = f"{volume / 1_000:.0f}K"
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
        console.print(
            f"\n[bold]Running {scan_strategy.upper()} backtest on screened stocks...[/bold]\n"
        )
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
    console.print(
        "\n[dim]Usage: trader screen <preset> [--limit N] [--scan STRATEGY][/dim]"
    )
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


@app.command("multi")
def multi_strategy(
    symbols: str = typer.Argument("AAPL,MSFT", help="Comma-separated symbols to trade"),
    strategies_list: str = typer.Option(
        "sma,rsi,momentum",
        "--strategies",
        "-S",
        help="Comma-separated strategy names",
    ),
    aggregation: str = typer.Option(
        "weighted",
        "--aggregation",
        "-a",
        help="Signal aggregation: weighted, majority, unanimous, any, first",
    ),
    min_confidence: float = typer.Option(
        0.5, "--min-confidence", help="Minimum confidence threshold (0-1)"
    ),
    interval: int = typer.Option(
        60, "--interval", "-i", help="Check interval in seconds"
    ),
    capital: float = typer.Option(100000.0, "--capital", "-c", help="Initial capital"),
) -> None:
    """Run multi-strategy paper trading with multiple strategies."""
    symbol_list = [s.strip().upper() for s in symbols.split(",")]
    strategy_names = [s.strip().lower() for s in strategies_list.split(",")]

    asyncio.run(
        _run_multi_strategy(
            symbols=symbol_list,
            strategy_names=strategy_names,
            aggregation=aggregation,
            min_confidence=min_confidence,
            interval=interval,
            capital=capital,
        )
    )


async def _run_multi_strategy(
    symbols: list[str],
    strategy_names: list[str],
    aggregation: str,
    min_confidence: float,
    interval: int,
    capital: float,
) -> None:
    """Run multi-strategy paper trading."""
    from trader.broker.paper import PaperBroker
    from trader.strategies.multi import MultiStrategyProcessor, create_strategy_group

    console.print("\n[bold blue]Multi-Strategy Paper Trading[/bold blue]\n")
    console.print(f"Symbols: {', '.join(symbols)}")
    console.print(f"Strategies: {', '.join(strategy_names)}")
    console.print(f"Aggregation: {aggregation}")
    console.print(f"Min Confidence: {min_confidence}")
    console.print(f"Initial Capital: ${capital:,.2f}")
    console.print(f"Check Interval: {interval}s")
    console.print("\n[yellow]Using mock data (no API key required)[/yellow]")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")

    # Create strategy group
    try:
        group = create_strategy_group(
            strategies=strategy_names,
            aggregation=aggregation,
            min_confidence=min_confidence,
        )
        processor = MultiStrategyProcessor(group)
    except ValueError as e:
        console.print(f"[red]Error creating strategy group: {e}[/red]")
        raise typer.Exit(1) from None

    console.print(
        f"[green]Initialized {len(processor.strategy_names)} strategies[/green]"
    )

    # Create components
    broker = PaperBroker(initial_capital=capital)

    # Set initial prices for paper broker
    for symbol in symbols:
        broker.set_price(symbol, 150.0)

    try:
        console.print("[green]Starting multi-strategy engine...[/green]")
        await broker.connect()

        account_value = await broker.get_account_value()
        console.print(f"Account Value: ${account_value:,.2f}")
        console.print(f"Cash: ${await broker.get_cash():,.2f}")

        # Demo: Process symbols with multi-strategy
        console.print("\n[bold]Signal Generation Demo:[/bold]\n")

        import random

        import pandas as pd

        # Generate mock data for each symbol
        for symbol in symbols:
            # Create mock OHLCV data
            dates = pd.date_range(end=datetime.now(), periods=100, freq="D")
            prices = [150.0]
            for _ in range(99):
                prices.append(prices[-1] * (1 + random.uniform(-0.02, 0.02)))

            data = pd.DataFrame(
                {
                    "open": prices,
                    "high": [p * 1.01 for p in prices],
                    "low": [p * 0.99 for p in prices],
                    "close": prices,
                    "volume": [1000000] * 100,
                },
                index=dates,
            )

            # Process symbol with multi-strategy
            signal = processor.process_symbol(data, symbol, position=None)

            # Display result
            action_color = {
                "buy": "green",
                "sell": "red",
                "hold": "yellow",
            }[signal.action.value]

            console.print(f"[bold]{symbol}[/bold]:")
            console.print(
                f"  Action: [{action_color}]{signal.action.value.upper()}[/{action_color}]"
            )
            console.print(f"  Confidence: {signal.confidence:.1%}")
            console.print(f"  Reason: {signal.reason}")
            if "scores" in signal.metadata:
                scores = signal.metadata["scores"]
                console.print(
                    f"  Scores: BUY={scores['buy']:.2f}, SELL={scores['sell']:.2f}, HOLD={scores['hold']:.2f}"
                )
            console.print()

        await broker.disconnect()
        console.print("[green]Multi-strategy demo complete![/green]")
        console.print(
            "\n[dim]Tip: Use 'trader strategies' to see available strategies[/dim]"
        )

    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping...[/yellow]")


@app.command("runs")
def list_runs(
    strategy: str | None = typer.Option(
        None, "--strategy", "-S", help="Filter by strategy"
    ),
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

    console.print(
        f"\n[bold blue]Recent Backtest Runs[/bold blue] ({len(runs)} results)\n"
    )

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
        run_id_short = (
            run["run_id"][:25] + "..." if len(run["run_id"]) > 28 else run["run_id"]
        )

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
        console.print(
            f"  Avg P&L: ${stats['avg_pnl']:,.2f} ({stats['avg_pnl_pct']:.1f}%)"
        )
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
    date: str | None = typer.Option(
        None, "--date", "-D", help="Date to scan (YYYY-MM-DD), default: today"
    ),
    start_time: str = typer.Option("09:30", "--start", help="Start time in ET (HH:MM)"),
    end_time: str | None = typer.Option(
        None, "--end", help="End time in ET (HH:MM), default: start + hours"
    ),
    min_change: float = typer.Option(
        0.0, "--min-change", "-m", help="Minimum % change to show"
    ),
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
        console.print(
            f"\n[bold blue]Intraday Scan: {len(symbol_list)} stocks[/bold blue]"
        )

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
        console.print(
            "[red]Error: Intraday scanning requires Alpaca API credentials.[/red]"
        )
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
            console.print(
                f"[red]Invalid date format: {scan_date}. Use YYYY-MM-DD[/red]"
            )
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
        base_date.year, base_date.month, base_date.day, start_hour, start_min, tzinfo=et
    )

    # Calculate end time
    if end_time_str:
        try:
            end_hour, end_min = map(int, end_time_str.split(":"))
            end_et = datetime(
                base_date.year,
                base_date.month,
                base_date.day,
                end_hour,
                end_min,
                tzinfo=et,
            )
        except ValueError:
            console.print(f"[red]Invalid end time: {end_time_str}. Use HH:MM[/red]")
            raise typer.Exit(1) from None
    else:
        end_et = start_et + timedelta(hours=hours)

    # Convert to UTC for API
    start_time = start_et.astimezone(utc)
    end_time = end_et.astimezone(utc)

    console.print(
        f"Period: {start_et.strftime('%Y-%m-%d %H:%M')} to {end_et.strftime('%H:%M')} ET"
    )
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

                results.append(
                    {
                        "symbol": symbol,
                        "change_pct": change_pct,
                        "first_price": first_price,
                        "last_price": last_price,
                        "volume": total_volume,
                        "range_pct": range_pct,
                        "bars": len(data),
                    }
                )

            except Exception as e:
                logger.debug(f"Error fetching {symbol}: {e}")
                continue

    if not results:
        console.print(
            "[yellow]No data available. Market may be closed or holiday.[/yellow]"
        )
        console.print(
            f"[dim]Requested: {start_et.strftime('%Y-%m-%d %H:%M')} to {end_et.strftime('%H:%M')} ET[/dim]"
        )
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
            vol_str = f"{vol / 1_000_000:.1f}M"
        elif vol >= 1_000:
            vol_str = f"{vol / 1_000:.0f}K"
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

    console.print(
        f"\n[dim]Gainers: {gainers} | Losers: {losers} | Avg: {avg_change:+.2f}%[/dim]"
    )
    console.print(
        f"[dim]Data from {start_et.strftime('%H:%M')} to {end_et.strftime('%H:%M')} ET[/dim]\n"
    )


@app.command("live")
def live_trading(
    symbols: str = typer.Argument(
        "AAPL", help="Comma-separated symbols to trade (e.g., AAPL,MSFT)"
    ),
    strategy: str = typer.Option("sma", "--strategy", "-S", help="Strategy to use"),
    interval: int = typer.Option(
        60, "--interval", "-i", help="Check interval in seconds"
    ),
    day_trade: bool = typer.Option(False, "--day-trade", help="Close positions at EOD"),
    max_position: float = typer.Option(
        10000.0, "--max-position", help="Max $ per position"
    ),
    max_portfolio: float = typer.Option(
        50000.0, "--max-portfolio", help="Max $ total in positions"
    ),
    max_daily_loss: float = typer.Option(
        500.0, "--max-daily-loss", help="Stop trading if daily loss exceeds"
    ),
    max_trades: int = typer.Option(20, "--max-trades", help="Max trades per day"),
    stop_loss: float | None = typer.Option(
        None, "--stop-loss", "-sl", help="Stop loss % (e.g., 5 for 5%)"
    ),
    take_profit: float | None = typer.Option(
        None, "--take-profit", "-tp", help="Take profit % (e.g., 10 for 10%)"
    ),
    trailing_stop: float | None = typer.Option(
        None, "--trailing-stop", "-ts", help="Trailing stop % (e.g., 5 for 5%)"
    ),
    confirm: bool = typer.Option(
        True, "--confirm/--no-confirm", help="Require confirmation for each trade"
    ),
) -> None:
    """
    Run live trading with Alpaca (paper or live).

    Connects to Alpaca API and executes trades based on strategy signals.
    Uses ALPACA_PAPER=true by default (paper trading).

    Safety features:
      - Position size limits
      - Portfolio value limits
      - Daily loss limits
      - Max trades per day
      - Stop loss / take profit / trailing stop (bracket orders)
      - Optional trade confirmation

    Examples:
      trader live AAPL --strategy sma
      trader live AAPL,MSFT --interval 300 --day-trade
      trader live NVDA --stop-loss 5 --take-profit 10
      trader live NVDA --trailing-stop 5 --take-profit 10
      trader live AAPL --max-position 5000 --no-confirm
    """
    symbol_list = [s.strip().upper() for s in symbols.split(",")]
    asyncio.run(
        _run_live_trading(
            symbols=symbol_list,
            strategy_name=strategy,
            interval=interval,
            day_trade=day_trade,
            max_position=max_position,
            max_portfolio=max_portfolio,
            max_daily_loss=max_daily_loss,
            max_trades=max_trades,
            stop_loss_pct=stop_loss / 100 if stop_loss else None,
            take_profit_pct=take_profit / 100 if take_profit else None,
            trailing_stop_pct=trailing_stop / 100 if trailing_stop else None,
            confirm=confirm,
        )
    )


async def _run_live_trading(
    symbols: list[str],
    strategy_name: str,
    interval: int,
    day_trade: bool,
    max_position: float,
    max_portfolio: float,
    max_daily_loss: float,
    max_trades: int,
    stop_loss_pct: float | None,
    take_profit_pct: float | None,
    trailing_stop_pct: float | None,
    confirm: bool,
) -> None:
    """Run live trading with Alpaca broker."""
    from trader.broker.alpaca import AlpacaBroker
    from trader.data.fetcher import AlpacaDataFetcher
    from trader.engine.live import (
        EngineConfig,
        LiveTradingEngine,
        SafetyLimits,
        TradingMode,
    )
    from trader.risk.manager import RiskConfig, RiskManager

    settings = get_settings()

    # Check for Alpaca credentials
    if not settings.has_alpaca_credentials:
        console.print(
            "[red]Error: Alpaca API credentials required for live trading.[/red]"
        )
        console.print("\nSet environment variables or create .env file:")
        console.print("  ALPACA_API_KEY=your_key")
        console.print("  ALPACA_SECRET_KEY=your_secret")
        console.print("  ALPACA_PAPER=true  # for paper trading")
        raise typer.Exit(1)

    # Create strategy
    try:
        strat = get_strategy(strategy_name)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from None

    mode = "paper" if settings.alpaca_paper else "LIVE"
    console.print(f"\n[bold blue]Trader Live Trading ({mode})[/bold blue]")
    console.print(f"Symbols: {', '.join(symbols)}")
    console.print(f"Strategy: {strat.description}")
    console.print(f"Check Interval: {interval}s")
    console.print(f"Mode: {'Day Trading' if day_trade else 'Swing Trading'}")

    console.print("\n[bold]Safety Limits:[/bold]")
    console.print(f"  Max per position: ${max_position:,.0f}")
    console.print(f"  Max portfolio: ${max_portfolio:,.0f}")
    console.print(f"  Max daily loss: ${max_daily_loss:,.0f}")
    console.print(f"  Max trades/day: {max_trades}")
    if trailing_stop_pct:
        console.print(f"  Trailing Stop: {trailing_stop_pct * 100:.1f}%")
    elif stop_loss_pct:
        console.print(f"  Stop Loss: {stop_loss_pct * 100:.1f}%")
    if take_profit_pct:
        console.print(f"  Take Profit: {take_profit_pct * 100:.1f}%")
    console.print(f"  Confirmation: {'Required' if confirm else 'Auto'}")

    if not settings.alpaca_paper:
        console.print("\n[red bold]⚠️  WARNING: LIVE TRADING MODE[/red bold]")
        console.print("[red]Real money will be used for trades![/red]")
        if not typer.confirm("Are you sure you want to continue?"):
            console.print("Aborted.")
            raise typer.Exit(0)

    console.print("\n[dim]Press Ctrl+C to stop[/dim]\n")

    # Confirmation callback
    from trader.core.models import Signal

    def on_signal(signal: Signal, symbol: str, quantity: int) -> bool:
        """Prompt for trade confirmation."""
        console.print("\n[yellow]Trade Signal:[/yellow]")
        console.print(f"  Action: {signal.action.value.upper()}")
        console.print(f"  Symbol: {symbol}")
        console.print(f"  Quantity: {quantity}")
        console.print(f"  Reason: {signal.reason}")
        return typer.confirm("Execute this trade?")

    # Create components
    broker = AlpacaBroker(
        api_key=settings.alpaca_api_key.get_secret_value(),
        secret_key=settings.alpaca_secret_key.get_secret_value(),
        paper=settings.alpaca_paper,
    )

    data_fetcher = AlpacaDataFetcher(
        api_key=settings.alpaca_api_key.get_secret_value(),
        secret_key=settings.alpaca_secret_key.get_secret_value(),
    )

    risk_manager = RiskManager(
        RiskConfig(
            max_position_size_pct=0.20,
            max_open_positions=len(symbols),
        )
    )

    safety = SafetyLimits(
        max_position_value=max_position,
        max_portfolio_value=max_portfolio,
        max_loss_per_day=max_daily_loss,
        max_trades_per_day=max_trades,
        require_confirmation=confirm,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        trailing_stop_pct=trailing_stop_pct,
    )

    config = EngineConfig(
        symbols=symbols,
        trading_mode=TradingMode.DAY if day_trade else TradingMode.SWING,
        check_interval_seconds=interval,
        safety=safety,
    )

    # Setup Discord notifications if configured
    notifier = None
    if settings.discord_webhook_url:
        from trader.notifications import DiscordNotifier

        notifier = DiscordNotifier(settings.discord_webhook_url)
        console.print("[dim]Discord notifications enabled[/dim]")

    engine = LiveTradingEngine(
        broker=broker,
        data_fetcher=data_fetcher,
        strategy=strat,
        risk_manager=risk_manager,
        config=config,
        on_signal=on_signal if confirm else None,
        notifier=notifier,
    )

    try:
        console.print("[green]Starting live trading engine...[/green]")
        await engine.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping...[/yellow]")
        await engine.stop()
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        logger.exception("Live trading error")
        raise typer.Exit(1) from None

    console.print("[green]Trading engine stopped.[/green]")


@app.command("status")
def trading_status() -> None:
    """Show current trading status and positions (requires Alpaca)."""
    asyncio.run(_show_trading_status())


async def _show_trading_status() -> None:
    """Show trading status from Alpaca."""
    from trader.broker.alpaca import AlpacaBroker

    settings = get_settings()

    if not settings.has_alpaca_credentials:
        console.print("[red]Error: Alpaca API credentials required.[/red]")
        raise typer.Exit(1)

    broker = AlpacaBroker(
        api_key=settings.alpaca_api_key.get_secret_value(),
        secret_key=settings.alpaca_secret_key.get_secret_value(),
        paper=settings.alpaca_paper,
    )

    try:
        await broker.connect()

        mode = "Paper" if settings.alpaca_paper else "Live"
        console.print(f"\n[bold blue]Trading Status ({mode})[/bold blue]\n")

        # Account info
        account_value = await broker.get_account_value()
        buying_power = await broker.get_buying_power()
        cash = await broker.get_cash()

        console.print("[bold]Account:[/bold]")
        console.print(f"  Equity: ${account_value:,.2f}")
        console.print(f"  Cash: ${cash:,.2f}")
        console.print(f"  Buying Power: ${buying_power:,.2f}")

        # Positions
        positions = await broker.get_positions()

        if positions:
            console.print(f"\n[bold]Positions ({len(positions)}):[/bold]")

            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("Symbol", style="bold")
            table.add_column("Qty", justify="right")
            table.add_column("Avg Entry", justify="right")
            table.add_column("Current", justify="right")
            table.add_column("Value", justify="right")
            table.add_column("P&L", justify="right")
            table.add_column("P&L %", justify="right")

            total_pnl = 0
            for pos in positions:
                pnl = float(pos.unrealized_pnl or 0)
                total_pnl += pnl
                pnl_pct = pos.unrealized_pnl_pct * 100 if pos.unrealized_pnl_pct else 0

                if pnl >= 0:
                    pnl_str = f"[green]+${pnl:,.2f}[/green]"
                    pnl_pct_str = f"[green]+{pnl_pct:.1f}%[/green]"
                else:
                    pnl_str = f"[red]-${abs(pnl):,.2f}[/red]"
                    pnl_pct_str = f"[red]{pnl_pct:.1f}%[/red]"

                table.add_row(
                    pos.symbol,
                    str(pos.quantity),
                    f"${float(pos.avg_entry_price):,.2f}",
                    f"${float(pos.current_price or 0):,.2f}",
                    f"${float(pos.market_value or 0):,.2f}",
                    pnl_str,
                    pnl_pct_str,
                )

            console.print(table)

            if total_pnl >= 0:
                console.print(
                    f"\n[bold]Total Unrealized P&L: [green]+${total_pnl:,.2f}[/green][/bold]"
                )
            else:
                console.print(
                    f"\n[bold]Total Unrealized P&L: [red]-${abs(total_pnl):,.2f}[/red][/bold]"
                )
        else:
            console.print("\n[dim]No open positions[/dim]")

        # Open orders
        open_orders = await broker.get_open_orders()
        if open_orders:
            console.print(f"\n[bold]Open Orders ({len(open_orders)}):[/bold]")
            for order in open_orders:
                console.print(
                    f"  {order.side.value.upper()} {order.quantity} {order.symbol} "
                    f"@ {order.order_type.value} ({order.status.value})"
                )

        await broker.disconnect()
        console.print()

    except Exception as e:
        console.print(f"[red]Error connecting to Alpaca: {e}[/red]")
        raise typer.Exit(1) from None


@app.command()
def version() -> None:
    """Show version information."""
    from trader import __version__

    console.print(f"Trader version {__version__}")


@app.command("config")
def show_config(
    init: bool = typer.Option(False, "--init", help="Create default config file"),
) -> None:
    """Show or initialize configuration file."""
    from trader.config.strategy_config import (
        DEFAULT_CONFIG_PATH,
        create_default_config,
        load_config,
    )

    if init:
        if DEFAULT_CONFIG_PATH.exists():
            console.print(
                f"[yellow]Config already exists at {DEFAULT_CONFIG_PATH}[/yellow]"
            )
            console.print("[dim]Delete it first if you want to recreate[/dim]")
            return

        create_default_config()
        console.print(f"[green]Created config at {DEFAULT_CONFIG_PATH}[/green]")
        console.print("[dim]Edit this file to customize your strategies[/dim]")
        return

    config = load_config()

    console.print("\n[bold blue]Trading Configuration[/bold blue]")
    console.print(f"[dim]File: {DEFAULT_CONFIG_PATH}[/dim]\n")

    # Strategies
    console.print("[bold]Strategies:[/bold]")
    if config.strategies:
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Name")
        table.add_column("Enabled")
        table.add_column("Symbols")
        table.add_column("Params")

        for s in config.strategies:
            status = "[green]Yes[/green]" if s.enabled else "[dim]No[/dim]"
            symbols = ", ".join(s.symbols[:3])
            if len(s.symbols) > 3:
                symbols += f" (+{len(s.symbols) - 3})"
            params = ", ".join(f"{k}={v}" for k, v in list(s.params.items())[:2])
            if len(s.params) > 2:
                params += "..."
            table.add_row(
                s.name,
                status,
                symbols or "[dim]default[/dim]",
                params or "[dim]default[/dim]",
            )

        console.print(table)
    else:
        console.print("  [dim]No strategies configured[/dim]")

    # Risk
    console.print("\n[bold]Risk Settings:[/bold]")
    console.print(f"  Max position size: {config.risk.max_position_size_pct:.0%}")
    console.print(f"  Max daily loss: {config.risk.max_daily_loss_pct:.0%}")
    console.print(f"  Stop loss: {config.risk.stop_loss_pct:.0%}")
    console.print(f"  Max open positions: {config.risk.max_open_positions}")

    # Backtest
    console.print("\n[bold]Backtest Defaults:[/bold]")
    console.print(f"  Initial capital: ${config.backtest.initial_capital:,.0f}")
    console.print(f"  Commission: ${config.backtest.commission:.2f}")
    console.print(f"  Days: {config.backtest.days}")

    # Watchlists
    if config.watchlists:
        console.print("\n[bold]Watchlists:[/bold]")
        for w in config.watchlists:
            console.print(
                f"  {w.name}: {', '.join(w.symbols[:5])}"
                + (f" (+{len(w.symbols) - 5})" if len(w.symbols) > 5 else "")
            )

    console.print()


@app.command("config-run")
def run_from_config(
    strategy_name: str = typer.Argument(..., help="Strategy name from config"),
    days: int | None = typer.Option(
        None, "--days", "-d", help="Override days from config"
    ),
) -> None:
    """Run backtest using settings from config file."""
    from trader.config.strategy_config import load_config

    config = load_config()
    strategy_config = config.get_strategy(strategy_name)

    if strategy_config is None:
        console.print(f"[red]Strategy '{strategy_name}' not found in config[/red]")
        available = [s.name for s in config.strategies]
        console.print(f"[dim]Available: {', '.join(available)}[/dim]")
        raise typer.Exit(1)

    if not strategy_config.enabled:
        console.print(
            f"[yellow]Strategy '{strategy_name}' is disabled in config[/yellow]"
        )
        console.print("[dim]Set enabled: true in config to enable[/dim]")
        raise typer.Exit(1)

    symbols = strategy_config.symbols or config.default_symbols
    backtest_days = days or config.backtest.days

    console.print(f"\n[bold blue]Running {strategy_name} from config[/bold blue]")
    console.print(f"Symbols: {', '.join(symbols)}")
    console.print(f"Params: {strategy_config.params}")
    console.print(f"Days: {backtest_days}\n")

    # Run backtest for each symbol
    asyncio.run(
        _run_config_backtest(
            strategy_name=strategy_name,
            symbols=symbols,
            params=strategy_config.params,
            days=backtest_days,
            capital=config.backtest.initial_capital,
            commission=config.backtest.commission,
        )
    )


async def _run_config_backtest(
    strategy_name: str,
    symbols: list[str],
    params: dict,
    days: int,
    capital: float,
    commission: float,
) -> None:
    """Run backtests from config file."""
    settings = get_settings()
    fetcher = get_data_fetcher(settings)

    # Create strategy with config params
    try:
        strategy = get_strategy(strategy_name, **params)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from None

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    results: list[dict] = []

    with console.status("[bold green]Running backtests...") as status:
        for symbol in symbols:
            status.update(f"[bold green]Testing {symbol}...")

            try:
                data = await fetcher.fetch_bars_df(
                    symbol=symbol,
                    timeframe=TimeFrame.DAY,
                    start=start_date,
                    end=end_date,
                )

                if len(data) < strategy.min_bars_required:
                    continue

                engine = BacktestEngine(initial_capital=capital, commission=commission)
                result = await engine.run(strategy=strategy, data=data, symbol=symbol)

                # Save to database
                store = get_trade_store(settings.database_path)
                store.save_backtest(result)

                summary = result.summary()
                results.append(
                    {
                        "symbol": symbol,
                        "return_pct": summary["total_return_pct"],
                        "sharpe": summary["sharpe_ratio"],
                        "trades": summary["num_trades"],
                    }
                )

            except Exception as e:
                logger.debug(f"Error backtesting {symbol}: {e}")

    if not results:
        console.print("[yellow]No results[/yellow]")
        return

    # Display results
    table = Table(
        title=f"{strategy_name} Results", show_header=True, header_style="bold cyan"
    )
    table.add_column("Symbol")
    table.add_column("Return", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("Trades", justify="right")

    for r in results:
        ret = r["return_pct"]
        ret_str = (
            f"[green]+{ret:.1f}%[/green]" if ret >= 0 else f"[red]{ret:.1f}%[/red]"
        )
        table.add_row(r["symbol"], ret_str, f"{r['sharpe']:.2f}", str(r["trades"]))

    console.print(table)

    avg_return = sum(r["return_pct"] for r in results) / len(results)
    console.print(
        f"\n[dim]Avg return: {avg_return:+.1f}% across {len(results)} symbols[/dim]\n"
    )


@app.command("report")
def performance_report(
    days: int = typer.Option(30, "--days", "-d", help="Number of days to include"),
    live: bool = typer.Option(False, "--live", "-L", help="Show live trades only"),
) -> None:
    """Show performance report with P&L summary."""
    settings = get_settings()
    store = get_trade_store(settings.database_path)

    console.print(f"\n[bold blue]Performance Report ({days} days)[/bold blue]\n")

    if live:
        # Live trading stats
        stats = store.get_live_trade_stats(days=days)
        source = "Live Trading"
    else:
        # Backtest stats
        stats = store.get_trade_stats()
        source = "Backtest"

    if stats["total_trades"] == 0:
        console.print(f"[yellow]No {source.lower()} trades found.[/yellow]")
        console.print("[dim]Run some backtests or live trades first.[/dim]")
        return

    # Summary table
    table = Table(title=f"{source} Performance", show_header=False)
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")

    table.add_row("Total Trades", str(stats["total_trades"]))
    table.add_row("Winning Trades", str(stats["winning_trades"]))
    table.add_row("Losing Trades", str(stats["losing_trades"]))

    win_rate = stats["win_rate"]
    if win_rate >= 50:
        table.add_row("Win Rate", f"[green]{win_rate:.1f}%[/green]")
    else:
        table.add_row("Win Rate", f"[red]{win_rate:.1f}%[/red]")

    table.add_row("", "")

    total_pnl = stats["total_pnl"]
    if total_pnl >= 0:
        table.add_row("Total P&L", f"[green]+${total_pnl:,.2f}[/green]")
    else:
        table.add_row("Total P&L", f"[red]-${abs(total_pnl):,.2f}[/red]")

    table.add_row("Avg P&L/Trade", f"${stats['avg_pnl']:,.2f}")
    table.add_row("Avg Return", f"{stats['avg_pnl_pct']:.2f}%")

    table.add_row("", "")
    table.add_row("Best Trade", f"[green]+${stats['best_trade']:,.2f}[/green]")
    table.add_row("Worst Trade", f"[red]${stats['worst_trade']:,.2f}[/red]")

    if "profit_factor" in stats and stats["profit_factor"] > 0:
        pf = stats["profit_factor"]
        if pf >= 1.5:
            table.add_row("Profit Factor", f"[green]{pf:.2f}[/green]")
        elif pf >= 1.0:
            table.add_row("Profit Factor", f"{pf:.2f}")
        else:
            table.add_row("Profit Factor", f"[red]{pf:.2f}[/red]")

    console.print(table)

    # Daily P&L report if available
    report = store.get_performance_report(days)
    if report["trading_days"] > 0:
        console.print("\n[bold]Daily Summary:[/bold]")
        console.print(f"  Trading Days: {report['trading_days']}")
        console.print(f"  Win Days: {report['win_days']} ({report['win_rate']:.0f}%)")
        console.print(f"  Avg Daily P&L: ${report['avg_daily_pnl']:,.2f}")
        console.print(f"  Best Day: ${report['best_day']:,.2f}")
        console.print(f"  Worst Day: ${report['worst_day']:,.2f}")

    console.print()


@app.command("export")
def export_trades(
    output: str = typer.Argument("trades.csv", help="Output file path"),
    source: str = typer.Option(
        "all", "--source", "-s", help="Source: 'backtest', 'live', or 'all'"
    ),
    symbol: str | None = typer.Option(None, "--symbol", help="Filter by symbol"),
    days: int | None = typer.Option(None, "--days", "-d", help="Filter to last N days"),
    daily: bool = typer.Option(False, "--daily", help="Export daily P&L instead"),
) -> None:
    """Export trades or daily P&L to CSV file."""
    settings = get_settings()
    store = get_trade_store(settings.database_path)

    if daily:
        count = store.export_daily_pnl_csv(output, days or 365)
        if count > 0:
            console.print(
                f"[green]Exported {count} daily P&L records to {output}[/green]"
            )
        else:
            console.print("[yellow]No records to export[/yellow]")
    else:
        count = store.export_trades_csv(output, source, symbol, days)
        if count > 0:
            console.print(f"[green]Exported {count} trades to {output}[/green]")
        else:
            console.print("[yellow]No trades to export[/yellow]")


@app.command("pnl")
def show_pnl(
    days: int = typer.Option(7, "--days", "-d", help="Number of days to show"),
) -> None:
    """Show daily P&L history."""
    settings = get_settings()
    store = get_trade_store(settings.database_path)

    records = store.get_daily_pnl(days)

    if not records:
        console.print("[yellow]No daily P&L records found.[/yellow]")
        console.print("[dim]P&L is recorded during live trading sessions.[/dim]")
        return

    console.print(f"\n[bold blue]Daily P&L (Last {days} Days)[/bold blue]\n")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Date")
    table.add_column("Start Equity", justify="right")
    table.add_column("End Equity", justify="right")
    table.add_column("Daily P&L", justify="right")
    table.add_column("Return", justify="right")
    table.add_column("Trades", justify="right")
    table.add_column("W/L", justify="right")

    total_pnl = 0
    for record in records:
        daily_pnl = record["ending_equity"] - record["starting_equity"]
        total_pnl += daily_pnl
        daily_return = (daily_pnl / record["starting_equity"]) * 100

        if daily_pnl >= 0:
            pnl_str = f"[green]+${daily_pnl:,.2f}[/green]"
            ret_str = f"[green]+{daily_return:.2f}%[/green]"
        else:
            pnl_str = f"[red]-${abs(daily_pnl):,.2f}[/red]"
            ret_str = f"[red]{daily_return:.2f}%[/red]"

        table.add_row(
            record["date"],
            f"${record['starting_equity']:,.2f}",
            f"${record['ending_equity']:,.2f}",
            pnl_str,
            ret_str,
            str(record["num_trades"]),
            f"{record['winning_trades']}/{record['losing_trades']}",
        )

    console.print(table)

    if total_pnl >= 0:
        console.print(f"\n[bold]Total P&L: [green]+${total_pnl:,.2f}[/green][/bold]")
    else:
        console.print(f"\n[bold]Total P&L: [red]-${abs(total_pnl):,.2f}[/red][/bold]")

    console.print()


@app.command("live-trades")
def list_live_trades(
    symbol: str | None = typer.Option(None, "--symbol", help="Filter by symbol"),
    status: str = typer.Option(
        "all", "--status", "-s", help="Filter: open, closed, all"
    ),
    days: int | None = typer.Option(None, "--days", "-d", help="Filter to last N days"),
    limit: int = typer.Option(20, "--limit", "-l", help="Max trades to show"),
) -> None:
    """List live/paper trading history."""
    settings = get_settings()
    store = get_trade_store(settings.database_path)

    status_filter = None if status == "all" else status
    trades = store.get_live_trades(
        symbol=symbol, status=status_filter, days=days, limit=limit
    )

    if not trades:
        console.print("[yellow]No live trades found.[/yellow]")
        console.print("[dim]Run 'trader live' to start paper trading.[/dim]")
        return

    console.print(f"\n[bold blue]Live Trades[/bold blue] ({len(trades)} results)\n")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("ID", style="dim")
    table.add_column("Symbol", style="bold")
    table.add_column("Side")
    table.add_column("Qty", justify="right")
    table.add_column("Entry", justify="right")
    table.add_column("Exit", justify="right")
    table.add_column("P&L", justify="right")
    table.add_column("Status")

    for trade in trades:
        entry_str = f"${trade['entry_price']:,.2f}"

        if trade["status"] == "open":
            exit_str = "[dim]-[/dim]"
            pnl_str = "[dim]-[/dim]"
            status_str = "[yellow]Open[/yellow]"
        else:
            exit_str = f"${trade['exit_price']:,.2f}"
            pnl = trade["pnl"]
            if pnl >= 0:
                pnl_str = f"[green]+${pnl:,.2f}[/green]"
            else:
                pnl_str = f"[red]-${abs(pnl):,.2f}[/red]"
            status_str = "[green]Closed[/green]"

        table.add_row(
            str(trade["id"]),
            trade["symbol"],
            trade["side"].upper(),
            str(trade["quantity"]),
            entry_str,
            exit_str,
            pnl_str,
            status_str,
        )

    console.print(table)

    # Show stats for closed trades
    stats = store.get_live_trade_stats(symbol=symbol, days=days)
    if stats["total_trades"] > 0:
        console.print("\n[bold]Closed Trade Stats:[/bold]")
        console.print(f"  Total: {stats['total_trades']} trades")
        console.print(f"  Win Rate: {stats['win_rate']:.1f}%")
        total_pnl = stats["total_pnl"]
        if total_pnl >= 0:
            console.print(f"  Total P&L: [green]+${total_pnl:,.2f}[/green]")
        else:
            console.print(f"  Total P&L: [red]-${abs(total_pnl):,.2f}[/red]")

    console.print()


@app.command("notify-test")
def test_notification(
    webhook: str | None = typer.Option(
        None,
        "--webhook",
        "-w",
        help="Discord webhook URL (or use DISCORD_WEBHOOK_URL env)",
    ),
    message: str = typer.Option(
        "Test notification from Trader Bot", "--message", "-m", help="Test message"
    ),
) -> None:
    """Test Discord webhook notification."""
    asyncio.run(_test_notification(webhook, message))


async def _test_notification(webhook_url: str | None, message: str) -> None:
    """Send a test notification to Discord."""
    from trader.notifications import DiscordNotifier

    settings = get_settings()
    url = webhook_url or settings.discord_webhook_url

    if not url:
        console.print("[red]Error: No Discord webhook URL configured.[/red]")
        console.print("\nProvide via --webhook or set DISCORD_WEBHOOK_URL in .env")
        raise typer.Exit(1)

    notifier = DiscordNotifier(url)

    console.print("Sending test notification to Discord...")

    # Send a test message
    success = await notifier.send_message(f"🧪 {message}")

    if success:
        console.print("[green]Test notification sent successfully![/green]")
    else:
        console.print("[red]Failed to send notification. Check webhook URL.[/red]")

    # Also send a sample trade notification
    console.print("Sending sample trade notification...")

    from decimal import Decimal

    from trader.core.models import Signal, SignalAction

    sample_signal = Signal(
        action=SignalAction.BUY,
        symbol="AAPL",
        confidence=0.8,
        reason="SMA crossover - bullish signal",
    )

    success = await notifier.notify_trade_signal(
        signal=sample_signal,
        symbol="AAPL",
        quantity=10,
        price=Decimal("150.00"),
        approved=True,
    )

    if success:
        console.print("[green]Sample trade notification sent![/green]")
    else:
        console.print("[red]Failed to send trade notification.[/red]")

    await notifier.close()


@app.command("notify-summary")
def send_summary(
    webhook: str | None = typer.Option(
        None, "--webhook", "-w", help="Discord webhook URL"
    ),
) -> None:
    """Send daily summary notification to Discord."""
    asyncio.run(_send_summary_notification(webhook))


async def _send_summary_notification(webhook_url: str | None) -> None:
    """Send daily summary to Discord."""
    from trader.notifications import DiscordNotifier

    settings = get_settings()
    url = webhook_url or settings.discord_webhook_url

    if not url:
        console.print("[red]Error: No Discord webhook URL configured.[/red]")
        raise typer.Exit(1)

    store = get_trade_store(settings.database_path)
    stats = store.get_trade_stats()

    if stats["total_trades"] == 0:
        console.print("[yellow]No trades to summarize.[/yellow]")
        return

    notifier = DiscordNotifier(url)

    today = datetime.now().strftime("%Y-%m-%d")

    success = await notifier.notify_daily_summary(
        date=today,
        starting_equity=100000.0,  # Placeholder
        ending_equity=100000.0 + stats["total_pnl"],
        realized_pnl=stats["total_pnl"],
        num_trades=stats["total_trades"],
        winning_trades=stats["winning_trades"],
        losing_trades=stats["losing_trades"],
    )

    if success:
        console.print("[green]Daily summary sent to Discord![/green]")
    else:
        console.print("[red]Failed to send summary.[/red]")

    await notifier.close()


# =============================================================================
# Position Management Commands
# =============================================================================


@app.command("account")
def account_info() -> None:
    """Show account summary (equity, buying power, cash)."""
    asyncio.run(_show_account())


async def _show_account() -> None:
    """Display account information."""
    from trader.broker.alpaca import AlpacaBroker

    settings = get_settings()

    if not settings.has_alpaca_credentials:
        console.print("[red]Error: Alpaca API credentials required.[/red]")
        console.print("\nSet ALPACA_API_KEY and ALPACA_SECRET_KEY in .env")
        raise typer.Exit(1)

    broker = AlpacaBroker(
        api_key=settings.alpaca_api_key.get_secret_value(),
        secret_key=settings.alpaca_secret_key.get_secret_value(),
        paper=settings.alpaca_paper,
    )

    await broker.connect()

    equity = await broker.get_account_value()
    buying_power = await broker.get_buying_power()
    cash = await broker.get_cash()

    mode = "Paper" if broker.is_paper else "Live"

    console.print(f"\n[bold blue]Account Summary ({mode})[/bold blue]\n")

    table = Table(show_header=False, box=None)
    table.add_column("Label", style="dim")
    table.add_column("Value", style="bold")

    table.add_row("Equity", f"${float(equity):,.2f}")
    table.add_row("Buying Power", f"${float(buying_power):,.2f}")
    table.add_row("Cash", f"${float(cash):,.2f}")

    positions = await broker.get_positions()
    position_value = sum(float(p.market_value or 0) for p in positions)
    table.add_row("Positions Value", f"${position_value:,.2f}")
    table.add_row("Open Positions", str(len(positions)))

    console.print(table)

    await broker.disconnect()


@app.command("positions")
def list_positions() -> None:
    """List all open positions with P&L."""
    asyncio.run(_list_positions())


async def _list_positions() -> None:
    """Display open positions."""
    from trader.broker.alpaca import AlpacaBroker

    settings = get_settings()

    if not settings.has_alpaca_credentials:
        console.print("[red]Error: Alpaca API credentials required.[/red]")
        raise typer.Exit(1)

    broker = AlpacaBroker(
        api_key=settings.alpaca_api_key.get_secret_value(),
        secret_key=settings.alpaca_secret_key.get_secret_value(),
        paper=settings.alpaca_paper,
    )

    await broker.connect()

    positions = await broker.get_positions()

    if not positions:
        console.print("\n[yellow]No open positions.[/yellow]")
        await broker.disconnect()
        return

    mode = "Paper" if broker.is_paper else "Live"
    console.print(f"\n[bold blue]Open Positions ({mode})[/bold blue]\n")

    table = Table()
    table.add_column("Symbol", style="bold")
    table.add_column("Qty", justify="right")
    table.add_column("Avg Entry", justify="right")
    table.add_column("Current", justify="right")
    table.add_column("Market Value", justify="right")
    table.add_column("P&L", justify="right")
    table.add_column("P&L %", justify="right")

    total_value = 0.0
    total_pnl = 0.0

    for pos in positions:
        pnl = float(pos.unrealized_pnl or 0)
        pnl_pct = (pos.unrealized_pnl_pct or 0) * 100
        market_value = float(pos.market_value or 0)

        total_value += market_value
        total_pnl += pnl

        pnl_color = "green" if pnl >= 0 else "red"
        pnl_str = f"[{pnl_color}]${pnl:,.2f}[/{pnl_color}]"
        pnl_pct_str = f"[{pnl_color}]{pnl_pct:+.2f}%[/{pnl_color}]"

        table.add_row(
            pos.symbol,
            str(pos.quantity),
            f"${float(pos.avg_entry_price):,.2f}",
            f"${float(pos.current_price or 0):,.2f}",
            f"${market_value:,.2f}",
            pnl_str,
            pnl_pct_str,
        )

    console.print(table)

    # Summary
    pnl_color = "green" if total_pnl >= 0 else "red"
    console.print(f"\n[bold]Total Value:[/bold] ${total_value:,.2f}")
    console.print(
        f"[bold]Total P&L:[/bold] [{pnl_color}]${total_pnl:,.2f}[/{pnl_color}]"
    )

    await broker.disconnect()


@app.command("orders")
def list_orders() -> None:
    """List all open orders."""
    asyncio.run(_list_orders())


async def _list_orders() -> None:
    """Display open orders."""
    from trader.broker.alpaca import AlpacaBroker

    settings = get_settings()

    if not settings.has_alpaca_credentials:
        console.print("[red]Error: Alpaca API credentials required.[/red]")
        raise typer.Exit(1)

    broker = AlpacaBroker(
        api_key=settings.alpaca_api_key.get_secret_value(),
        secret_key=settings.alpaca_secret_key.get_secret_value(),
        paper=settings.alpaca_paper,
    )

    await broker.connect()

    orders = await broker.get_open_orders()

    if not orders:
        console.print("\n[yellow]No open orders.[/yellow]")
        await broker.disconnect()
        return

    mode = "Paper" if broker.is_paper else "Live"
    console.print(f"\n[bold blue]Open Orders ({mode})[/bold blue]\n")

    table = Table()
    table.add_column("Order ID", style="dim")
    table.add_column("Symbol", style="bold")
    table.add_column("Side")
    table.add_column("Qty", justify="right")
    table.add_column("Type")
    table.add_column("Limit", justify="right")
    table.add_column("Status")
    table.add_column("Filled", justify="right")

    for order in orders:
        side_color = "green" if order.side.value == "buy" else "red"
        side_str = f"[{side_color}]{order.side.value.upper()}[/{side_color}]"

        limit_str = f"${float(order.limit_price):,.2f}" if order.limit_price else "-"
        filled_str = f"{order.filled_quantity}/{order.quantity}"

        table.add_row(
            order.broker_order_id or order.order_id[:8],
            order.symbol,
            side_str,
            str(order.quantity),
            order.order_type.value,
            limit_str,
            order.status.value,
            filled_str,
        )

    console.print(table)

    await broker.disconnect()


@app.command("close")
def close_position(
    symbol: str = typer.Argument(
        ..., help="Symbol to close (or 'all' to close all positions)"
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation prompt"),
) -> None:
    """Close a position (or all positions with 'all')."""
    asyncio.run(_close_position(symbol, force))


async def _close_position(symbol: str, force: bool) -> None:
    """Close position(s)."""
    from trader.broker.alpaca import AlpacaBroker

    settings = get_settings()

    if not settings.has_alpaca_credentials:
        console.print("[red]Error: Alpaca API credentials required.[/red]")
        raise typer.Exit(1)

    broker = AlpacaBroker(
        api_key=settings.alpaca_api_key.get_secret_value(),
        secret_key=settings.alpaca_secret_key.get_secret_value(),
        paper=settings.alpaca_paper,
    )

    await broker.connect()

    mode = "Paper" if broker.is_paper else "LIVE"

    if symbol.lower() == "all":
        positions = await broker.get_positions()
        if not positions:
            console.print("[yellow]No positions to close.[/yellow]")
            await broker.disconnect()
            return

        console.print(f"\n[bold]Close ALL {len(positions)} positions? ({mode})[/bold]")
        for pos in positions:
            pnl = float(pos.unrealized_pnl or 0)
            pnl_color = "green" if pnl >= 0 else "red"
            console.print(
                f"  {pos.symbol}: {pos.quantity} shares "
                f"([{pnl_color}]${pnl:,.2f}[/{pnl_color}])"
            )

        if not force and not typer.confirm("\nProceed?"):
            console.print("Cancelled.")
            await broker.disconnect()
            return

        orders = await broker.close_all_positions()
        console.print(f"\n[green]Closed {len(orders)} positions.[/green]")

    else:
        symbol = symbol.upper()
        position = await broker.get_position(symbol)

        if not position:
            console.print(f"[yellow]No position found for {symbol}.[/yellow]")
            await broker.disconnect()
            return

        pnl = float(position.unrealized_pnl or 0)
        pnl_color = "green" if pnl >= 0 else "red"

        console.print(f"\n[bold]Close {symbol} position? ({mode})[/bold]")
        console.print(f"  Quantity: {position.quantity} shares")
        console.print(f"  Entry: ${float(position.avg_entry_price):,.2f}")
        console.print(f"  Current: ${float(position.current_price or 0):,.2f}")
        console.print(f"  P&L: [{pnl_color}]${pnl:,.2f}[/{pnl_color}]")

        if not force and not typer.confirm("\nProceed?"):
            console.print("Cancelled.")
            await broker.disconnect()
            return

        order = await broker.close_position(symbol)
        if order:
            console.print(f"\n[green]Closed {symbol} position.[/green]")
        else:
            console.print(f"\n[red]Failed to close {symbol} position.[/red]")

    await broker.disconnect()


@app.command("cancel")
def cancel_order(
    order_id: str = typer.Argument(
        ..., help="Order ID to cancel (or 'all' to cancel all orders)"
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation prompt"),
) -> None:
    """Cancel an order (or all orders with 'all')."""
    asyncio.run(_cancel_order(order_id, force))


async def _cancel_order(order_id: str, force: bool) -> None:
    """Cancel order(s)."""
    from trader.broker.alpaca import AlpacaBroker

    settings = get_settings()

    if not settings.has_alpaca_credentials:
        console.print("[red]Error: Alpaca API credentials required.[/red]")
        raise typer.Exit(1)

    broker = AlpacaBroker(
        api_key=settings.alpaca_api_key.get_secret_value(),
        secret_key=settings.alpaca_secret_key.get_secret_value(),
        paper=settings.alpaca_paper,
    )

    await broker.connect()

    mode = "Paper" if broker.is_paper else "LIVE"

    if order_id.lower() == "all":
        orders = await broker.get_open_orders()
        if not orders:
            console.print("[yellow]No open orders to cancel.[/yellow]")
            await broker.disconnect()
            return

        console.print(f"\n[bold]Cancel ALL {len(orders)} orders? ({mode})[/bold]")
        for order in orders:
            console.print(
                f"  {order.symbol}: {order.side.value.upper()} {order.quantity} "
                f"({order.order_type.value})"
            )

        if not force and not typer.confirm("\nProceed?"):
            console.print("Cancelled.")
            await broker.disconnect()
            return

        cancelled = 0
        for order in orders:
            if await broker.cancel_order(order.broker_order_id or order.order_id):
                cancelled += 1

        console.print(f"\n[green]Cancelled {cancelled} orders.[/green]")

    else:
        order = await broker.get_order(order_id)

        if not order:
            console.print(f"[yellow]Order {order_id} not found.[/yellow]")
            await broker.disconnect()
            return

        console.print(f"\n[bold]Cancel order? ({mode})[/bold]")
        console.print(f"  Symbol: {order.symbol}")
        console.print(f"  Side: {order.side.value.upper()}")
        console.print(f"  Quantity: {order.quantity}")
        console.print(f"  Type: {order.order_type.value}")
        console.print(f"  Status: {order.status.value}")

        if not force and not typer.confirm("\nProceed?"):
            console.print("Cancelled.")
            await broker.disconnect()
            return

        if await broker.cancel_order(order_id):
            console.print(f"\n[green]Order {order_id} cancelled.[/green]")
        else:
            console.print(f"\n[red]Failed to cancel order {order_id}.[/red]")

    await broker.disconnect()


# =============================================================================
# Portfolio Rebalancing Commands
# =============================================================================


@app.command("allocations")
def show_allocations() -> None:
    """Show current portfolio allocations."""
    asyncio.run(_show_allocations())


async def _show_allocations() -> None:
    """Display current portfolio allocations."""
    from trader.broker.alpaca import AlpacaBroker

    settings = get_settings()

    if not settings.has_alpaca_credentials:
        console.print("[red]Error: Alpaca API credentials required.[/red]")
        raise typer.Exit(1)

    broker = AlpacaBroker(
        api_key=settings.alpaca_api_key.get_secret_value(),
        secret_key=settings.alpaca_secret_key.get_secret_value(),
        paper=settings.alpaca_paper,
    )

    await broker.connect()

    positions = await broker.get_positions()
    account_value = await broker.get_account_value()
    cash = await broker.get_cash()

    mode = "Paper" if broker.is_paper else "Live"
    console.print(f"\n[bold blue]Portfolio Allocations ({mode})[/bold blue]\n")

    if not positions:
        console.print("[yellow]No positions. Portfolio is 100% cash.[/yellow]")
        console.print(f"Cash: ${float(cash):,.2f}")
        await broker.disconnect()
        return

    table = Table()
    table.add_column("Symbol", style="bold")
    table.add_column("Value", justify="right")
    table.add_column("Allocation", justify="right")
    table.add_column("Shares", justify="right")

    total_position_value = 0.0
    for pos in positions:
        market_value = float(pos.market_value or 0)
        total_position_value += market_value
        allocation = market_value / float(account_value) * 100

        table.add_row(
            pos.symbol,
            f"${market_value:,.2f}",
            f"{allocation:.1f}%",
            str(pos.quantity),
        )

    # Add cash row
    cash_pct = float(cash) / float(account_value) * 100
    table.add_row(
        "[dim]CASH[/dim]",
        f"[dim]${float(cash):,.2f}[/dim]",
        f"[dim]{cash_pct:.1f}%[/dim]",
        "[dim]-[/dim]",
    )

    console.print(table)
    console.print(f"\n[bold]Total Equity:[/bold] ${float(account_value):,.2f}")

    await broker.disconnect()


@app.command("rebalance")
def rebalance_portfolio(
    targets: str = typer.Argument(
        ...,
        help="Target allocations as SYMBOL:PCT,... (e.g., AAPL:30,MSFT:30,GOOGL:40)",
    ),
    drift: float = typer.Option(
        5.0,
        "--drift",
        "-d",
        help="Drift threshold to trigger rebalance (0.05 = 5%)",
    ),
    min_trade: float = typer.Option(
        100.0, "--min-trade", help="Minimum trade value in dollars"
    ),
    dry_run: bool = typer.Option(
        True, "--dry-run/--execute", help="Preview orders without executing"
    ),
) -> None:
    """Rebalance portfolio to target allocations.

    Example: trader rebalance AAPL:30,MSFT:30,GOOGL:40
    """
    asyncio.run(
        _run_rebalance(
            targets_str=targets,
            drift_threshold=drift / 100,
            min_trade_value=min_trade,
            dry_run=dry_run,
        )
    )


async def _run_rebalance(
    targets_str: str,
    drift_threshold: float,
    min_trade_value: float,
    dry_run: bool,
) -> None:
    """Execute portfolio rebalance."""
    from trader.broker.alpaca import AlpacaBroker
    from trader.portfolio.rebalance import (
        PortfolioAllocation,
        RebalanceAction,
        RebalanceConfig,
        RebalanceEngine,
    )

    settings = get_settings()

    if not settings.has_alpaca_credentials:
        console.print("[red]Error: Alpaca API credentials required.[/red]")
        raise typer.Exit(1)

    # Parse targets
    allocations = []
    for item in targets_str.split(","):
        parts = item.strip().split(":")
        if len(parts) != 2:
            console.print(f"[red]Invalid target format: {item}[/red]")
            console.print("[dim]Use SYMBOL:PCT format (e.g., AAPL:30)[/dim]")
            raise typer.Exit(1)

        symbol = parts[0].upper()
        try:
            pct = float(parts[1]) / 100
        except ValueError:
            console.print(f"[red]Invalid percentage: {parts[1]}[/red]")
            raise typer.Exit(1) from None

        allocations.append(PortfolioAllocation(symbol=symbol, target_pct=pct))

    # Validate allocations sum to 100%
    total = sum(a.target_pct for a in allocations)
    if abs(total - 1.0) > 0.01:
        console.print(
            f"[red]Allocations must sum to 100%, got {total * 100:.1f}%[/red]"
        )
        raise typer.Exit(1)

    config = RebalanceConfig(
        allocations=allocations,
        drift_threshold=drift_threshold,
        min_trade_value=min_trade_value,
        dry_run=dry_run,
    )

    broker = AlpacaBroker(
        api_key=settings.alpaca_api_key.get_secret_value(),
        secret_key=settings.alpaca_secret_key.get_secret_value(),
        paper=settings.alpaca_paper,
    )

    await broker.connect()

    mode = "Paper" if broker.is_paper else "LIVE"
    console.print(f"\n[bold blue]Portfolio Rebalance ({mode})[/bold blue]\n")

    # Display target allocations
    console.print("[bold]Target Allocations:[/bold]")
    for alloc in config.allocations:
        console.print(f"  {alloc.symbol}: {alloc.target_pct * 100:.1f}%")
    console.print(f"\nDrift threshold: {drift_threshold * 100:.1f}%")
    console.print(f"Min trade value: ${min_trade_value:,.0f}")

    engine = RebalanceEngine(broker, config)
    result = await engine.calculate_rebalance()

    if not result.needs_rebalance:
        console.print("\n[green]Portfolio is already balanced![/green]")
        await broker.disconnect()
        return

    # Display orders
    console.print("\n[bold]Rebalance Orders:[/bold]\n")

    table = Table()
    table.add_column("Symbol", style="bold")
    table.add_column("Action")
    table.add_column("Shares", justify="right")
    table.add_column("Value", justify="right")
    table.add_column("Current", justify="right")
    table.add_column("Target", justify="right")
    table.add_column("Drift", justify="right")

    for order in result.orders:
        if order.action == RebalanceAction.HOLD:
            action_str = "[dim]HOLD[/dim]"
        elif order.action == RebalanceAction.BUY:
            action_str = "[green]BUY[/green]"
        else:
            action_str = "[red]SELL[/red]"

        drift_color = "red" if abs(order.drift_pct) > drift_threshold else "dim"

        table.add_row(
            order.symbol,
            action_str,
            str(order.quantity) if order.quantity > 0 else "-",
            f"${order.trade_value:,.0f}" if order.trade_value > 0 else "-",
            f"{order.current_pct * 100:.1f}%",
            f"{order.target_pct * 100:.1f}%",
            f"[{drift_color}]{order.drift_pct * 100:+.1f}%[/{drift_color}]",
        )

    console.print(table)
    console.print("\n[bold]Summary:[/bold]")
    console.print(f"  Total buys: ${result.total_buys:,.2f}")
    console.print(f"  Total sells: ${result.total_sells:,.2f}")
    console.print(f"  Net cash change: ${result.net_cash_change:,.2f}")

    if dry_run:
        console.print("\n[yellow]Dry run mode - no orders executed.[/yellow]")
        console.print("[dim]Use --execute to place orders.[/dim]")
    else:
        if not typer.confirm("\nExecute these orders?"):
            console.print("Cancelled.")
            await broker.disconnect()
            return

        result = await engine.execute_rebalance(result)

        if result.executed:
            console.print("\n[green]Rebalance executed successfully![/green]")
        else:
            console.print("\n[red]Rebalance failed:[/red]")
            for error in result.errors:
                console.print(f"  - {error}")

    await broker.disconnect()


@app.command("rebalance-equal")
def rebalance_equal(
    symbols: str = typer.Argument(
        ..., help="Comma-separated symbols (e.g., AAPL,MSFT,GOOGL)"
    ),
    drift: float = typer.Option(5.0, "--drift", "-d", help="Drift threshold (%)"),
    dry_run: bool = typer.Option(
        True, "--dry-run/--execute", help="Preview without executing"
    ),
) -> None:
    """Rebalance to equal weight allocation.

    Example: trader rebalance-equal AAPL,MSFT,GOOGL
    """
    symbol_list = [s.strip().upper() for s in symbols.split(",")]
    weight = 100 / len(symbol_list)
    targets = ",".join(f"{s}:{weight:.2f}" for s in symbol_list)

    asyncio.run(
        _run_rebalance(
            targets_str=targets,
            drift_threshold=drift / 100,
            min_trade_value=100.0,
            dry_run=dry_run,
        )
    )


# =============================================================================
# Dashboard Command
# =============================================================================


# =============================================================================
# Risk Analytics Commands
# =============================================================================


@app.command("risk")
def risk_analysis(
    symbols: str = typer.Argument(
        "mag7", help="Comma-separated symbols or group name (mag7, faang, tech)"
    ),
    days: int = typer.Option(365, "--days", "-d", help="Days of history for analysis"),
    benchmark: str = typer.Option("SPY", "--benchmark", "-b", help="Benchmark symbol"),
    confidence: float = typer.Option(
        95.0, "--confidence", "-c", help="VaR confidence level (e.g., 95 or 99)"
    ),
) -> None:
    """
    Calculate portfolio risk metrics (VaR, volatility, Sharpe, beta).

    Examples:
      trader risk mag7 --days 365
      trader risk AAPL,MSFT,GOOGL --benchmark SPY
      trader risk tech --confidence 99
    """
    # Parse symbols
    if symbols.lower() in STOCK_GROUPS:
        symbol_list = STOCK_GROUPS[symbols.lower()]
        console.print(f"\n[bold blue]Risk Analysis: {symbols.upper()}[/bold blue]")
    else:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        console.print(f"\n[bold blue]Risk Analysis: {len(symbol_list)} Stocks[/bold blue]")

    asyncio.run(
        _run_risk_analysis(
            symbols=symbol_list,
            days=days,
            benchmark=benchmark,
            confidence_level=confidence / 100,
        )
    )


async def _run_risk_analysis(
    symbols: list[str],
    days: int,
    benchmark: str,
    confidence_level: float,
) -> None:
    """Run risk analysis for given symbols."""
    import pandas as pd

    from trader.risk.analytics import PortfolioAnalytics

    settings = get_settings()
    fetcher = get_data_fetcher(settings)
    analytics = PortfolioAnalytics()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    console.print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    console.print(f"Benchmark: {benchmark}")
    console.print(f"VaR Confidence: {confidence_level * 100:.0f}%\n")

    # Fetch benchmark data
    benchmark_data = None
    benchmark_returns = None
    with console.status("[bold green]Fetching benchmark data..."):
        try:
            benchmark_df = await fetcher.fetch_bars_df(
                symbol=benchmark,
                timeframe=TimeFrame.DAY,
                start=start_date,
                end=end_date,
            )
            if len(benchmark_df) > 1:
                benchmark_data = benchmark_df["close"]
                benchmark_returns = analytics.calculate_returns(benchmark_data, method="simple")
        except Exception as e:
            logger.warning(f"Could not fetch benchmark {benchmark}: {e}")

    # Fetch data for all symbols
    price_data: dict[str, pd.Series] = {}
    returns_data: dict[str, pd.Series] = {}
    results: list[dict] = []

    with console.status("[bold green]Fetching price data...") as status:
        for symbol in symbols:
            status.update(f"[bold green]Fetching {symbol}...")
            try:
                data = await fetcher.fetch_bars_df(
                    symbol=symbol,
                    timeframe=TimeFrame.DAY,
                    start=start_date,
                    end=end_date,
                )

                if len(data) < 20:
                    continue

                price_data[symbol] = data["close"]
                returns = analytics.calculate_returns(data["close"], method="simple")
                returns_data[symbol] = returns

                # Calculate individual metrics
                var = analytics.calculate_var(returns, confidence_level)
                vol = analytics.calculate_volatility(returns)
                sharpe = analytics.calculate_sharpe_ratio(returns)
                sortino = analytics.calculate_sortino_ratio(returns)
                max_dd, _, _ = analytics.calculate_max_drawdown(data["close"])

                beta = None
                alpha = None
                if benchmark_returns is not None:
                    beta = analytics.calculate_beta(returns, benchmark_returns)
                    alpha = analytics.calculate_alpha(returns, benchmark_returns)

                total_return = (data["close"].iloc[-1] - data["close"].iloc[0]) / data["close"].iloc[0]

                results.append({
                    "symbol": symbol,
                    "return": total_return * 100,
                    "volatility": vol * 100,
                    "var": var * 100,
                    "sharpe": sharpe,
                    "sortino": sortino,
                    "max_dd": max_dd * 100,
                    "beta": beta,
                    "alpha": alpha * 100 if alpha else None,
                })

            except Exception as e:
                logger.debug(f"Error fetching {symbol}: {e}")
                continue

    if not results:
        console.print("[yellow]No data available for analysis.[/yellow]")
        return

    # Display individual stock metrics
    console.print("[bold]Individual Stock Risk Metrics:[/bold]\n")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Symbol", style="bold")
    table.add_column("Return", justify="right")
    table.add_column("Volatility", justify="right")
    table.add_column(f"VaR {confidence_level*100:.0f}%", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("Sortino", justify="right")
    table.add_column("Max DD", justify="right")
    if benchmark_returns is not None:
        table.add_column("Beta", justify="right")
        table.add_column("Alpha", justify="right")

    for r in sorted(results, key=lambda x: -x["sharpe"]):
        ret_color = "green" if r["return"] >= 0 else "red"
        ret_str = f"[{ret_color}]{r['return']:+.1f}%[/{ret_color}]"

        sharpe_color = "green" if r["sharpe"] >= 1 else "yellow" if r["sharpe"] >= 0 else "red"
        sharpe_str = f"[{sharpe_color}]{r['sharpe']:.2f}[/{sharpe_color}]"

        sortino_color = "green" if r["sortino"] >= 1.5 else "yellow" if r["sortino"] >= 0 else "red"
        sortino_str = f"[{sortino_color}]{r['sortino']:.2f}[/{sortino_color}]"

        row = [
            r["symbol"],
            ret_str,
            f"{r['volatility']:.1f}%",
            f"{r['var']:.1f}%",
            sharpe_str,
            sortino_str,
            f"[red]{r['max_dd']:.1f}%[/red]",
        ]

        if benchmark_returns is not None:
            beta_str = f"{r['beta']:.2f}" if r["beta"] else "-"
            alpha_str = f"{r['alpha']:+.1f}%" if r["alpha"] else "-"
            row.extend([beta_str, alpha_str])

        table.add_row(*row)

    console.print(table)

    # Calculate portfolio metrics (equal weight)
    if len(results) > 1:
        console.print("\n[bold]Portfolio Metrics (Equal Weight):[/bold]\n")

        weights = {s: 1.0 / len(returns_data) for s in returns_data}
        portfolio_var = analytics.calculate_portfolio_var(weights, returns_data, confidence_level)

        # Calculate correlation matrix
        corr_result = analytics.calculate_correlation_matrix(price_data)

        # Build portfolio returns
        portfolio_returns = pd.DataFrame(returns_data).mean(axis=1)
        port_vol = analytics.calculate_volatility(portfolio_returns)
        port_sharpe = analytics.calculate_sharpe_ratio(portfolio_returns)
        port_sortino = analytics.calculate_sortino_ratio(portfolio_returns)

        # Portfolio total return
        portfolio_prices = pd.DataFrame(price_data)
        portfolio_value = portfolio_prices.mean(axis=1)
        port_return = (portfolio_value.iloc[-1] - portfolio_value.iloc[0]) / portfolio_value.iloc[0]
        port_max_dd, _, _ = analytics.calculate_max_drawdown(portfolio_value)

        port_beta = None
        if benchmark_returns is not None:
            port_beta = analytics.calculate_beta(portfolio_returns, benchmark_returns)

        stats_table = Table(show_header=False, box=None)
        stats_table.add_column("Metric", style="dim")
        stats_table.add_column("Value", style="bold")

        ret_color = "green" if port_return >= 0 else "red"
        stats_table.add_row("Total Return", f"[{ret_color}]{port_return * 100:+.1f}%[/{ret_color}]")
        stats_table.add_row("Volatility (Ann.)", f"{port_vol * 100:.1f}%")
        stats_table.add_row(f"VaR {confidence_level*100:.0f}%", f"{portfolio_var * 100:.2f}%")
        stats_table.add_row("Sharpe Ratio", f"{port_sharpe:.2f}")
        stats_table.add_row("Sortino Ratio", f"{port_sortino:.2f}")
        stats_table.add_row("Max Drawdown", f"[red]{port_max_dd * 100:.1f}%[/red]")
        if port_beta is not None:
            stats_table.add_row("Beta vs " + benchmark, f"{port_beta:.2f}")

        console.print(stats_table)

        # Show correlation matrix if requested
        if len(corr_result.symbols) <= 10:
            console.print("\n[bold]Correlation Matrix:[/bold]\n")

            corr_table = Table(show_header=True, header_style="bold")
            corr_table.add_column("")
            for sym in corr_result.symbols:
                corr_table.add_column(sym, justify="center")

            for sym in corr_result.symbols:
                row = [f"[bold]{sym}[/bold]"]
                for other in corr_result.symbols:
                    corr = corr_result.matrix.loc[sym, other]
                    if sym == other:
                        row.append("[dim]1.00[/dim]")
                    elif corr >= 0.7:
                        row.append(f"[red]{corr:.2f}[/red]")
                    elif corr >= 0.4:
                        row.append(f"[yellow]{corr:.2f}[/yellow]")
                    else:
                        row.append(f"[green]{corr:.2f}[/green]")
                corr_table.add_row(*row)

            console.print(corr_table)
            console.print("\n[dim]Correlation: [green]< 0.4[/green] low | [yellow]0.4-0.7[/yellow] moderate | [red]> 0.7[/red] high[/dim]")

    console.print()


@app.command("var")
def value_at_risk(
    symbol: str = typer.Argument(..., help="Symbol to analyze"),
    days: int = typer.Option(365, "--days", "-d", help="Days of history"),
    confidence: float = typer.Option(95.0, "--confidence", "-c", help="Confidence level"),
    amount: float = typer.Option(10000.0, "--amount", "-a", help="Investment amount ($)"),
) -> None:
    """
    Calculate Value at Risk for a single position.

    Shows potential loss at specified confidence level.

    Example: trader var AAPL --amount 50000 --confidence 99
    """
    asyncio.run(_calculate_var(symbol, days, confidence / 100, amount))


async def _calculate_var(
    symbol: str,
    days: int,
    confidence_level: float,
    amount: float,
) -> None:
    """Calculate VaR for a symbol."""
    from trader.risk.analytics import PortfolioAnalytics

    settings = get_settings()
    fetcher = get_data_fetcher(settings)
    analytics = PortfolioAnalytics()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    symbol = symbol.upper()
    console.print(f"\n[bold blue]Value at Risk: {symbol}[/bold blue]\n")

    with console.status(f"[bold green]Fetching {symbol} data..."):
        try:
            data = await fetcher.fetch_bars_df(
                symbol=symbol,
                timeframe=TimeFrame.DAY,
                start=start_date,
                end=end_date,
            )
        except Exception as e:
            console.print(f"[red]Error fetching data: {e}[/red]")
            return

    if len(data) < 20:
        console.print("[yellow]Insufficient data for VaR calculation.[/yellow]")
        return

    returns = analytics.calculate_returns(data["close"], method="simple")

    # Calculate various VaR metrics
    var_hist = analytics.calculate_var(returns, confidence_level, method="historical")
    var_param = analytics.calculate_var(returns, confidence_level, method="parametric")
    cvar = analytics.calculate_cvar(returns, confidence_level)
    vol = analytics.calculate_volatility(returns)

    # Dollar amounts
    var_hist_dollar = amount * var_hist
    var_param_dollar = amount * var_param
    cvar_dollar = amount * cvar

    console.print(f"Investment Amount: ${amount:,.2f}")
    console.print(f"Confidence Level: {confidence_level * 100:.0f}%")
    console.print(f"Analysis Period: {len(returns)} trading days")
    console.print(f"Annualized Volatility: {vol * 100:.1f}%\n")

    table = Table(title=f"{symbol} Value at Risk", show_header=True, header_style="bold cyan")
    table.add_column("Method", style="bold")
    table.add_column("Daily VaR %", justify="right")
    table.add_column("Daily VaR $", justify="right")
    table.add_column("Description")

    table.add_row(
        "Historical VaR",
        f"[red]{var_hist * 100:.2f}%[/red]",
        f"[red]${var_hist_dollar:,.2f}[/red]",
        f"{confidence_level*100:.0f}% of days, loss won't exceed this",
    )

    table.add_row(
        "Parametric VaR",
        f"[red]{var_param * 100:.2f}%[/red]",
        f"[red]${var_param_dollar:,.2f}[/red]",
        "Assumes normal distribution",
    )

    table.add_row(
        "CVaR (Expected Shortfall)",
        f"[red]{cvar * 100:.2f}%[/red]",
        f"[red]${cvar_dollar:,.2f}[/red]",
        "Avg loss when exceeding VaR",
    )

    console.print(table)

    # Additional context
    console.print("\n[bold]Interpretation:[/bold]")
    console.print(
        f"  With {confidence_level*100:.0f}% confidence, your daily loss on ${amount:,.0f} "
        f"won't exceed [red]${var_hist_dollar:,.2f}[/red]"
    )
    console.print(
        f"  In the worst {(1-confidence_level)*100:.0f}% of days, expect to lose "
        f"around [red]${cvar_dollar:,.2f}[/red] on average"
    )
    console.print()


@app.command("correlation")
def correlation_matrix(
    symbols: str = typer.Argument(
        "mag7", help="Comma-separated symbols or group name"
    ),
    days: int = typer.Option(180, "--days", "-d", help="Days of history"),
) -> None:
    """
    Show correlation matrix between assets.

    Lower correlation = better diversification.

    Example: trader correlation AAPL,MSFT,GOOGL,AMZN
    """
    # Parse symbols
    if symbols.lower() in STOCK_GROUPS:
        symbol_list = STOCK_GROUPS[symbols.lower()]
    else:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]

    asyncio.run(_show_correlation(symbol_list, days))


async def _show_correlation(symbols: list[str], days: int) -> None:
    """Display correlation matrix."""
    import pandas as pd

    from trader.risk.analytics import PortfolioAnalytics

    settings = get_settings()
    fetcher = get_data_fetcher(settings)
    analytics = PortfolioAnalytics()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    console.print("\n[bold blue]Correlation Matrix[/bold blue]")
    console.print(f"Period: {days} days\n")

    # Fetch data
    price_data: dict[str, pd.Series] = {}

    with console.status("[bold green]Fetching price data...") as status:
        for symbol in symbols:
            status.update(f"[bold green]Fetching {symbol}...")
            try:
                data = await fetcher.fetch_bars_df(
                    symbol=symbol,
                    timeframe=TimeFrame.DAY,
                    start=start_date,
                    end=end_date,
                )
                if len(data) > 20:
                    price_data[symbol] = data["close"]
            except Exception as e:
                logger.debug(f"Error fetching {symbol}: {e}")

    if len(price_data) < 2:
        console.print("[yellow]Need at least 2 symbols for correlation.[/yellow]")
        return

    corr_result = analytics.calculate_correlation_matrix(price_data)

    # Display matrix
    table = Table(show_header=True, header_style="bold")
    table.add_column("")
    for sym in corr_result.symbols:
        table.add_column(sym, justify="center")

    for sym in corr_result.symbols:
        row = [f"[bold]{sym}[/bold]"]
        for other in corr_result.symbols:
            corr = corr_result.matrix.loc[sym, other]
            if sym == other:
                row.append("[dim]1.00[/dim]")
            elif corr >= 0.8:
                row.append(f"[red bold]{corr:.2f}[/red bold]")
            elif corr >= 0.6:
                row.append(f"[red]{corr:.2f}[/red]")
            elif corr >= 0.4:
                row.append(f"[yellow]{corr:.2f}[/yellow]")
            elif corr >= 0.2:
                row.append(f"[green]{corr:.2f}[/green]")
            else:
                row.append(f"[green bold]{corr:.2f}[/green bold]")
        table.add_row(*row)

    console.print(table)

    # Summary stats
    import numpy as np
    avg_corr = corr_result.matrix.values[~np.eye(len(corr_result.symbols), dtype=bool)].mean()

    console.print("\n[bold]Diversification Analysis:[/bold]")
    console.print(f"  Average Correlation: {avg_corr:.2f}")

    if avg_corr >= 0.7:
        console.print("  [red]⚠ High correlation - limited diversification benefit[/red]")
    elif avg_corr >= 0.4:
        console.print("  [yellow]Moderate correlation - some diversification[/yellow]")
    else:
        console.print("  [green]✓ Low correlation - good diversification[/green]")

    console.print("\n[dim]Legend: [green bold]< 0.2[/green bold] very low | [green]0.2-0.4[/green] low | [yellow]0.4-0.6[/yellow] moderate | [red]0.6-0.8[/red] high | [red bold]> 0.8[/red bold] very high[/dim]")
    console.print()


@app.command("dashboard")
def dashboard(
    refresh: float = typer.Option(
        2.0, "--refresh", "-r", help="Refresh rate in seconds"
    ),
) -> None:
    """Launch real-time trading dashboard with live updates."""
    asyncio.run(_run_dashboard(refresh_rate=refresh))


async def _run_dashboard(refresh_rate: float) -> None:
    """Run the trading dashboard."""
    from trader.broker.alpaca import AlpacaBroker
    from trader.dashboard.live import TradingDashboard

    settings = get_settings()

    if not settings.has_alpaca_credentials:
        console.print("[red]Error: Alpaca API credentials required.[/red]")
        console.print("\nSet ALPACA_API_KEY and ALPACA_SECRET_KEY in .env")
        raise typer.Exit(1)

    broker = AlpacaBroker(
        api_key=settings.alpaca_api_key.get_secret_value(),
        secret_key=settings.alpaca_secret_key.get_secret_value(),
        paper=settings.alpaca_paper,
    )

    await broker.connect()

    mode = "Paper" if broker.is_paper else "LIVE"
    console.print(f"\n[bold blue]Starting Trading Dashboard ({mode})[/bold blue]")
    console.print(f"Refresh rate: {refresh_rate}s")
    console.print("\nPress Q to quit, R to refresh, C to close all positions\n")

    try:
        dashboard = TradingDashboard(broker, refresh_rate=refresh_rate)
        await dashboard.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard stopped.[/yellow]")
    finally:
        await broker.disconnect()


if __name__ == "__main__":
    app()
