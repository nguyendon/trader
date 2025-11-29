#!/usr/bin/env python
"""
Run a backtest with the SMA crossover strategy.

This script can be run directly without installing the package:
    python scripts/run_backtest.py AAPL --days 365

Or after installing:
    trader backtest AAPL --days 365
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trader.config.settings import get_settings
from trader.core.models import TimeFrame
from trader.data.fetcher import get_data_fetcher
from trader.engine.backtest import BacktestEngine
from trader.strategies.builtin.sma_crossover import SMACrossover


async def main(
    symbol: str = "AAPL",
    days: int = 365,
    fast_period: int = 10,
    slow_period: int = 50,
    initial_capital: float = 100_000.0,
) -> None:
    """Run backtest with given parameters."""
    print(f"\n{'='*50}")
    print("Trader Backtest")
    print(f"{'='*50}")
    print(f"Symbol: {symbol}")
    print(f"Strategy: SMA Crossover ({fast_period}/{slow_period})")
    print(f"Period: {days} days")
    print(f"Initial Capital: ${initial_capital:,.2f}")

    # Check for credentials
    settings = get_settings()
    if not settings.has_alpaca_credentials:
        print("\nNote: No Alpaca credentials found. Using mock data.")
        print("Set ALPACA_API_KEY and ALPACA_SECRET_KEY for real data.")

    # Get data fetcher
    fetcher = get_data_fetcher(settings)

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    print(f"\nFetching data from {start_date.date()} to {end_date.date()}...")

    # Fetch data
    data = await fetcher.fetch_bars_df(
        symbol=symbol,
        timeframe=TimeFrame.DAY,
        start=start_date,
        end=end_date,
    )

    if len(data) == 0:
        print("Error: No data returned")
        return

    print(f"Fetched {len(data)} bars")

    # Create strategy
    strategy = SMACrossover(fast_period=fast_period, slow_period=slow_period)

    # Run backtest
    engine = BacktestEngine(initial_capital=initial_capital)

    print("Running backtest...")
    result = await engine.run(
        strategy=strategy,
        data=data,
        symbol=symbol,
    )

    # Print results
    result.print_summary()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a backtest")
    parser.add_argument("symbol", nargs="?", default="AAPL", help="Stock symbol")
    parser.add_argument("--days", "-d", type=int, default=365, help="Days of history")
    parser.add_argument("--fast", "-f", type=int, default=10, help="Fast SMA period")
    parser.add_argument("--slow", "-s", type=int, default=50, help="Slow SMA period")
    parser.add_argument(
        "--capital", "-c", type=float, default=100000, help="Initial capital"
    )

    args = parser.parse_args()

    asyncio.run(
        main(
            symbol=args.symbol,
            days=args.days,
            fast_period=args.fast,
            slow_period=args.slow,
            initial_capital=args.capital,
        )
    )
