# Trader

Automated stock trading platform with backtesting, paper trading, and live trading support.

## Features

- **Backtesting Engine**: Test strategies on historical data with performance metrics
- **Mock Data**: Run backtests without API credentials using realistic simulated data
- **Strategy Framework**: Easily create and test custom trading strategies
- **CLI Interface**: Command-line tools for running backtests

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run backtest with mock data (no API key needed)
python scripts/run_backtest.py AAPL --days 365

# Or use the CLI
trader backtest AAPL --days 365 --fast 10 --slow 50
```

## Configuration

Copy `.env.example` to `.env` and add your Alpaca API keys for real market data:

```bash
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
ALPACA_PAPER=true
```

## Running Tests

```bash
pytest
pytest --cov=src/trader
```

## Project Structure

```
trader/
├── src/trader/
│   ├── config/       # Settings and configuration
│   ├── core/         # Domain models (Bar, Signal, Order)
│   ├── data/         # Data fetching (Alpaca + Mock)
│   ├── strategies/   # Trading strategies
│   └── engine/       # Backtesting engine
├── scripts/          # CLI scripts
└── tests/            # Test suite
```

## License

MIT
