# Trader

Automated stock trading platform for backtesting strategies, paper trading, and live trading with Alpaca.

## What This Does

1. **Backtest** - Test trading strategies on historical data to see how they would have performed
2. **Paper Trade** - Run strategies in real-time with fake money to validate them
3. **Live Trade** - Execute strategies with real money via Alpaca (coming soon)

## Quick Start

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone <repo>
cd trader
uv sync

# Run a backtest (no API key needed - uses mock data)
just backtest AAPL

# Run all checks
just check
```

## Commands

```bash
just                    # List all commands
just install            # Install dependencies
just check              # Run lint + typecheck + tests
just test               # Run tests only
just backtest AAPL      # Backtest SMA strategy on AAPL
just backtest AAPL --days 365 --fast 10 --slow 50
just paper AAPL,MSFT    # Paper trade demo
```

## What is Backtesting?

Backtesting runs a trading strategy on historical data to see how it would have performed. For example:

```bash
just backtest AAPL --days 365
```

This runs the SMA (Simple Moving Average) crossover strategy on AAPL for the past year and shows:

| Metric | What it means |
|--------|---------------|
| **Total Return** | How much money you made/lost (%) |
| **Sharpe Ratio** | Risk-adjusted return. >1 is good, >2 is great |
| **Max Drawdown** | Biggest drop from peak. <20% is acceptable |
| **Win Rate** | % of trades that made money |
| **Profit Factor** | Gross profits / gross losses. >1.5 is good |

### Example Output

```
Backtest Results
━━━━━━━━━━━━━━━━━━━━━━━━
Strategy:        SMA Crossover (10/50)
Symbol:          AAPL
Period:          2024-01-01 to 2024-12-01

Initial Capital: $100,000.00
Final Capital:   $112,450.00
Total Return:    +12.45%

Total Trades:    8
Win Rate:        62.5%
Profit Factor:   2.15
Max Drawdown:    8.3%
Sharpe Ratio:    1.42
```

### What to Look For

- **Sharpe Ratio > 1**: Strategy is worth the risk
- **Max Drawdown < 20%**: You won't lose too much during bad times
- **Profit Factor > 1.5**: Winners outweigh losers
- **Enough Trades**: At least 20+ trades to be statistically meaningful

### Warning Signs

- **Very high returns (>100%/year)**: Probably overfitting or bug
- **Only a few trades**: Not enough data to trust results
- **High drawdown (>30%)**: Too risky, you might panic sell

## How the SMA Strategy Works

The included SMA Crossover strategy:

1. Calculates a **fast** moving average (default: 10 days)
2. Calculates a **slow** moving average (default: 50 days)
3. **BUY** when fast crosses above slow (momentum turning up)
4. **SELL** when fast crosses below slow (momentum turning down)

```
Price
  │    ╱╲
  │   ╱  ╲_____ Fast MA (10-day)
  │  ╱        ╲
  │ ╱          ╲____
  │╱                ╲
  │──────────────────── Slow MA (50-day)
  │        ↑
  │      BUY signal (fast crosses above slow)
```

## Creating Your Own Strategy

Strategies inherit from `BaseStrategy`:

```python
# src/trader/strategies/builtin/my_strategy.py
from trader.strategies.base import BaseStrategy
from trader.core.models import Signal, Position
import pandas as pd

class MyStrategy(BaseStrategy):
    name = "my-strategy"
    min_bars_required = 20  # Need at least 20 bars of data

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add your indicators to the dataframe."""
        data = data.copy()
        data["rsi"] = self._calculate_rsi(data["close"], 14)
        return data

    def generate_signal(
        self,
        data: pd.DataFrame,
        symbol: str,
        position: Position | None = None,
    ) -> Signal:
        """Generate buy/sell/hold signal."""
        rsi = data["rsi"].iloc[-1]

        if rsi < 30:
            return self._buy_signal(symbol, reason="RSI oversold")
        elif rsi > 70 and position:
            return self._sell_signal(symbol, reason="RSI overbought")
        else:
            return self._hold_signal(symbol, reason="No signal")
```

## Configuration

For real market data (optional), create `.env`:

```bash
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
ALPACA_PAPER=true
```

Without API keys, the platform uses realistic mock data.

## Project Structure

```
trader/
├── src/trader/
│   ├── config/       # Settings
│   ├── core/         # Domain models (Bar, Signal, Order, Position)
│   ├── data/         # Data fetching (Alpaca + Mock)
│   ├── strategies/   # Trading strategies
│   ├── broker/       # Broker integrations (Paper, Alpaca)
│   ├── risk/         # Risk management
│   └── engine/       # Backtest & live trading engines
├── tests/            # Test suite (101 tests)
├── .claude/          # Claude Code skills & plans
├── justfile          # Dev commands
└── pyproject.toml    # Dependencies
```

## Development

```bash
just check              # Run lint + typecheck + tests
just test               # Run tests only
just lint-fix           # Auto-fix linting issues
just format             # Format code
```

## License

MIT
