# Trader Implementation Plan

## Overview
Automated stock trading platform with backtesting, paper trading, and live trading capabilities.

## Tech Stack
- **Python 3.11+** with async/await
- **Alpaca** for broker integration (paper + live)
- **uv** for dependency management
- **Pydantic** for configuration and validation
- **pytest** for testing
- **ruff** for linting, **mypy** for type checking

---

## Phase 1: Foundation & Backtesting ✅

### Core Infrastructure
- [x] Project structure with `src/trader/` layout
- [x] Configuration with pydantic-settings
- [x] Domain models (Bar, Signal, Order, Position)
- [x] Logging with loguru

### Data Layer
- [x] Base data fetcher interface
- [x] Mock data fetcher (geometric Brownian motion)
- [x] Alpaca data fetcher

### Strategy Framework
- [x] Base strategy abstract class
- [x] SMA crossover strategy implementation
- [x] Signal generation with confidence scores

### Backtesting Engine
- [x] Backtest runner with position tracking
- [x] Performance metrics (Sharpe, drawdown, win rate)
- [x] Trade recording and analysis
- [x] CLI command: `trader backtest AAPL`
- [x] Multi-stock scanning: `trader scan mag7 --strategy momentum`

---

## Phase 2: Paper & Live Trading ✅

### Broker Abstraction
- [x] Base broker interface
- [x] Paper broker (simulated trading)
- [x] Alpaca broker implementation
- [x] Bracket orders (stop loss + take profit)
- [x] Trailing stop support (falls back to fixed stop due to API limitation)

### Risk Management
- [x] Risk manager with configurable limits
- [x] Position sizing (percentage, max value)
- [x] Daily loss limits
- [x] Max positions limit
- [x] Stop loss / take profit
- [x] Circuit breakers (max trades/day, max daily loss)

### Live Trading Engine
- [x] Real-time trading loop
- [x] Market hours awareness
- [x] Day trading mode (close EOD)
- [x] Graceful shutdown
- [x] Safety limits configuration

### Trading Strategies
- [x] SMA crossover
- [x] RSI strategy
- [x] MACD crossover
- [x] Momentum strategy

### Notifications
- [x] Discord webhook integration
- [x] Trade signal notifications
- [x] Order fill notifications
- [x] Daily summary notifications
- [x] Circuit breaker alerts

### Storage & Reporting
- [x] SQLite database for trade history
- [x] Live trade tracking (entry/exit)
- [x] Daily P&L recording
- [x] Performance reports
- [x] CSV export

### CLI Commands
- [x] `trader backtest AAPL` - Run backtest
- [x] `trader scan mag7` - Multi-stock scan
- [x] `trader strategies` - List strategies
- [x] `trader live AAPL --strategy sma` - Live trading
- [x] `trader report` - Performance report
- [x] `trader export` - Export trades to CSV
- [x] `trader pnl` - Daily P&L history
- [x] `trader live-trades` - Trade history
- [x] `trader notify-test` - Test Discord notifications

---

## Phase 3: Position Management & Dashboard ✅

### 3.1 Position Management Commands ✅
- [x] `trader positions` - List all open positions with P&L
- [x] `trader close AAPL` - Close a specific position
- [x] `trader close all` - Close all positions
- [x] `trader orders` - List open orders
- [x] `trader cancel <order-id>` - Cancel specific order
- [x] `trader cancel all` - Cancel all open orders
- [x] `trader account` - Show account summary (equity, buying power, etc.)

### 3.2 Performance Dashboard ✅
- [x] Rich terminal UI with live updates
- [x] Real-time P&L display
- [x] Open positions table
- [x] Recent trades list
- [x] Account equity chart (sparkline)
- [x] Strategy performance metrics
- [x] Keyboard shortcuts for quick actions

### 3.3 Multi-Strategy Support ✅
- [x] Run multiple strategies simultaneously
- [x] Strategy-per-symbol configuration
- [x] Strategy performance comparison
- [x] CLI: `trader live AAPL:sma,MSFT:momentum`
- [x] Strategy allocation weights

---

## Phase 4: Portfolio Management 📊

### 4.1 Portfolio Rebalancing ✅
- [x] Target allocation configuration (e.g., AAPL: 30%, MSFT: 20%)
- [x] Automatic rebalancing on schedule
- [x] Rebalancing threshold (e.g., rebalance if >5% drift)
- [ ] Tax-aware rebalancing (minimize short-term gains)
- [x] CLI: `trader rebalance --dry-run`
- [x] CLI: `trader allocations` - Show current vs target

### 4.2 Additional Strategies ✅
- [x] Bollinger Bands strategy
- [x] Mean Reversion strategy
- [x] VWAP strategy
- [ ] Pairs trading strategy
- [ ] Custom strategy loader (from file)

### 4.3 Risk Analytics
- [ ] Value at Risk (VaR) calculation
- [ ] Portfolio correlation matrix
- [ ] Sector exposure analysis
- [ ] Beta calculation vs benchmark
- [ ] Maximum drawdown tracking
- [ ] CLI: `trader risk` - Risk dashboard

### 4.4 Session & Trading History Persistence
- [ ] Persist dashboard session stats to database
- [ ] Save equity snapshots over time
- [ ] Track strategy performance across sessions
- [ ] CLI: `trader sessions` - View past dashboard sessions
- [ ] Compare paper vs backtest results
- [ ] Performance charts with matplotlib/plotly
- [ ] Strategy comparison charts
- [ ] Export to HTML report

---

## Phase 5: Advanced Features 🚀

### 5.1 True Trailing Stops
- [ ] WebSocket connection to Alpaca trading stream
- [ ] Monitor order fills in real-time
- [ ] Submit trailing stop order after entry fills
- [ ] Track trailing stop state
- [ ] Handle partial fills

### 5.2 Scheduled Trading
- [ ] Cron-like scheduler for strategy runs
- [ ] Market open/close triggers
- [ ] CLI: `trader schedule AAPL --strategy sma --at "09:35"`
- [ ] Schedule management: `trader schedules list/add/remove`
- [ ] Daemon mode: `trader daemon start`

---

## Development Workflow

```bash
# Setup
uv sync

# Daily commands
just check          # lint + typecheck + test
just test           # run tests
just backtest AAPL  # run backtest

# Before committing
just all            # format + lint + typecheck + test
```

---

## Project Structure

```
trader/
├── src/trader/
│   ├── config/         # Settings, configuration
│   ├── core/           # Domain models
│   ├── data/           # Data fetching
│   ├── strategies/     # Trading strategies
│   ├── broker/         # Broker integrations
│   ├── risk/           # Risk management
│   ├── engine/         # Backtest & live engines
│   ├── storage/        # Database, persistence
│   ├── notifications/  # Discord, alerts
│   └── cli.py          # CLI entry point
├── tests/              # Test suite (157 tests)
├── .claude/            # Claude skills & plans
├── justfile            # Dev commands
└── pyproject.toml      # Project config
```

---

## Implementation Priority

| Priority | Feature | Phase | Status |
|----------|---------|-------|--------|
| 1 | Position management commands | 3.1 | ✅ Done |
| 2 | Performance dashboard | 3.2 | ✅ Done |
| 3 | Multi-strategy support | 3.3 | ✅ Done |
| 4 | Portfolio rebalancing | 4.1 | ✅ Done |
| 5 | More strategies (Bollinger, Mean Reversion, VWAP) | 4.2 | ✅ Done |
| 6 | Session & history persistence | 4.4 | Next |
| 7 | Risk analytics (VaR, correlation) | 4.3 | Planned |
| 8 | True trailing stops (WebSocket) | 5.1 | Planned |
| 9 | Scheduled trading | 5.2 | Planned |
