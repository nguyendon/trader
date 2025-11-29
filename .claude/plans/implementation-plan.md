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

---

## Phase 2: Paper Trading ✅

### Broker Abstraction
- [x] Base broker interface
- [x] Paper broker (simulated trading)
- [x] Alpaca broker implementation

### Risk Management
- [x] Risk manager with configurable limits
- [x] Position sizing (percentage, max value)
- [x] Daily loss limits
- [x] Max positions limit
- [x] Stop loss / take profit

### Live Trading Engine
- [x] Real-time trading loop
- [x] Market hours awareness
- [x] Day trading mode (close EOD)
- [x] Graceful shutdown

### Paper Trading CLI
- [x] CLI command: `trader paper AAPL,MSFT`
- [x] Demo mode with mock data

---

## Phase 3: Live Trading (Planned)

### Alpaca Integration
- [ ] Real API key configuration
- [ ] Order execution with Alpaca
- [ ] Position sync on startup
- [ ] Error handling and retries

### Monitoring
- [ ] Trade logging to database
- [ ] Performance dashboard
- [ ] Alerting (email/SMS on trades)

### Advanced Features
- [ ] Multiple strategies
- [ ] Portfolio-level risk management
- [ ] Scheduled strategy runs

---

## Phase 4: Advanced Strategies (Future)

### Additional Strategies
- [ ] RSI mean reversion
- [ ] Bollinger Band strategy
- [ ] MACD crossover
- [ ] Custom strategy loader

### Machine Learning
- [ ] Feature engineering pipeline
- [ ] Model training framework
- [ ] Prediction-based signals

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
│   ├── config/       # Settings, configuration
│   ├── core/         # Domain models
│   ├── data/         # Data fetching
│   ├── strategies/   # Trading strategies
│   ├── broker/       # Broker integrations
│   ├── risk/         # Risk management
│   ├── engine/       # Backtest & live engines
│   └── cli.py        # CLI entry point
├── tests/            # Test suite
├── .claude/          # Claude skills
├── justfile          # Dev commands
└── pyproject.toml    # Project config
```
