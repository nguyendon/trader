# Trader - Automated Stock Trading Platform

## Project Overview

An automated stock trading platform built with Python and Alpaca for backtesting, paper trading, and live trading.

**Current Status**: Phase 1 (Backtesting) and Phase 2 (Paper Trading) complete. See `.claude/plans/implementation-plan.md` for roadmap.

## Tech Stack

- **Python 3.11+** with full async/await
- **uv** for dependency management (fast, creates .venv automatically)
- **Alpaca** for broker integration (paper + live trading)
- **Pydantic** for configuration and validation
- **pytest** + **pytest-asyncio** for testing
- **ruff** for linting + formatting
- **mypy** for type checking

## Quick Commands

```bash
just              # List all commands
just check        # Run lint + typecheck + tests
just test         # Run tests only
just backtest AAPL --days 365
just paper AAPL,MSFT
```

## Code Conventions

### Python Style

- Follow PEP 8 with 88 character line length (ruff default)
- Use type hints for all function signatures
- Use `from __future__ import annotations` for forward references
- Prefer `dataclasses` or `pydantic.BaseModel` for data structures
- Use async/await for all I/O operations (broker, data fetching)

### Imports

```python
# Standard library
from __future__ import annotations
import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

# Third party
import pandas as pd
from pydantic import BaseModel

# Local
from trader.core.models import Signal
```

Import order is enforced by ruff (isort rules).

### Naming Conventions

- Classes: `PascalCase` (e.g., `BaseStrategy`, `RiskManager`)
- Functions/methods: `snake_case` (e.g., `calculate_indicators`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_TIMEFRAME`)
- Private: prefix with `_` (e.g., `_validate_signal`)

### Error Handling

- Use custom exceptions in `trader.core.exceptions`
- Always log errors with context before re-raising
- Use `loguru` for all logging

## Git Conventions

### Commit Messages (Conventional Commits)

Format: `<type>(<scope>): <description>`

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Formatting, no code change
- `refactor`: Code change that neither fixes nor adds
- `test`: Adding or updating tests
- `chore`: Build process, dependencies, tooling

Examples:
```
feat(strategies): add SMA crossover strategy
fix(backtest): correct position sizing calculation
test(risk): add unit tests for RiskManager
docs: update README with setup instructions
```

### Branch Naming

- `feat/description` - New features
- `fix/description` - Bug fixes
- `refactor/description` - Refactoring

## Testing Requirements

### Test Structure

```
tests/
├── unit/           # Fast, isolated tests
├── integration/    # Tests with real dependencies
└── conftest.py     # Shared fixtures
```

### Coverage Requirements

- Minimum 80% coverage for `src/trader/`
- 100% coverage for `strategies/` and `risk/` modules
- All public functions must have tests

### Test Naming

```python
def test_<function_name>_<scenario>_<expected_result>():
    # Example: test_calculate_position_size_with_max_limit_returns_capped_value
```

### Running Tests

```bash
just test                    # Run all tests
just test-cov                # Run with coverage
uv run pytest tests/unit/    # Run specific directory
```

## Project Structure

```
trader/
├── src/trader/
│   ├── config/       # Settings, configuration
│   ├── core/         # Domain models (Bar, Signal, Order, Position)
│   ├── data/         # Data fetching (Alpaca + Mock)
│   ├── strategies/   # Trading strategies
│   ├── broker/       # Broker integrations (Paper, Alpaca)
│   ├── risk/         # Risk management
│   ├── engine/       # Backtest & live trading engines
│   └── cli.py        # CLI entry point
├── scripts/          # Utility scripts
├── tests/            # Test suite (101 tests)
├── .claude/          # Claude Code skills & plans
├── justfile          # Dev commands
└── pyproject.toml    # Project config
```

## Development Workflow

1. Create feature branch from `main`
2. Write tests first (TDD encouraged)
3. Implement feature
4. Run `just check` (lint + typecheck + tests)
5. Commit with conventional commit message
6. Create PR when ready

## Environment Variables

Optional in `.env` (mock data works without these):
```
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
ALPACA_PAPER=true
```

## Key Design Patterns

### Strategy Pattern
All trading strategies inherit from `BaseStrategy` and implement:
- `calculate_indicators(data: pd.DataFrame) -> pd.DataFrame`
- `generate_signal(data: pd.DataFrame, symbol: str, position: Position | None) -> Signal`

### Broker Abstraction
All brokers implement `BaseBroker` interface:
- `PaperBroker` - Simulated trading for testing
- `AlpacaBroker` - Real/paper trading via Alpaca API

### Risk Management
`RiskManager` enforces:
- Position sizing limits
- Daily loss limits
- Max positions
- Stop loss / take profit

## Linting & Type Checking

Configured in `pyproject.toml`:

```bash
just lint       # Run ruff check
just typecheck  # Run mypy
just lint-fix   # Auto-fix linting issues
```

Rules enforced:
- E, W, F: pycodestyle errors/warnings, pyflakes
- I: isort (import ordering)
- B: bugbear (common bugs)
- C4: comprehensions
- UP: pyupgrade
- SIM: simplify
