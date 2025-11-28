# Trader - Automated Stock Trading Platform

## Project Overview

An automated stock trading platform built with Python and Alpaca for backtesting, paper trading, and live trading.

## Tech Stack

- **Python 3.11+** with full async/await
- **Alpaca** for broker integration (paper + live trading)
- **backtesting.py** for strategy backtesting
- **SQLite** for data caching (MVP), PostgreSQL later
- **Pydantic** for configuration and validation
- **pytest** + **pytest-asyncio** for testing

## Code Conventions

### Python Style

- Follow PEP 8 with 88 character line length (Black default)
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
# Run all tests
pytest

# Run with coverage
pytest --cov=src/trader --cov-report=term-missing

# Run specific module
pytest tests/unit/test_strategies.py
```

## Project Structure

```
trader/
├── src/trader/
│   ├── config/       # Settings, configuration
│   ├── core/         # Domain models, events
│   ├── data/         # Data fetching, storage
│   ├── strategies/   # Trading strategies
│   ├── broker/       # Broker integrations
│   ├── risk/         # Risk management
│   ├── engine/       # Backtest & live engines
│   └── portfolio/    # Portfolio tracking
├── scripts/          # CLI entry points
├── tests/            # Test suite
└── notebooks/        # Jupyter notebooks
```

## Development Workflow

1. Create feature branch from `main`
2. Write tests first (TDD encouraged)
3. Implement feature
4. Run `pytest` and ensure passing
5. Run `ruff check` for linting
6. Commit with conventional commit message
7. Create PR when ready

## Environment Variables

Required in `.env`:
```
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
ALPACA_PAPER=true
```

## Key Design Patterns

### Strategy Pattern
All trading strategies inherit from `BaseStrategy` and implement:
- `calculate_indicators(data: pd.DataFrame) -> pd.DataFrame`
- `generate_signal(data: pd.DataFrame) -> Signal`

### Broker Abstraction
All brokers implement `IBroker` interface for easy swapping.

### Event-Driven (Phase 3)
Components communicate via `EventBus` for loose coupling.
