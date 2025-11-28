---
name: pytest-testing
description: Comprehensive pytest testing guidance including fixtures, mocking, parametrization, async testing, coverage, and test organization. Use when writing unit tests, integration tests, setting up test fixtures, mocking dependencies, or debugging test failures.
---

# Pytest Testing Guide

## Test Structure

### File Organization
```
tests/
├── conftest.py          # Shared fixtures
├── unit/
│   ├── test_strategies.py
│   ├── test_risk.py
│   └── test_models.py
├── integration/
│   ├── test_broker.py
│   └── test_data_fetcher.py
└── fixtures/
    └── sample_data.json
```

### Test Naming Convention
```python
def test_<function>_<scenario>_<expected>():
    """Test that <function> does <expected> when <scenario>."""
    pass

# Examples:
def test_calculate_position_size_with_max_limit_caps_at_limit():
def test_generate_signal_when_sma_crosses_returns_buy():
def test_submit_order_with_invalid_symbol_raises_error():
```

## Fixtures

### Basic Fixtures
```python
# conftest.py
import pytest
import pandas as pd

@pytest.fixture
def sample_ohlcv():
    """Sample OHLCV data for testing."""
    return pd.DataFrame({
        'open': [100, 101, 102],
        'high': [105, 106, 107],
        'low': [99, 100, 101],
        'close': [104, 105, 106],
        'volume': [1000, 1100, 1200],
    })

@pytest.fixture
def risk_config():
    """Default risk configuration."""
    return RiskConfig(
        max_position_size_pct=0.1,
        max_daily_loss_pct=0.02,
        stop_loss_pct=0.05,
    )
```

### Fixture Scopes
```python
@pytest.fixture(scope="session")
def database():
    """One database connection for entire test session."""
    db = create_test_database()
    yield db
    db.cleanup()

@pytest.fixture(scope="module")
def api_client():
    """One client per test module."""
    return TestClient()

@pytest.fixture  # default scope="function"
def fresh_portfolio():
    """Fresh portfolio for each test."""
    return Portfolio(initial_capital=100000)
```

### Async Fixtures
```python
@pytest.fixture
async def connected_broker():
    """Broker fixture that handles connection lifecycle."""
    broker = MockBroker()
    await broker.connect()
    yield broker
    await broker.disconnect()
```

## Mocking

### Basic Mocking
```python
from unittest.mock import Mock, patch, MagicMock

def test_with_mock():
    mock_broker = Mock()
    mock_broker.get_positions.return_value = []

    manager = PortfolioManager(broker=mock_broker)
    result = manager.get_total_value()

    mock_broker.get_positions.assert_called_once()
```

### Patching
```python
@patch('trader.data.fetcher.AlpacaClient')
def test_fetcher(mock_client_class):
    mock_client = mock_client_class.return_value
    mock_client.get_bars.return_value = sample_data

    fetcher = DataFetcher()
    result = fetcher.fetch("AAPL")

    assert len(result) > 0
```

### Async Mocking
```python
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_async_broker():
    mock_broker = AsyncMock()
    mock_broker.submit_order.return_value = {"id": "123", "status": "filled"}

    result = await mock_broker.submit_order("AAPL", 10, "buy")
    assert result["status"] == "filled"
```

## Parametrization

### Basic Parametrize
```python
@pytest.mark.parametrize("input,expected", [
    (100, 10),    # 10% of 100
    (1000, 100),  # 10% of 1000
    (0, 0),       # Edge case
])
def test_calculate_position_size(input, expected, risk_config):
    result = calculate_position_size(input, risk_config)
    assert result == expected
```

### Multiple Parameters
```python
@pytest.mark.parametrize("symbol,timeframe,expected_count", [
    ("AAPL", "1D", 252),
    ("AAPL", "1H", 1764),
    ("MSFT", "1D", 252),
])
def test_fetch_bars(symbol, timeframe, expected_count):
    result = fetch_bars(symbol, timeframe, days=365)
    assert len(result) == expected_count
```

### IDs for Clarity
```python
@pytest.mark.parametrize("action,position,expected", [
    pytest.param("buy", None, True, id="buy-no-position"),
    pytest.param("buy", "long", False, id="buy-already-long"),
    pytest.param("sell", "long", True, id="sell-close-long"),
])
def test_should_execute(action, position, expected):
    assert should_execute(action, position) == expected
```

## Async Testing

```python
import pytest

@pytest.mark.asyncio
async def test_fetch_data():
    result = await fetch_data("AAPL")
    assert isinstance(result, pd.DataFrame)

@pytest.mark.asyncio
async def test_concurrent_fetches():
    symbols = ["AAPL", "MSFT", "GOOGL"]
    results = await fetch_multiple(symbols)
    assert len(results) == 3
```

## Assertions

### Pandas DataFrames
```python
import pandas.testing as pdt

def test_dataframe_equality():
    pdt.assert_frame_equal(result, expected)

def test_series_equality():
    pdt.assert_series_equal(result['close'], expected['close'])
```

### Approximate Values (Floats)
```python
def test_sharpe_ratio():
    assert result == pytest.approx(1.5, rel=0.01)  # 1% tolerance
    assert result == pytest.approx(1.5, abs=0.1)   # Absolute tolerance
```

### Exceptions
```python
def test_invalid_symbol_raises():
    with pytest.raises(ValueError, match="Invalid symbol"):
        fetch_data("INVALID$$$")
```

## Coverage

```bash
# Run with coverage
pytest --cov=src/trader --cov-report=term-missing

# Generate HTML report
pytest --cov=src/trader --cov-report=html

# Fail if coverage below threshold
pytest --cov=src/trader --cov-fail-under=80
```

## Markers

```python
# pytest.ini or pyproject.toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow",
    "integration: marks integration tests",
    "requires_api: requires live API connection",
]

# Usage
@pytest.mark.slow
def test_full_backtest():
    ...

@pytest.mark.integration
@pytest.mark.asyncio
async def test_alpaca_connection():
    ...

# Run specific markers
# pytest -m "not slow"
# pytest -m integration
```

## Test Patterns

### Arrange-Act-Assert
```python
def test_position_sizing():
    # Arrange
    portfolio_value = 100000
    price = 150
    config = RiskConfig(max_position_size_pct=0.1)
    manager = RiskManager(config)

    # Act
    size = manager.calculate_position_size(portfolio_value, price)

    # Assert
    assert size == 66  # floor(10000 / 150)
```

### Given-When-Then (BDD Style)
```python
def test_sma_crossover_generates_buy_signal():
    """
    Given: SMA fast crosses above SMA slow
    When: generate_signal is called
    Then: returns a buy signal
    """
    # Given
    data = create_uptrend_data()
    strategy = SMACrossover(fast=10, slow=20)
    data = strategy.calculate_indicators(data)

    # When
    signal = strategy.generate_signal(data)

    # Then
    assert signal.action == "buy"
```
