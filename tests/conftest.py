"""Shared test fixtures."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pandas as pd
import pytest

from trader.core.models import Bar, Position, Signal, SignalAction, TimeFrame


@pytest.fixture
def sample_bar() -> Bar:
    """Create a sample bar for testing."""
    return Bar(
        symbol="AAPL",
        timestamp=datetime(2024, 1, 15, 9, 30),
        open=Decimal("185.00"),
        high=Decimal("187.50"),
        low=Decimal("184.00"),
        close=Decimal("186.25"),
        volume=1_000_000,
        timeframe=TimeFrame.DAY,
    )


@pytest.fixture
def sample_bars() -> list[Bar]:
    """Create sample bars for testing strategies."""
    base_time = datetime(2024, 1, 1, 9, 30)
    prices = [
        (100, 102, 99, 101),
        (101, 103, 100, 102),
        (102, 104, 101, 103),
        (103, 105, 102, 101),  # Price drop
        (101, 103, 100, 102),
        (102, 106, 101, 105),  # Breakout
        (105, 108, 104, 107),
        (107, 110, 106, 109),
        (109, 111, 108, 110),
        (110, 112, 109, 111),
    ]

    bars = []
    for i, (o, h, l, c) in enumerate(prices):
        bars.append(
            Bar(
                symbol="TEST",
                timestamp=base_time.replace(day=i + 1),
                open=Decimal(str(o)),
                high=Decimal(str(h)),
                low=Decimal(str(l)),
                close=Decimal(str(c)),
                volume=100_000 + i * 10_000,
            )
        )
    return bars


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """Create sample OHLCV DataFrame for testing."""
    return pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 101, 102, 105, 107, 109, 110],
            "high": [102, 103, 104, 105, 103, 106, 108, 110, 111, 112],
            "low": [99, 100, 101, 102, 100, 101, 104, 106, 108, 109],
            "close": [101, 102, 103, 101, 102, 105, 107, 109, 110, 111],
            "volume": [100000, 110000, 120000, 130000, 140000, 150000, 160000, 170000, 180000, 190000],
        },
        index=pd.date_range("2024-01-01", periods=10, freq="D"),
    )


@pytest.fixture
def sample_signal() -> Signal:
    """Create a sample signal for testing."""
    return Signal(
        action=SignalAction.BUY,
        symbol="AAPL",
        quantity=100,
        confidence=0.8,
        reason="SMA crossover",
    )


@pytest.fixture
def sample_position() -> Position:
    """Create a sample position for testing."""
    return Position(
        symbol="AAPL",
        quantity=100,
        avg_entry_price=Decimal("150.00"),
        current_price=Decimal("155.00"),
    )
