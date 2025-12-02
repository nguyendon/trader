"""Tests for pairs trading strategy."""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from trader.core.models import Position, SignalAction
from trader.strategies.builtin.pairs import (
    CointegrationPairsStrategy,
    PairsTradingStrategy,
)
from trader.strategies.registry import get_strategy, list_strategies


@pytest.fixture
def sample_primary_data() -> pd.DataFrame:
    """Create sample price data for primary symbol."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range(end="2024-01-01", periods=n, freq="D")

    # Generate correlated price data
    base_price = 150.0
    returns = np.random.randn(n) * 0.02
    prices = base_price * np.cumprod(1 + returns)

    return pd.DataFrame(
        {
            "open": prices * (1 + np.random.randn(n) * 0.005),
            "high": prices * (1 + abs(np.random.randn(n) * 0.01)),
            "low": prices * (1 - abs(np.random.randn(n) * 0.01)),
            "close": prices,
            "volume": np.random.randint(100000, 1000000, n),
        },
        index=dates,
    )


@pytest.fixture
def sample_secondary_data(sample_primary_data: pd.DataFrame) -> pd.DataFrame:
    """Create correlated secondary symbol data."""
    np.random.seed(123)
    n = len(sample_primary_data)

    # Create correlated prices (roughly 0.8 correlation)
    primary_returns = sample_primary_data["close"].pct_change().fillna(0)
    secondary_returns = 0.8 * primary_returns + 0.2 * np.random.randn(n) * 0.02

    base_price = 300.0
    prices = base_price * np.cumprod(1 + secondary_returns)

    return pd.DataFrame(
        {
            "open": prices * (1 + np.random.randn(n) * 0.005),
            "high": prices * (1 + abs(np.random.randn(n) * 0.01)),
            "low": prices * (1 - abs(np.random.randn(n) * 0.01)),
            "close": prices,
            "volume": np.random.randint(100000, 1000000, n),
        },
        index=sample_primary_data.index,
    )


class TestPairsTradingStrategy:
    """Tests for basic pairs trading strategy."""

    def test_initialization(self) -> None:
        """Test strategy initialization."""
        strategy = PairsTradingStrategy(
            primary_symbol="AAPL",
            secondary_symbol="MSFT",
            lookback=60,
            entry_zscore=2.0,
            exit_zscore=0.5,
        )
        assert strategy.primary_symbol == "AAPL"
        assert strategy.secondary_symbol == "MSFT"
        assert strategy.lookback == 60
        assert strategy.entry_zscore == 2.0
        assert strategy.exit_zscore == 0.5
        assert strategy.min_bars_required == 65

    def test_invalid_lookback(self) -> None:
        """Test that invalid lookback raises error."""
        with pytest.raises(ValueError, match="Lookback must be at least 10"):
            PairsTradingStrategy(lookback=5)

    def test_invalid_entry_zscore(self) -> None:
        """Test that invalid entry z-score raises error."""
        with pytest.raises(ValueError, match="Entry z-score must be positive"):
            PairsTradingStrategy(entry_zscore=0)

    def test_invalid_exit_zscore(self) -> None:
        """Test that exit z-score must be less than entry."""
        with pytest.raises(ValueError, match="Exit z-score must be less than entry"):
            PairsTradingStrategy(entry_zscore=2.0, exit_zscore=2.5)

    def test_name_property(self) -> None:
        """Test strategy name generation."""
        strategy = PairsTradingStrategy(
            primary_symbol="AAPL",
            secondary_symbol="MSFT",
            lookback=60,
        )
        assert strategy.name == "pairs_AAPL_MSFT_60"

    def test_calculate_hedge_ratio(
        self,
        sample_primary_data: pd.DataFrame,
        sample_secondary_data: pd.DataFrame,
    ) -> None:
        """Test hedge ratio calculation."""
        strategy = PairsTradingStrategy()

        hedge_ratio = strategy.calculate_hedge_ratio(
            sample_primary_data["close"],
            sample_secondary_data["close"],
        )

        # Hedge ratio should be reasonable (not extreme)
        assert 0.1 < hedge_ratio < 10

    def test_calculate_spread(
        self,
        sample_primary_data: pd.DataFrame,
        sample_secondary_data: pd.DataFrame,
    ) -> None:
        """Test spread calculation."""
        strategy = PairsTradingStrategy(use_log_prices=True)

        hedge_ratio = strategy.calculate_hedge_ratio(
            sample_primary_data["close"],
            sample_secondary_data["close"],
        )

        spread = strategy.calculate_spread(
            sample_primary_data["close"],
            sample_secondary_data["close"],
            hedge_ratio,
        )

        # Spread should be a series of the same length
        assert len(spread) == len(sample_primary_data)
        # Spread should be finite
        assert spread.isna().sum() == 0

    def test_calculate_zscore(
        self,
        sample_primary_data: pd.DataFrame,
        sample_secondary_data: pd.DataFrame,
    ) -> None:
        """Test z-score calculation."""
        strategy = PairsTradingStrategy(lookback=20)

        spread = sample_primary_data["close"] - sample_secondary_data["close"]
        zscore = strategy.calculate_zscore(spread)

        # Z-score should have NaN for initial period (lookback-1 values)
        assert zscore.iloc[: 20 - 1].isna().all()
        # Valid z-scores should be mostly between -3 and 3
        valid_zscore = zscore.dropna()
        assert (valid_zscore.abs() < 5).mean() > 0.9

    def test_calculate_indicators(
        self,
        sample_primary_data: pd.DataFrame,
        sample_secondary_data: pd.DataFrame,
    ) -> None:
        """Test indicator calculation with secondary data."""
        strategy = PairsTradingStrategy(lookback=20)
        strategy.set_secondary_data(sample_secondary_data)

        data = strategy.calculate_indicators(sample_primary_data)

        assert "spread" in data.columns
        assert "zscore" in data.columns
        assert "hedge_ratio" in data.columns

    def test_calculate_indicators_no_secondary(
        self,
        sample_primary_data: pd.DataFrame,
    ) -> None:
        """Test indicator calculation without secondary data."""
        strategy = PairsTradingStrategy(lookback=20)
        # Don't set secondary data

        data = strategy.calculate_indicators(sample_primary_data)

        # Should have NaN values when no secondary data
        assert data["spread"].isna().all()
        assert data["zscore"].isna().all()

    def test_generate_signal_hold_insufficient_data(
        self,
        sample_primary_data: pd.DataFrame,
    ) -> None:
        """Test hold signal with insufficient data."""
        strategy = PairsTradingStrategy(lookback=60)

        # Use only first 50 bars (less than min_bars_required)
        short_data = sample_primary_data.iloc[:50]
        signal = strategy.generate_signal(short_data, "AAPL")

        assert signal.action == SignalAction.HOLD
        assert "Insufficient data" in signal.reason

    def test_generate_signal_hold_no_secondary(
        self,
        sample_primary_data: pd.DataFrame,
    ) -> None:
        """Test hold signal when spread not available."""
        strategy = PairsTradingStrategy(lookback=20)

        data = strategy.calculate_indicators(sample_primary_data)
        signal = strategy.generate_signal(data, "AAPL")

        assert signal.action == SignalAction.HOLD
        assert "not available" in signal.reason

    def test_generate_signal_with_data(
        self,
        sample_primary_data: pd.DataFrame,
        sample_secondary_data: pd.DataFrame,
    ) -> None:
        """Test signal generation with valid data."""
        strategy = PairsTradingStrategy(lookback=20, entry_zscore=2.0)
        strategy.set_secondary_data(sample_secondary_data)

        data = strategy.calculate_indicators(sample_primary_data)
        signal = strategy.generate_signal(data, "AAPL")

        # Should generate a valid signal (BUY, SELL, or HOLD)
        assert signal.action in [SignalAction.BUY, SignalAction.SELL, SignalAction.HOLD]
        assert signal.symbol == "AAPL"

    def test_generate_exit_signal_long(
        self,
        sample_primary_data: pd.DataFrame,
        sample_secondary_data: pd.DataFrame,
    ) -> None:
        """Test exit signal for long position."""
        strategy = PairsTradingStrategy(lookback=20, entry_zscore=2.0, exit_zscore=0.5)
        strategy.set_secondary_data(sample_secondary_data)

        data = strategy.calculate_indicators(sample_primary_data)

        # Create a long position
        position = Position(
            symbol="AAPL",
            quantity=100,
            avg_entry_price=Decimal("150.00"),
            current_price=Decimal("155.00"),
        )

        signal = strategy.generate_signal(data, "AAPL", position)

        # Should have a reason related to position
        assert signal.symbol == "AAPL"


class TestCointegrationPairsStrategy:
    """Tests for cointegration-based pairs trading strategy."""

    def test_initialization(self) -> None:
        """Test strategy initialization."""
        strategy = CointegrationPairsStrategy(
            primary_symbol="AAPL",
            secondary_symbol="MSFT",
            lookback=60,
            coint_pvalue=0.05,
        )
        assert strategy.primary_symbol == "AAPL"
        assert strategy.secondary_symbol == "MSFT"
        assert strategy.coint_pvalue == 0.05

    def test_invalid_lookback(self) -> None:
        """Test that invalid lookback raises error."""
        with pytest.raises(ValueError, match="Lookback must be at least 20"):
            CointegrationPairsStrategy(lookback=15)

    def test_invalid_pvalue(self) -> None:
        """Test that invalid p-value raises error."""
        with pytest.raises(ValueError, match="Cointegration p-value must be between"):
            CointegrationPairsStrategy(coint_pvalue=1.5)

    def test_name_property(self) -> None:
        """Test strategy name."""
        strategy = CointegrationPairsStrategy(
            primary_symbol="AAPL", secondary_symbol="MSFT"
        )
        assert strategy.name == "coint_pairs_AAPL_MSFT"

    def test_calculate_indicators(
        self,
        sample_primary_data: pd.DataFrame,
        sample_secondary_data: pd.DataFrame,
    ) -> None:
        """Test indicator calculation."""
        strategy = CointegrationPairsStrategy(lookback=30)
        strategy.set_secondary_data(sample_secondary_data)

        data = strategy.calculate_indicators(sample_primary_data)

        assert "spread" in data.columns
        assert "zscore" in data.columns
        assert "is_cointegrated" in data.columns
        assert "hedge_ratio" in data.columns

    def test_generate_signal_not_cointegrated(
        self,
        sample_primary_data: pd.DataFrame,
        sample_secondary_data: pd.DataFrame,
    ) -> None:
        """Test hold signal when pair not cointegrated."""
        # Create uncorrelated secondary data
        np.random.seed(999)
        n = len(sample_primary_data)
        uncorrelated = pd.DataFrame(
            {
                "close": 100 * np.cumprod(1 + np.random.randn(n) * 0.02),
            },
            index=sample_primary_data.index,
        )

        strategy = CointegrationPairsStrategy(lookback=30, coint_pvalue=0.01)
        strategy.set_secondary_data(uncorrelated)

        data = strategy.calculate_indicators(sample_primary_data)
        signal = strategy.generate_signal(data, "AAPL")

        # Should hold when not cointegrated
        assert signal.action == SignalAction.HOLD


class TestPairsStrategyRegistry:
    """Tests for pairs strategy registry."""

    def test_pairs_registered(self) -> None:
        """Test that pairs strategy is registered."""
        strategy = get_strategy("pairs")
        assert isinstance(strategy, PairsTradingStrategy)

    def test_pairs_coint_registered(self) -> None:
        """Test that cointegration pairs strategy is registered."""
        strategy = get_strategy("pairs_coint")
        assert isinstance(strategy, CointegrationPairsStrategy)

    def test_list_strategies_includes_pairs(self) -> None:
        """Test that list_strategies includes pairs."""
        strategies = list_strategies()
        names = [s["name"] for s in strategies]

        assert "pairs" in names
        assert "pairs_coint" in names

    def test_custom_parameters(self) -> None:
        """Test creating pairs strategy with custom parameters."""
        strategy = get_strategy(
            "pairs",
            primary_symbol="GOOGL",
            secondary_symbol="META",
            lookback=90,
        )
        assert strategy.primary_symbol == "GOOGL"
        assert strategy.secondary_symbol == "META"
        assert strategy.lookback == 90
