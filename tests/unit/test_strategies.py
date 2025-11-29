"""Tests for trading strategies."""

from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest

from trader.core.models import Position, SignalAction
from trader.strategies.builtin.sma_crossover import SMACrossover


class TestSMACrossover:
    """Tests for SMA Crossover strategy."""

    def test_init_default_periods(self) -> None:
        """Test default period values."""
        strategy = SMACrossover()
        assert strategy.fast_period == 10
        assert strategy.slow_period == 50

    def test_init_custom_periods(self) -> None:
        """Test custom period values."""
        strategy = SMACrossover(fast_period=5, slow_period=20)
        assert strategy.fast_period == 5
        assert strategy.slow_period == 20

    def test_init_invalid_periods_raises(self) -> None:
        """Test that invalid periods raise errors."""
        with pytest.raises(ValueError, match="fast_period must be less than slow_period"):
            SMACrossover(fast_period=20, slow_period=10)

        with pytest.raises(ValueError, match="fast_period must be less than slow_period"):
            SMACrossover(fast_period=10, slow_period=10)

    def test_init_non_positive_periods_raises(self) -> None:
        """Test that non-positive periods raise errors."""
        with pytest.raises(ValueError, match="Periods must be positive"):
            SMACrossover(fast_period=0, slow_period=10)

        with pytest.raises(ValueError, match="Periods must be positive"):
            SMACrossover(fast_period=10, slow_period=-5)

    def test_name_property(self) -> None:
        """Test strategy name generation."""
        strategy = SMACrossover(fast_period=10, slow_period=50)
        assert strategy.name == "sma_crossover_10_50"

    def test_min_bars_required(self) -> None:
        """Test minimum bars calculation."""
        strategy = SMACrossover(fast_period=10, slow_period=50)
        assert strategy.min_bars_required == 51  # slow_period + 1

    def test_calculate_indicators(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Test indicator calculation."""
        strategy = SMACrossover(fast_period=3, slow_period=5)
        result = strategy.calculate_indicators(sample_ohlcv_df)

        assert "sma_fast" in result.columns
        assert "sma_slow" in result.columns

        # First few values should be NaN
        assert pd.isna(result["sma_fast"].iloc[0])
        assert pd.isna(result["sma_slow"].iloc[0])

        # Later values should be calculated
        assert not pd.isna(result["sma_fast"].iloc[3])
        assert not pd.isna(result["sma_slow"].iloc[5])

    def test_calculate_indicators_does_not_modify_input(
        self, sample_ohlcv_df: pd.DataFrame
    ) -> None:
        """Test that original DataFrame is not modified."""
        strategy = SMACrossover(fast_period=3, slow_period=5)
        original_columns = list(sample_ohlcv_df.columns)

        strategy.calculate_indicators(sample_ohlcv_df)

        assert list(sample_ohlcv_df.columns) == original_columns

    def test_generate_signal_not_enough_data(self) -> None:
        """Test signal generation with insufficient data."""
        strategy = SMACrossover(fast_period=10, slow_period=50)
        data = pd.DataFrame(
            {"close": [100, 101, 102]},
            index=pd.date_range("2024-01-01", periods=3),
        )
        data = strategy.calculate_indicators(data)

        signal = strategy.generate_signal(data, "AAPL")

        assert signal.action == SignalAction.HOLD
        assert "Not enough data" in signal.reason

    def test_generate_signal_bullish_crossover(self) -> None:
        """Test buy signal on bullish crossover."""
        strategy = SMACrossover(fast_period=3, slow_period=5)

        # Create data where fast SMA crosses above slow SMA on the LAST bar
        # Previous: fast < slow, Current: fast > slow
        # We need: declining prices (so fast < slow), then sharp spike on last bar
        prices = [120, 115, 110, 105, 100, 95, 90, 85, 80, 75, 120]
        data = pd.DataFrame(
            {"close": prices},
            index=pd.date_range("2024-01-01", periods=len(prices)),
        )
        data = strategy.calculate_indicators(data)

        signal = strategy.generate_signal(data, "AAPL")

        assert signal.action == SignalAction.BUY
        assert "Bullish" in signal.reason
        assert signal.symbol == "AAPL"

    def test_generate_signal_bearish_crossover(self) -> None:
        """Test sell signal on bearish crossover."""
        strategy = SMACrossover(fast_period=3, slow_period=5)

        # Create data where fast SMA crosses below slow SMA on the LAST bar
        # Previous: fast > slow, Current: fast < slow
        # We need: rising prices (so fast > slow), then sharp drop on last bar
        prices = [80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 80]
        data = pd.DataFrame(
            {"close": prices},
            index=pd.date_range("2024-01-01", periods=len(prices)),
        )
        data = strategy.calculate_indicators(data)

        signal = strategy.generate_signal(data, "AAPL")

        assert signal.action == SignalAction.SELL
        assert "Bearish" in signal.reason

    def test_generate_signal_no_crossover(self) -> None:
        """Test hold signal when no crossover."""
        strategy = SMACrossover(fast_period=3, slow_period=5)

        # Steady uptrend - fast always above slow
        prices = [100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120]
        data = pd.DataFrame(
            {"close": prices},
            index=pd.date_range("2024-01-01", periods=len(prices)),
        )
        data = strategy.calculate_indicators(data)

        signal = strategy.generate_signal(data, "AAPL")

        assert signal.action == SignalAction.HOLD
        assert "No crossover" in signal.reason

    def test_generate_signal_with_position(self) -> None:
        """Test that position is accepted (for future use)."""
        strategy = SMACrossover(fast_period=3, slow_period=5)
        position = Position(
            symbol="AAPL",
            quantity=100,
            avg_entry_price=Decimal("100.00"),
        )

        prices = [100] * 10
        data = pd.DataFrame(
            {"close": prices},
            index=pd.date_range("2024-01-01", periods=len(prices)),
        )
        data = strategy.calculate_indicators(data)

        # Should not raise with position
        signal = strategy.generate_signal(data, "AAPL", position=position)
        assert signal is not None


class TestBaseStrategyHelpers:
    """Tests for BaseStrategy helper methods."""

    def test_hold_signal(self) -> None:
        """Test hold signal creation."""
        strategy = SMACrossover()
        signal = strategy.hold_signal("AAPL", "Test reason")

        assert signal.action == SignalAction.HOLD
        assert signal.symbol == "AAPL"
        assert signal.reason == "Test reason"

    def test_buy_signal(self) -> None:
        """Test buy signal creation."""
        strategy = SMACrossover()
        signal = strategy.buy_signal(
            "AAPL", "Test buy", confidence=0.8, quantity=100
        )

        assert signal.action == SignalAction.BUY
        assert signal.symbol == "AAPL"
        assert signal.reason == "Test buy"
        assert signal.confidence == 0.8
        assert signal.quantity == 100

    def test_sell_signal(self) -> None:
        """Test sell signal creation."""
        strategy = SMACrossover()
        signal = strategy.sell_signal("AAPL", "Test sell")

        assert signal.action == SignalAction.SELL
        assert signal.symbol == "AAPL"
        assert signal.confidence == 1.0  # Default

    def test_should_generate_signal(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Test should_generate_signal check."""
        strategy = SMACrossover(fast_period=3, slow_period=5)

        # 10 rows should be enough (min is 6)
        assert strategy.should_generate_signal(sample_ohlcv_df) is True

        # 3 rows is not enough
        small_df = sample_ohlcv_df.head(3)
        assert strategy.should_generate_signal(small_df) is False
