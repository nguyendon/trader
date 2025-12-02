"""Tests for trading strategies."""

from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest

from trader.core.models import Position, SignalAction
from trader.strategies.builtin.macd import MACDStrategy
from trader.strategies.builtin.momentum import MomentumStrategy
from trader.strategies.builtin.rsi import RSIStrategy
from trader.strategies.builtin.sma_crossover import SMACrossover
from trader.strategies.registry import get_strategy, list_strategies


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
        with pytest.raises(
            ValueError, match="fast_period must be less than slow_period"
        ):
            SMACrossover(fast_period=20, slow_period=10)

        with pytest.raises(
            ValueError, match="fast_period must be less than slow_period"
        ):
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
        signal = strategy.buy_signal("AAPL", "Test buy", confidence=0.8, quantity=100)

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


class TestRSIStrategy:
    """Tests for RSI Mean Reversion strategy."""

    def test_init_default_values(self) -> None:
        """Test default RSI values."""
        strategy = RSIStrategy()
        assert strategy.period == 14
        assert strategy.oversold == 30.0
        assert strategy.overbought == 70.0

    def test_init_custom_values(self) -> None:
        """Test custom RSI values."""
        strategy = RSIStrategy(period=10, oversold=25.0, overbought=75.0)
        assert strategy.period == 10
        assert strategy.oversold == 25.0
        assert strategy.overbought == 75.0

    def test_init_invalid_thresholds(self) -> None:
        """Test invalid threshold values."""
        with pytest.raises(ValueError, match="0 < oversold < overbought < 100"):
            RSIStrategy(oversold=70, overbought=30)  # reversed

        with pytest.raises(ValueError, match="0 < oversold < overbought < 100"):
            RSIStrategy(oversold=0, overbought=70)  # oversold at boundary

    def test_name_property(self) -> None:
        """Test strategy name."""
        strategy = RSIStrategy(period=14, oversold=30, overbought=70)
        assert strategy.name == "rsi_14_30_70"

    def test_calculate_indicators(self) -> None:
        """Test RSI indicator calculation."""
        strategy = RSIStrategy(period=5)

        # Create price data with enough bars
        prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109]
        data = pd.DataFrame(
            {"close": prices},
            index=pd.date_range("2024-01-01", periods=len(prices)),
        )
        result = strategy.calculate_indicators(data)

        assert "rsi" in result.columns
        # RSI should be between 0 and 100
        valid_rsi = result["rsi"].dropna()
        assert all(0 <= v <= 100 for v in valid_rsi)

    def test_generate_signal_oversold_buy(self) -> None:
        """Test buy signal when RSI is oversold."""
        strategy = RSIStrategy(period=5, oversold=30, overbought=70)

        # Create declining prices to push RSI below 30
        prices = [100, 95, 90, 85, 80, 75, 70, 65, 60, 55]
        data = pd.DataFrame(
            {"close": prices},
            index=pd.date_range("2024-01-01", periods=len(prices)),
        )
        data = strategy.calculate_indicators(data)

        signal = strategy.generate_signal(data, "AAPL", position=None)

        assert signal.action == SignalAction.BUY
        assert "oversold" in signal.reason.lower()

    def test_generate_signal_overbought_sell(self) -> None:
        """Test sell signal when RSI is overbought."""
        strategy = RSIStrategy(period=5, oversold=30, overbought=70)

        # Create rising prices to push RSI above 70
        prices = [100, 105, 110, 115, 120, 125, 130, 135, 140, 145]
        data = pd.DataFrame(
            {"close": prices},
            index=pd.date_range("2024-01-01", periods=len(prices)),
        )
        data = strategy.calculate_indicators(data)

        # Need a position to get sell signal
        position = Position(symbol="AAPL", quantity=100, avg_entry_price=Decimal("100"))
        signal = strategy.generate_signal(data, "AAPL", position=position)

        assert signal.action == SignalAction.SELL
        assert "overbought" in signal.reason.lower()

    def test_generate_signal_no_position_no_sell(self) -> None:
        """Test that overbought without position doesn't sell."""
        strategy = RSIStrategy(period=5, oversold=30, overbought=70)

        prices = [100, 105, 110, 115, 120, 125, 130, 135, 140, 145]
        data = pd.DataFrame(
            {"close": prices},
            index=pd.date_range("2024-01-01", periods=len(prices)),
        )
        data = strategy.calculate_indicators(data)

        signal = strategy.generate_signal(data, "AAPL", position=None)

        # Should hold, not sell (no position to sell)
        assert signal.action == SignalAction.HOLD


class TestMACDStrategy:
    """Tests for MACD strategy."""

    def test_init_default_values(self) -> None:
        """Test default MACD values."""
        strategy = MACDStrategy()
        assert strategy.fast_period == 12
        assert strategy.slow_period == 26
        assert strategy.signal_period == 9

    def test_init_invalid_periods(self) -> None:
        """Test invalid period configuration."""
        with pytest.raises(
            ValueError, match="fast_period must be less than slow_period"
        ):
            MACDStrategy(fast_period=26, slow_period=12)

    def test_name_property(self) -> None:
        """Test strategy name."""
        strategy = MACDStrategy()
        assert strategy.name == "macd_12_26_9"

    def test_calculate_indicators(self) -> None:
        """Test MACD indicator calculation."""
        strategy = MACDStrategy(fast_period=3, slow_period=6, signal_period=3)

        prices = list(range(100, 120)) + list(range(120, 100, -1))
        data = pd.DataFrame(
            {"close": prices},
            index=pd.date_range("2024-01-01", periods=len(prices)),
        )
        result = strategy.calculate_indicators(data)

        assert "macd" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_histogram" in result.columns

    def test_generate_signal_bullish_crossover(self) -> None:
        """Test buy signal on bullish MACD crossover."""
        strategy = MACDStrategy(fast_period=3, slow_period=6, signal_period=3)

        # Need prices where MACD crosses above signal on the LAST bar
        # Decline to establish negative MACD, then just enough rise to cross
        # Stop at the exact bar where crossover happens
        prices = [100, 98, 96, 94, 92, 90, 88, 86, 84, 82, 80, 78, 76, 74, 72, 70, 72]
        data = pd.DataFrame(
            {"close": prices},
            index=pd.date_range("2024-01-01", periods=len(prices)),
        )
        data = strategy.calculate_indicators(data)

        signal = strategy.generate_signal(data, "AAPL")

        # Should get a buy signal when MACD crosses above signal
        assert signal.action == SignalAction.BUY
        assert "Bullish" in signal.reason


class TestMomentumStrategy:
    """Tests for Momentum strategy."""

    def test_init_default_values(self) -> None:
        """Test default momentum values."""
        strategy = MomentumStrategy()
        assert strategy.lookback_days == 126
        assert strategy.skip_days == 5
        assert strategy.hold_days == 5

    def test_init_invalid_lookback(self) -> None:
        """Test invalid lookback configuration."""
        with pytest.raises(
            ValueError, match="lookback_days must be greater than skip_days"
        ):
            MomentumStrategy(lookback_days=5, skip_days=10)

    def test_name_property(self) -> None:
        """Test strategy name."""
        strategy = MomentumStrategy(lookback_days=126, skip_days=5)
        assert strategy.name == "momentum_126_5"

    def test_calculate_indicators(self) -> None:
        """Test momentum indicator calculation."""
        strategy = MomentumStrategy(lookback_days=20, skip_days=5)

        # Create uptrending data
        prices = [100 + i for i in range(30)]
        data = pd.DataFrame(
            {"close": prices},
            index=pd.date_range("2024-01-01", periods=len(prices)),
        )
        result = strategy.calculate_indicators(data)

        assert "momentum" in result.columns
        assert "return_5d" in result.columns
        assert "return_20d" in result.columns

        # Momentum should be positive for uptrend
        valid_momentum = result["momentum"].dropna()
        assert len(valid_momentum) > 0
        assert valid_momentum.iloc[-1] > 0

    def test_generate_signal_positive_momentum(self) -> None:
        """Test buy signal with positive momentum."""
        strategy = MomentumStrategy(lookback_days=20, skip_days=5, momentum_threshold=0)

        # Strong uptrend
        prices = [100 + i * 2 for i in range(30)]
        data = pd.DataFrame(
            {"close": prices},
            index=pd.date_range("2024-01-01", periods=len(prices)),
        )
        data = strategy.calculate_indicators(data)

        signal = strategy.generate_signal(data, "AAPL", position=None)

        assert signal.action == SignalAction.BUY
        assert "Positive momentum" in signal.reason

    def test_generate_signal_negative_momentum_sell(self) -> None:
        """Test sell signal with negative momentum when holding."""
        strategy = MomentumStrategy(lookback_days=20, skip_days=5)

        # Strong downtrend
        prices = [200 - i * 2 for i in range(30)]
        data = pd.DataFrame(
            {"close": prices},
            index=pd.date_range("2024-01-01", periods=len(prices)),
        )
        data = strategy.calculate_indicators(data)

        position = Position(symbol="AAPL", quantity=100, avg_entry_price=Decimal("150"))
        signal = strategy.generate_signal(data, "AAPL", position=position)

        assert signal.action == SignalAction.SELL
        assert "Negative momentum" in signal.reason

    def test_rank_symbols(self) -> None:
        """Test multi-symbol ranking."""
        strategy = MomentumStrategy(lookback_days=20, skip_days=5)

        # Create data for multiple symbols with different trends
        dates = pd.date_range("2024-01-01", periods=30)

        symbol_data = {
            "AAPL": pd.DataFrame(
                {"close": [100 + i * 3 for i in range(30)]}, index=dates
            ),  # Strong up
            "MSFT": pd.DataFrame(
                {"close": [100 + i for i in range(30)]}, index=dates
            ),  # Moderate up
            "GOOGL": pd.DataFrame(
                {"close": [100 - i for i in range(30)]}, index=dates
            ),  # Down
        }

        rankings = strategy.rank_symbols(symbol_data)

        assert len(rankings) == 3
        # AAPL should be first (highest momentum)
        assert rankings[0][0] == "AAPL"
        # GOOGL should be last (negative momentum)
        assert rankings[2][0] == "GOOGL"


class TestStrategyRegistry:
    """Tests for strategy registry."""

    def test_get_strategy_sma(self) -> None:
        """Test getting SMA strategy."""
        strategy = get_strategy("sma")
        assert isinstance(strategy, SMACrossover)

    def test_get_strategy_rsi(self) -> None:
        """Test getting RSI strategy."""
        strategy = get_strategy("rsi")
        assert isinstance(strategy, RSIStrategy)

    def test_get_strategy_macd(self) -> None:
        """Test getting MACD strategy."""
        strategy = get_strategy("macd")
        assert isinstance(strategy, MACDStrategy)

    def test_get_strategy_momentum(self) -> None:
        """Test getting momentum strategy."""
        strategy = get_strategy("momentum")
        assert isinstance(strategy, MomentumStrategy)

    def test_get_strategy_with_kwargs(self) -> None:
        """Test getting strategy with custom kwargs."""
        strategy = get_strategy("sma", fast_period=5, slow_period=20)
        assert strategy.fast_period == 5
        assert strategy.slow_period == 20

    def test_get_strategy_unknown(self) -> None:
        """Test getting unknown strategy raises."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            get_strategy("unknown_strategy")

    def test_list_strategies(self) -> None:
        """Test listing all strategies."""
        strategies = list_strategies()
        assert len(strategies) >= 4
        names = [s["name"] for s in strategies]
        assert "sma" in names
        assert "rsi" in names
        assert "macd" in names
        assert "momentum" in names
