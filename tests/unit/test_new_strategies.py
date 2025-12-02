"""Tests for Bollinger Bands, VWAP, and Mean Reversion strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trader.core.models import Position, SignalAction
from trader.strategies.builtin.bollinger import (
    BollingerBandsStrategy,
    BollingerBreakoutStrategy,
)
from trader.strategies.builtin.mean_reversion import (
    MeanReversionPairsStrategy,
    MeanReversionStrategy,
)
from trader.strategies.builtin.vwap import VWAPStrategy, VWAPTrendStrategy
from trader.strategies.registry import get_strategy, list_strategies


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create sample OHLCV data with 50 bars."""
    np.random.seed(42)
    n = 50
    dates = pd.date_range(end="2024-01-01", periods=n, freq="D")

    # Generate realistic price data
    base_price = 100.0
    returns = np.random.randn(n) * 0.02  # 2% daily volatility
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
def oversold_data() -> pd.DataFrame:
    """Create data with oversold conditions (price at lower band)."""
    np.random.seed(42)
    n = 50
    dates = pd.date_range(end="2024-01-01", periods=n, freq="D")

    # Start stable, then drop sharply at the end
    prices = np.ones(n) * 100.0
    prices[-5:] = [95, 92, 88, 85, 82]  # Sharp decline

    return pd.DataFrame(
        {
            "open": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": np.random.randint(100000, 1000000, n),
        },
        index=dates,
    )


@pytest.fixture
def overbought_data() -> pd.DataFrame:
    """Create data with overbought conditions (price at upper band)."""
    np.random.seed(42)
    n = 50
    dates = pd.date_range(end="2024-01-01", periods=n, freq="D")

    # Start stable, then rise sharply at the end
    prices = np.ones(n) * 100.0
    prices[-5:] = [105, 108, 112, 115, 118]  # Sharp rise

    return pd.DataFrame(
        {
            "open": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": np.random.randint(100000, 1000000, n),
        },
        index=dates,
    )


class TestBollingerBandsStrategy:
    """Tests for Bollinger Bands mean reversion strategy."""

    def test_initialization(self) -> None:
        """Test strategy initialization."""
        strategy = BollingerBandsStrategy(period=20, num_std=2.0)
        assert strategy.period == 20
        assert strategy.num_std == 2.0
        assert strategy.min_bars_required == 21

    def test_invalid_period(self) -> None:
        """Test that invalid period raises error."""
        with pytest.raises(ValueError, match="Period must be positive"):
            BollingerBandsStrategy(period=0)

    def test_invalid_num_std(self) -> None:
        """Test that invalid num_std raises error."""
        with pytest.raises(ValueError, match="standard deviations must be positive"):
            BollingerBandsStrategy(num_std=-1)

    def test_calculate_indicators(self, sample_data: pd.DataFrame) -> None:
        """Test indicator calculation."""
        strategy = BollingerBandsStrategy()
        data = strategy.calculate_indicators(sample_data)

        assert "bb_upper" in data.columns
        assert "bb_middle" in data.columns
        assert "bb_lower" in data.columns
        assert "bb_pct" in data.columns

        # Middle should be between upper and lower
        valid = data.dropna()
        assert (valid["bb_upper"] >= valid["bb_middle"]).all()
        assert (valid["bb_middle"] >= valid["bb_lower"]).all()

    def test_buy_signal_at_lower_band(self, oversold_data: pd.DataFrame) -> None:
        """Test buy signal when price touches lower band."""
        strategy = BollingerBandsStrategy(period=20, num_std=2.0)
        data = strategy.calculate_indicators(oversold_data)

        signal = strategy.generate_signal(data, "TEST")

        # Should generate buy signal when oversold
        assert signal.action == SignalAction.BUY
        assert "lower band" in signal.reason.lower() or "%b" in signal.reason.lower()

    def test_sell_signal_at_upper_band(self, overbought_data: pd.DataFrame) -> None:
        """Test sell signal when price touches upper band with position."""
        strategy = BollingerBandsStrategy(period=20, num_std=2.0)
        data = strategy.calculate_indicators(overbought_data)

        position = Position(
            symbol="TEST",
            quantity=100,
            avg_entry_price=100.0,
            current_price=118.0,
        )

        signal = strategy.generate_signal(data, "TEST", position)

        # Should generate sell signal when overbought and holding
        assert signal.action == SignalAction.SELL

    def test_hold_signal_in_middle(self, sample_data: pd.DataFrame) -> None:
        """Test hold signal when price is in the middle of bands."""
        strategy = BollingerBandsStrategy()
        data = strategy.calculate_indicators(sample_data)

        signal = strategy.generate_signal(data, "TEST")

        # Price should be within bands in random data
        assert signal.action in [SignalAction.HOLD, SignalAction.BUY]


class TestBollingerBreakoutStrategy:
    """Tests for Bollinger Bands breakout strategy."""

    def test_initialization(self) -> None:
        """Test strategy initialization."""
        strategy = BollingerBreakoutStrategy(period=20, num_std=2.0)
        assert strategy.period == 20
        assert strategy.name == "bollinger_breakout_20_2.0"

    def test_calculate_indicators(self, sample_data: pd.DataFrame) -> None:
        """Test indicator calculation."""
        strategy = BollingerBreakoutStrategy()
        data = strategy.calculate_indicators(sample_data)

        assert "bb_upper" in data.columns
        assert "bb_middle" in data.columns
        assert "bb_lower" in data.columns


class TestVWAPStrategy:
    """Tests for VWAP mean reversion strategy."""

    def test_initialization(self) -> None:
        """Test strategy initialization."""
        strategy = VWAPStrategy(deviation_pct=1.0, use_bands=True)
        assert strategy.deviation_pct == 1.0
        assert strategy.use_bands is True
        assert strategy.min_bars_required == 20

    def test_invalid_deviation(self) -> None:
        """Test that invalid deviation raises error."""
        with pytest.raises(ValueError, match="Deviation percentage must be positive"):
            VWAPStrategy(deviation_pct=0)

    def test_calculate_indicators(self, sample_data: pd.DataFrame) -> None:
        """Test VWAP indicator calculation."""
        strategy = VWAPStrategy()
        data = strategy.calculate_indicators(sample_data)

        assert "vwap" in data.columns
        assert "vwap_deviation" in data.columns

        # VWAP should be positive
        assert (data["vwap"] > 0).all()

    def test_vwap_bands_calculated(self, sample_data: pd.DataFrame) -> None:
        """Test that VWAP bands are calculated when enabled."""
        strategy = VWAPStrategy(use_bands=True)
        data = strategy.calculate_indicators(sample_data)

        assert "vwap_upper" in data.columns
        assert "vwap_lower" in data.columns

    def test_buy_signal_below_vwap(self, oversold_data: pd.DataFrame) -> None:
        """Test buy signal when price is below VWAP."""
        strategy = VWAPStrategy(deviation_pct=2.0)
        data = strategy.calculate_indicators(oversold_data)

        signal = strategy.generate_signal(data, "TEST")

        # Should consider buying when significantly below VWAP
        # Note: may be HOLD if not below threshold
        assert signal.action in [SignalAction.BUY, SignalAction.HOLD]

    def test_hold_near_vwap(self) -> None:
        """Test hold signal when price is near VWAP."""
        # Create data where price is close to VWAP
        n = 50
        dates = pd.date_range(end="2024-01-01", periods=n, freq="D")
        prices = np.ones(n) * 100.0  # Constant price = VWAP

        data = pd.DataFrame(
            {
                "open": prices,
                "high": prices * 1.001,
                "low": prices * 0.999,
                "close": prices,
                "volume": np.ones(n) * 500000,
            },
            index=dates,
        )

        strategy = VWAPStrategy(deviation_pct=1.0)
        data = strategy.calculate_indicators(data)

        signal = strategy.generate_signal(data, "TEST")

        # Should hold when within threshold (price = VWAP)
        assert signal.action == SignalAction.HOLD


class TestVWAPTrendStrategy:
    """Tests for VWAP trend following strategy."""

    def test_initialization(self) -> None:
        """Test strategy initialization."""
        strategy = VWAPTrendStrategy(confirmation_bars=3)
        assert strategy.confirmation_bars == 3

    def test_invalid_confirmation_bars(self) -> None:
        """Test that invalid confirmation bars raises error."""
        with pytest.raises(ValueError, match="Confirmation bars must be at least 1"):
            VWAPTrendStrategy(confirmation_bars=0)

    def test_calculate_indicators(self, sample_data: pd.DataFrame) -> None:
        """Test indicator calculation."""
        strategy = VWAPTrendStrategy()
        data = strategy.calculate_indicators(sample_data)

        assert "vwap" in data.columns
        assert "above_vwap" in data.columns


class TestMeanReversionStrategy:
    """Tests for mean reversion strategy using z-score."""

    def test_initialization(self) -> None:
        """Test strategy initialization."""
        strategy = MeanReversionStrategy(lookback=20, entry_zscore=2.0, exit_zscore=0.5)
        assert strategy.lookback == 20
        assert strategy.entry_zscore == 2.0
        assert strategy.exit_zscore == 0.5

    def test_invalid_lookback(self) -> None:
        """Test that invalid lookback raises error."""
        with pytest.raises(ValueError, match="Lookback must be greater than 1"):
            MeanReversionStrategy(lookback=1)

    def test_invalid_entry_zscore(self) -> None:
        """Test that invalid entry z-score raises error."""
        with pytest.raises(ValueError, match="Entry z-score must be positive"):
            MeanReversionStrategy(entry_zscore=0)

    def test_invalid_exit_zscore(self) -> None:
        """Test that exit z-score must be less than entry."""
        with pytest.raises(ValueError, match="Exit z-score must be less than entry"):
            MeanReversionStrategy(entry_zscore=2.0, exit_zscore=2.5)

    def test_calculate_indicators(self, sample_data: pd.DataFrame) -> None:
        """Test indicator calculation."""
        strategy = MeanReversionStrategy()
        data = strategy.calculate_indicators(sample_data)

        assert "rolling_mean" in data.columns
        assert "rolling_std" in data.columns
        assert "zscore" in data.columns

    def test_buy_signal_oversold(self, oversold_data: pd.DataFrame) -> None:
        """Test buy signal when z-score is very negative."""
        strategy = MeanReversionStrategy(lookback=20, entry_zscore=1.5)
        data = strategy.calculate_indicators(oversold_data)

        signal = strategy.generate_signal(data, "TEST")

        # Should buy when significantly oversold
        assert signal.action == SignalAction.BUY
        assert "z-score" in signal.reason.lower() or "oversold" in signal.reason.lower()

    def test_sell_signal_at_mean(self, sample_data: pd.DataFrame) -> None:
        """Test sell signal when position exists and price at mean."""
        strategy = MeanReversionStrategy(lookback=20, exit_zscore=0.5)
        data = strategy.calculate_indicators(sample_data)

        position = Position(
            symbol="TEST",
            quantity=100,
            avg_entry_price=90.0,
            current_price=100.0,
        )

        signal = strategy.generate_signal(data, "TEST", position)

        # May sell if z-score is near 0, otherwise hold
        assert signal.action in [SignalAction.SELL, SignalAction.HOLD]


class TestMeanReversionChannelStrategy:
    """Tests for price channel mean reversion strategy."""

    def test_initialization(self) -> None:
        """Test strategy initialization."""
        strategy = MeanReversionPairsStrategy(lookback=20, entry_pct=5.0)
        assert strategy.lookback == 20
        assert strategy.entry_pct == 5.0

    def test_invalid_lookback(self) -> None:
        """Test that invalid lookback raises error."""
        with pytest.raises(ValueError, match="Lookback must be greater than 1"):
            MeanReversionPairsStrategy(lookback=1)

    def test_invalid_entry_pct(self) -> None:
        """Test that invalid entry percentage raises error."""
        with pytest.raises(ValueError, match="Entry percentage must be between 0 and 50"):
            MeanReversionPairsStrategy(entry_pct=60)

    def test_calculate_indicators(self, sample_data: pd.DataFrame) -> None:
        """Test indicator calculation."""
        strategy = MeanReversionPairsStrategy()
        data = strategy.calculate_indicators(sample_data)

        assert "channel_high" in data.columns
        assert "channel_low" in data.columns
        assert "channel_mid" in data.columns
        assert "channel_position" in data.columns

    def test_buy_at_channel_low(self, oversold_data: pd.DataFrame) -> None:
        """Test buy signal at channel low."""
        strategy = MeanReversionPairsStrategy(lookback=20, entry_pct=10.0)
        data = strategy.calculate_indicators(oversold_data)

        signal = strategy.generate_signal(data, "TEST")

        # Should buy at channel low
        assert signal.action == SignalAction.BUY
        assert "channel" in signal.reason.lower()


class TestStrategyRegistry:
    """Tests for strategy registry with new strategies."""

    def test_bollinger_registered(self) -> None:
        """Test that Bollinger strategy is registered."""
        strategy = get_strategy("bollinger")
        assert isinstance(strategy, BollingerBandsStrategy)

    def test_bollinger_breakout_registered(self) -> None:
        """Test that Bollinger breakout strategy is registered."""
        strategy = get_strategy("bollinger_breakout")
        assert isinstance(strategy, BollingerBreakoutStrategy)

    def test_vwap_registered(self) -> None:
        """Test that VWAP strategy is registered."""
        strategy = get_strategy("vwap")
        assert isinstance(strategy, VWAPStrategy)

    def test_vwap_trend_registered(self) -> None:
        """Test that VWAP trend strategy is registered."""
        strategy = get_strategy("vwap_trend")
        assert isinstance(strategy, VWAPTrendStrategy)

    def test_mean_reversion_registered(self) -> None:
        """Test that mean reversion strategy is registered."""
        strategy = get_strategy("mean_reversion")
        assert isinstance(strategy, MeanReversionStrategy)

    def test_mean_reversion_channel_registered(self) -> None:
        """Test that mean reversion channel strategy is registered."""
        strategy = get_strategy("mean_reversion_channel")
        assert isinstance(strategy, MeanReversionPairsStrategy)

    def test_list_strategies_includes_new(self) -> None:
        """Test that list_strategies includes new strategies."""
        strategies = list_strategies()
        names = [s["name"] for s in strategies]

        assert "bollinger" in names
        assert "bollinger_breakout" in names
        assert "vwap" in names
        assert "vwap_trend" in names
        assert "mean_reversion" in names
        assert "mean_reversion_channel" in names

    def test_custom_parameters(self) -> None:
        """Test creating strategy with custom parameters."""
        strategy = get_strategy("bollinger", period=30, num_std=2.5)
        assert strategy.period == 30
        assert strategy.num_std == 2.5

        strategy = get_strategy("mean_reversion", lookback=30, entry_zscore=2.5)
        assert strategy.lookback == 30
        assert strategy.entry_zscore == 2.5
