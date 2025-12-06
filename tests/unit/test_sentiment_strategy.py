"""Tests for sentiment-based trading strategies."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

import pandas as pd
import pytest

from trader.core.models import Position, SignalAction
from trader.data.sentiment import (
    MockSentimentFetcher,
    NewsArticle,
    SentimentData,
    SentimentLabel,
)
from trader.strategies.builtin.sentiment import (
    SentimentMomentumStrategy,
    SentimentStrategy,
)


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create sample OHLCV data."""
    dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
    return pd.DataFrame(
        {
            "open": [100 + i * 0.5 for i in range(50)],
            "high": [101 + i * 0.5 for i in range(50)],
            "low": [99 + i * 0.5 for i in range(50)],
            "close": [100.5 + i * 0.5 for i in range(50)],
            "volume": [1000000] * 50,
        },
        index=dates,
    )


@pytest.fixture
def mock_fetcher() -> MockSentimentFetcher:
    """Create a mock sentiment fetcher with fixed seed."""
    return MockSentimentFetcher(seed=42)


class TestSentimentStrategy:
    """Tests for SentimentStrategy."""

    def test_initialization_defaults(self) -> None:
        """Test default initialization."""
        strategy = SentimentStrategy()

        assert strategy.bullish_threshold == 0.15
        assert strategy.bearish_threshold == -0.15
        assert strategy.min_articles == 3
        assert strategy.lookback_hours == 24
        assert strategy.mode == "standalone"

    def test_initialization_custom(self) -> None:
        """Test custom initialization."""
        strategy = SentimentStrategy(
            bullish_threshold=0.20,
            bearish_threshold=-0.20,
            min_articles=5,
            lookback_hours=48,
            mode="filter",
        )

        assert strategy.bullish_threshold == 0.20
        assert strategy.bearish_threshold == -0.20
        assert strategy.min_articles == 5
        assert strategy.lookback_hours == 48
        assert strategy.mode == "filter"

    def test_invalid_thresholds(self) -> None:
        """Test that invalid thresholds raise error."""
        with pytest.raises(ValueError, match="bullish_threshold must be greater"):
            SentimentStrategy(bullish_threshold=-0.1, bearish_threshold=0.1)

    def test_invalid_mode(self) -> None:
        """Test that invalid mode raises error."""
        with pytest.raises(ValueError, match="mode must be"):
            SentimentStrategy(mode="invalid")

    def test_name_property(self) -> None:
        """Test strategy name generation."""
        strategy = SentimentStrategy(mode="standalone", lookback_hours=24)
        assert strategy.name == "sentiment_standalone_24h"

        strategy = SentimentStrategy(mode="filter", lookback_hours=48)
        assert strategy.name == "sentiment_filter_48h"

    def test_min_bars_required(self) -> None:
        """Test minimum bars required."""
        strategy = SentimentStrategy()
        assert strategy.min_bars_required == 1

    def test_calculate_indicators(self, sample_data: pd.DataFrame) -> None:
        """Test that calculate_indicators adds sentiment columns."""
        strategy = SentimentStrategy()
        result = strategy.calculate_indicators(sample_data)

        assert "sentiment_score" in result.columns
        assert "sentiment_label" in result.columns
        assert "sentiment_articles" in result.columns

    def test_interpret_sentiment_bullish(self) -> None:
        """Test bullish sentiment interpretation."""
        strategy = SentimentStrategy(bullish_threshold=0.15)

        sentiment = SentimentData(
            symbol="AAPL",
            timestamp=datetime.now(),
            overall_score=0.30,
            article_count=5,
        )

        action, confidence = strategy._interpret_sentiment(sentiment)
        assert action == "buy"
        assert confidence >= 0.5

    def test_interpret_sentiment_bearish(self) -> None:
        """Test bearish sentiment interpretation."""
        strategy = SentimentStrategy(bearish_threshold=-0.15)

        sentiment = SentimentData(
            symbol="AAPL",
            timestamp=datetime.now(),
            overall_score=-0.30,
            article_count=5,
        )

        action, confidence = strategy._interpret_sentiment(sentiment)
        assert action == "sell"
        assert confidence >= 0.5

    def test_interpret_sentiment_neutral(self) -> None:
        """Test neutral sentiment interpretation."""
        strategy = SentimentStrategy()

        sentiment = SentimentData(
            symbol="AAPL",
            timestamp=datetime.now(),
            overall_score=0.05,
            article_count=5,
        )

        action, confidence = strategy._interpret_sentiment(sentiment)
        assert action == "hold"

    def test_interpret_sentiment_insufficient_articles(self) -> None:
        """Test with insufficient articles."""
        strategy = SentimentStrategy(min_articles=5)

        sentiment = SentimentData(
            symbol="AAPL",
            timestamp=datetime.now(),
            overall_score=0.50,
            article_count=3,
        )

        action, confidence = strategy._interpret_sentiment(sentiment)
        assert action == "hold"
        assert confidence == 0.0

    def test_generate_signal_buy(
        self, sample_data: pd.DataFrame, mock_fetcher: MockSentimentFetcher
    ) -> None:
        """Test generating buy signal."""
        strategy = SentimentStrategy(
            mode="standalone", sentiment_fetcher=MockSentimentFetcher(seed=42)
        )

        # Manually inject bullish sentiment into the cache
        bullish_data = SentimentData(
            symbol="AAPL",
            timestamp=datetime.now(),
            overall_score=0.35,
            overall_label=SentimentLabel.BULLISH,
            article_count=10,
            bullish_count=7,
            bearish_count=2,
            neutral_count=1,
        )
        strategy._cache_sentiment("AAPL", bullish_data)

        signal = strategy.generate_signal(sample_data, "AAPL", position=None)

        assert signal.action == SignalAction.BUY
        assert "Bullish" in signal.reason

    def test_generate_signal_sell(
        self, sample_data: pd.DataFrame
    ) -> None:
        """Test generating sell signal."""
        strategy = SentimentStrategy(
            mode="standalone", sentiment_fetcher=MockSentimentFetcher(seed=42)
        )

        # Manually inject bearish sentiment into the cache
        bearish_data = SentimentData(
            symbol="AAPL",
            timestamp=datetime.now(),
            overall_score=-0.35,
            overall_label=SentimentLabel.BEARISH,
            article_count=10,
            bullish_count=2,
            bearish_count=7,
            neutral_count=1,
        )
        strategy._cache_sentiment("AAPL", bearish_data)

        position = Position(
            symbol="AAPL",
            quantity=100,
            avg_entry_price=Decimal("150.00"),
        )
        signal = strategy.generate_signal(sample_data, "AAPL", position=position)

        assert signal.action == SignalAction.SELL
        assert "Bearish" in signal.reason

    def test_generate_signal_hold_neutral(
        self, sample_data: pd.DataFrame
    ) -> None:
        """Test hold signal on neutral sentiment."""
        class NeutralFetcher(MockSentimentFetcher):
            async def fetch_sentiment(
                self, symbol: str, lookback_hours: int = 24
            ) -> SentimentData:
                return SentimentData(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    overall_score=0.05,
                    overall_label=SentimentLabel.NEUTRAL,
                    article_count=10,
                )

        strategy = SentimentStrategy(
            mode="standalone", sentiment_fetcher=NeutralFetcher()
        )
        signal = strategy.generate_signal(sample_data, "AAPL", position=None)

        assert signal.action == SignalAction.HOLD

    def test_filter_mode(self, sample_data: pd.DataFrame) -> None:
        """Test filter mode signals."""
        class BullishFetcher(MockSentimentFetcher):
            async def fetch_sentiment(
                self, symbol: str, lookback_hours: int = 24
            ) -> SentimentData:
                return SentimentData(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    overall_score=0.35,
                    overall_label=SentimentLabel.BULLISH,
                    article_count=10,
                )

        strategy = SentimentStrategy(mode="filter", sentiment_fetcher=BullishFetcher())
        signal = strategy.generate_signal(sample_data, "AAPL", position=None)

        assert signal.action == SignalAction.BUY
        assert "confirms BUY" in signal.reason

    def test_sentiment_caching(self, sample_data: pd.DataFrame) -> None:
        """Test that sentiment data is cached."""
        call_count = 0

        class CountingFetcher(MockSentimentFetcher):
            async def fetch_sentiment(
                self, symbol: str, lookback_hours: int = 24
            ) -> SentimentData:
                nonlocal call_count
                call_count += 1
                return SentimentData(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    overall_score=0.0,
                    article_count=5,
                )

        strategy = SentimentStrategy(sentiment_fetcher=CountingFetcher())

        # First call should fetch
        strategy.generate_signal(sample_data, "AAPL")
        assert call_count == 1

        # Second call should use cache
        strategy.generate_signal(sample_data, "AAPL")
        assert call_count == 1  # Still 1, not 2


class TestSentimentMomentumStrategy:
    """Tests for SentimentMomentumStrategy."""

    def test_initialization(self) -> None:
        """Test strategy initialization."""
        strategy = SentimentMomentumStrategy(
            momentum_days=20,
            sentiment_threshold=0.10,
            min_articles=2,
            lookback_hours=48,
        )

        assert strategy.momentum_days == 20
        assert strategy.sentiment_threshold == 0.10

    def test_name_property(self) -> None:
        """Test strategy name."""
        strategy = SentimentMomentumStrategy(momentum_days=30)
        assert strategy.name == "sentiment_momentum_30d"

    def test_min_bars_required(self) -> None:
        """Test minimum bars required."""
        strategy = SentimentMomentumStrategy(momentum_days=20)
        assert strategy.min_bars_required == 21

    def test_calculate_indicators(self, sample_data: pd.DataFrame) -> None:
        """Test momentum indicator calculation."""
        strategy = SentimentMomentumStrategy(momentum_days=20)
        result = strategy.calculate_indicators(sample_data)

        assert "momentum" in result.columns
        assert "momentum_5d" in result.columns
        assert "sma_20" in result.columns
        assert "above_sma" in result.columns

    def test_buy_signal_positive_momentum_bullish_sentiment(
        self, sample_data: pd.DataFrame
    ) -> None:
        """Test buy when momentum positive and sentiment bullish."""
        strategy = SentimentMomentumStrategy(
            sentiment_fetcher=MockSentimentFetcher(seed=42)
        )

        # Calculate indicators first
        data_with_indicators = strategy.calculate_indicators(sample_data)

        # Inject bullish sentiment into the internal strategy's cache
        bullish_data = SentimentData(
            symbol="AAPL",
            timestamp=datetime.now(),
            overall_score=0.30,
            overall_label=SentimentLabel.SOMEWHAT_BULLISH,
            article_count=5,
        )
        strategy._sentiment_strategy._cache_sentiment("AAPL", bullish_data)

        signal = strategy.generate_signal(data_with_indicators, "AAPL", position=None)

        assert signal.action == SignalAction.BUY
        assert "Momentum" in signal.reason or "momentum" in signal.reason

    def test_hold_positive_momentum_neutral_sentiment(
        self, sample_data: pd.DataFrame
    ) -> None:
        """Test hold when momentum positive but sentiment neutral."""
        strategy = SentimentMomentumStrategy(
            sentiment_fetcher=MockSentimentFetcher(seed=42)
        )

        # Calculate indicators first
        data_with_indicators = strategy.calculate_indicators(sample_data)

        # Inject neutral sentiment
        neutral_data = SentimentData(
            symbol="AAPL",
            timestamp=datetime.now(),
            overall_score=0.05,
            overall_label=SentimentLabel.NEUTRAL,
            article_count=5,
        )
        strategy._sentiment_strategy._cache_sentiment("AAPL", neutral_data)

        signal = strategy.generate_signal(data_with_indicators, "AAPL", position=None)

        assert signal.action == SignalAction.HOLD
        assert "not bullish" in signal.reason.lower()

    def test_sell_negative_momentum(self, sample_data: pd.DataFrame) -> None:
        """Test sell when momentum turns negative."""
        # Create data with negative momentum (declining prices)
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        declining_data = pd.DataFrame(
            {
                "open": [100 - i * 0.5 for i in range(50)],
                "high": [101 - i * 0.5 for i in range(50)],
                "low": [99 - i * 0.5 for i in range(50)],
                "close": [100.5 - i * 0.5 for i in range(50)],
                "volume": [1000000] * 50,
            },
            index=dates,
        )

        strategy = SentimentMomentumStrategy(
            sentiment_fetcher=MockSentimentFetcher(seed=42)
        )

        # Calculate indicators first
        data_with_indicators = strategy.calculate_indicators(declining_data)

        # Inject neutral sentiment
        neutral_data = SentimentData(
            symbol="AAPL",
            timestamp=datetime.now(),
            overall_score=0.0,
            article_count=5,
        )
        strategy._sentiment_strategy._cache_sentiment("AAPL", neutral_data)

        position = Position(
            symbol="AAPL",
            quantity=100,
            avg_entry_price=Decimal("100.00"),
        )

        signal = strategy.generate_signal(data_with_indicators, "AAPL", position=position)
        assert signal.action == SignalAction.SELL


class TestSentimentStrategyRegistration:
    """Test that sentiment strategies are registered correctly."""

    def test_strategies_registered(self) -> None:
        """Test that sentiment strategies are in the registry."""
        from trader.strategies.registry import get_strategy, list_strategies

        strategies = list_strategies()
        strategy_names = [s["name"] for s in strategies]

        assert "sentiment" in strategy_names
        assert "sentiment_momentum" in strategy_names

    def test_get_sentiment_strategy(self) -> None:
        """Test getting sentiment strategy from registry."""
        from trader.strategies.registry import get_strategy

        strategy = get_strategy("sentiment")
        assert isinstance(strategy, SentimentStrategy)

    def test_get_sentiment_momentum_strategy(self) -> None:
        """Test getting sentiment_momentum strategy from registry."""
        from trader.strategies.registry import get_strategy

        strategy = get_strategy("sentiment_momentum")
        assert isinstance(strategy, SentimentMomentumStrategy)
