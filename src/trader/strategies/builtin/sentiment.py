"""Sentiment-Enhanced Trading Strategy."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pandas as pd

from trader.core.models import Signal
from trader.data.sentiment import (
    SentimentData,
    get_sentiment_fetcher,
)
from trader.strategies.base import BaseStrategy

if TYPE_CHECKING:
    from trader.core.models import Position
    from trader.data.sentiment import BaseSentimentFetcher


class SentimentStrategy(BaseStrategy):
    """
    Sentiment-Enhanced Trading Strategy.

    Uses news sentiment to confirm or filter trading signals.
    Can be used standalone or combined with technical strategies.

    Modes:
    - standalone: Trade purely on sentiment signals
    - filter: Only take trades when sentiment aligns
    - boost: Adjust position size based on sentiment strength

    Parameters:
        bullish_threshold: Score above which sentiment is considered bullish (default: 0.15)
        bearish_threshold: Score below which sentiment is considered bearish (default: -0.15)
        min_articles: Minimum articles required for valid signal (default: 3)
        lookback_hours: Hours of news to analyze (default: 24)
        mode: Trading mode - "standalone", "filter", or "boost" (default: "standalone")
    """

    def __init__(
        self,
        bullish_threshold: float = 0.15,
        bearish_threshold: float = -0.15,
        min_articles: int = 3,
        lookback_hours: int = 24,
        mode: str = "standalone",
        sentiment_fetcher: BaseSentimentFetcher | None = None,
    ) -> None:
        """Initialize sentiment strategy.

        Args:
            bullish_threshold: Score above which to consider bullish
            bearish_threshold: Score below which to consider bearish
            min_articles: Minimum articles for valid signal
            lookback_hours: Hours of news history to analyze
            mode: Trading mode (standalone, filter, boost)
            sentiment_fetcher: Optional custom sentiment fetcher

        Raises:
            ValueError: If thresholds are invalid
        """
        if bullish_threshold <= bearish_threshold:
            raise ValueError("bullish_threshold must be greater than bearish_threshold")
        if min_articles < 1:
            raise ValueError("min_articles must be at least 1")
        if mode not in ("standalone", "filter", "boost"):
            raise ValueError("mode must be 'standalone', 'filter', or 'boost'")

        self.bullish_threshold = bullish_threshold
        self.bearish_threshold = bearish_threshold
        self.min_articles = min_articles
        self.lookback_hours = lookback_hours
        self.mode = mode
        self._fetcher = sentiment_fetcher

        # Cache sentiment data to avoid repeated API calls
        self._sentiment_cache: dict[str, SentimentData] = {}
        self._cache_timestamp: pd.Timestamp | None = None
        self._cache_ttl_minutes = 15  # Refresh every 15 minutes

    @property
    def fetcher(self) -> BaseSentimentFetcher:
        """Get sentiment fetcher (lazy initialization)."""
        if self._fetcher is None:
            self._fetcher = get_sentiment_fetcher()
        return self._fetcher

    @property
    def name(self) -> str:
        return f"sentiment_{self.mode}_{self.lookback_hours}h"

    @property
    def description(self) -> str:
        return (
            f"Sentiment Strategy ({self.mode} mode): "
            f"Buy when sentiment > {self.bullish_threshold}, "
            f"Sell when sentiment < {self.bearish_threshold}"
        )

    @property
    def min_bars_required(self) -> int:
        # Sentiment doesn't need price history, but we need at least 1 bar
        return 1

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add sentiment-related columns to data.

        Note: Actual sentiment fetching is async and done in generate_signal.
        This just adds placeholder columns for compatibility.
        """
        data = data.copy()
        data["sentiment_score"] = float("nan")
        data["sentiment_label"] = ""
        data["sentiment_articles"] = 0
        return data

    def _is_cache_valid(self) -> bool:
        """Check if sentiment cache is still valid."""
        if self._cache_timestamp is None:
            return False
        now = pd.Timestamp.now()
        age_minutes = (now - self._cache_timestamp).total_seconds() / 60
        return age_minutes < self._cache_ttl_minutes

    def _get_cached_sentiment(self, symbol: str) -> SentimentData | None:
        """Get cached sentiment data if valid."""
        if self._is_cache_valid() and symbol in self._sentiment_cache:
            return self._sentiment_cache[symbol]
        return None

    def _cache_sentiment(self, symbol: str, data: SentimentData) -> None:
        """Cache sentiment data."""
        self._sentiment_cache[symbol] = data
        self._cache_timestamp = pd.Timestamp.now()

    async def fetch_sentiment_async(self, symbol: str) -> SentimentData:
        """Fetch sentiment data (async)."""
        cached = self._get_cached_sentiment(symbol)
        if cached is not None:
            return cached

        sentiment = await self.fetcher.fetch_sentiment(
            symbol, lookback_hours=self.lookback_hours
        )
        self._cache_sentiment(symbol, sentiment)
        return sentiment

    def fetch_sentiment(self, symbol: str) -> SentimentData:
        """Fetch sentiment data (sync wrapper)."""
        cached = self._get_cached_sentiment(symbol)
        if cached is not None:
            return cached

        # Run async in new event loop if needed
        try:
            asyncio.get_running_loop()
            # We're in an async context, create a task
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    self.fetcher.fetch_sentiment(
                        symbol, lookback_hours=self.lookback_hours
                    ),
                )
                sentiment = future.result()
        except RuntimeError:
            # No running loop, we can use asyncio.run
            sentiment = asyncio.run(
                self.fetcher.fetch_sentiment(
                    symbol, lookback_hours=self.lookback_hours
                )
            )

        self._cache_sentiment(symbol, sentiment)
        return sentiment

    def _interpret_sentiment(
        self, sentiment: SentimentData
    ) -> tuple[str, float]:
        """
        Interpret sentiment data into action and confidence.

        Returns:
            Tuple of (action: "buy"|"sell"|"hold", confidence: 0.0-1.0)
        """
        # Not enough articles for reliable signal
        if sentiment.article_count < self.min_articles:
            return "hold", 0.0

        score = sentiment.overall_score

        if score >= self.bullish_threshold:
            # Bullish - confidence scales with score strength
            confidence = min(1.0, (score - self.bullish_threshold) / 0.5 + 0.5)
            return "buy", confidence
        elif score <= self.bearish_threshold:
            # Bearish
            confidence = min(1.0, (self.bearish_threshold - score) / 0.5 + 0.5)
            return "sell", confidence
        else:
            # Neutral
            return "hold", 0.5

    def generate_signal(
        self,
        data: pd.DataFrame,
        symbol: str,
        position: Position | None = None,
    ) -> Signal:
        """Generate trading signal based on sentiment."""
        if not self.should_generate_signal(data):
            return self.hold_signal(symbol, "Not enough price data")

        # Fetch sentiment
        sentiment = self.fetch_sentiment(symbol)
        action, confidence = self._interpret_sentiment(sentiment)

        # Build reason string
        if sentiment.article_count == 0:
            reason = "No news articles found"
            return self.hold_signal(symbol, reason)

        reason = (
            f"Sentiment: {sentiment.overall_label.value} "
            f"(score={sentiment.overall_score:.2f}, "
            f"articles={sentiment.article_count}, "
            f"bullish={sentiment.bullish_count}, bearish={sentiment.bearish_count})"
        )

        if sentiment.article_count < self.min_articles:
            return self.hold_signal(
                symbol,
                f"Insufficient news: {sentiment.article_count} < {self.min_articles} required",
            )

        # Generate signal based on mode
        if self.mode == "standalone":
            return self._standalone_signal(
                symbol, action, confidence, reason, position
            )
        elif self.mode == "filter":
            return self._filter_signal(symbol, action, confidence, reason, position)
        else:  # boost mode
            return self._boost_signal(symbol, action, confidence, reason, position)

    def _standalone_signal(
        self,
        symbol: str,
        action: str,
        confidence: float,
        reason: str,
        position: Position | None,
    ) -> Signal:
        """Generate standalone sentiment signal."""
        if action == "buy" and position is None:
            return self.buy_signal(symbol, reason, confidence=confidence)
        elif action == "sell" and position is not None:
            return self.sell_signal(symbol, reason, confidence=confidence)
        else:
            return self.hold_signal(symbol, reason)

    def _filter_signal(
        self,
        symbol: str,
        action: str,
        confidence: float,
        reason: str,
        position: Position | None,
    ) -> Signal:
        """
        Filter mode - return signal that can be used to confirm/reject other signals.

        In filter mode:
        - Return BUY only if sentiment is bullish (allows buying)
        - Return SELL only if sentiment is bearish (allows selling)
        - Return HOLD if sentiment is neutral (blocks trades)
        """
        if action == "buy":
            return self.buy_signal(
                symbol, f"Sentiment confirms BUY: {reason}", confidence=confidence
            )
        elif action == "sell":
            return self.sell_signal(
                symbol, f"Sentiment confirms SELL: {reason}", confidence=confidence
            )
        else:
            return self.hold_signal(symbol, f"Sentiment neutral - blocking trade: {reason}")

    def _boost_signal(
        self,
        symbol: str,
        action: str,
        confidence: float,
        reason: str,
        position: Position | None,
    ) -> Signal:
        """
        Boost mode - adjust confidence based on sentiment strength.

        Returns signal with confidence that can be used to adjust position size.
        """
        if action == "buy" and position is None:
            return self.buy_signal(
                symbol,
                f"Boosted BUY (sentiment +{confidence:.0%}): {reason}",
                confidence=confidence,
            )
        elif action == "sell" and position is not None:
            return self.sell_signal(
                symbol,
                f"Boosted SELL (sentiment -{confidence:.0%}): {reason}",
                confidence=confidence,
            )
        else:
            # In boost mode, neutral sentiment reduces confidence
            return self.hold_signal(symbol, f"No boost: {reason}")


class SentimentMomentumStrategy(BaseStrategy):
    """
    Combined Sentiment + Momentum Strategy.

    Uses momentum for entry timing and sentiment for confirmation.
    Only buys when:
    1. Momentum is positive (price trending up)
    2. Sentiment is bullish (news is positive)

    Sells when:
    1. Momentum turns negative, OR
    2. Sentiment turns bearish

    This combination filters out momentum trades that lack fundamental support.
    """

    def __init__(
        self,
        momentum_days: int = 20,
        sentiment_threshold: float = 0.10,
        min_articles: int = 2,
        lookback_hours: int = 48,
        sentiment_fetcher: BaseSentimentFetcher | None = None,
    ) -> None:
        """Initialize combined strategy."""
        self.momentum_days = momentum_days
        self.sentiment_threshold = sentiment_threshold
        self.min_articles = min_articles
        self.lookback_hours = lookback_hours

        # Create sub-strategies
        self._sentiment_strategy = SentimentStrategy(
            bullish_threshold=sentiment_threshold,
            bearish_threshold=-sentiment_threshold,
            min_articles=min_articles,
            lookback_hours=lookback_hours,
            mode="filter",
            sentiment_fetcher=sentiment_fetcher,
        )

    @property
    def name(self) -> str:
        return f"sentiment_momentum_{self.momentum_days}d"

    @property
    def description(self) -> str:
        return (
            f"Sentiment + Momentum: Buy when {self.momentum_days}d momentum positive "
            f"AND sentiment > {self.sentiment_threshold}"
        )

    @property
    def min_bars_required(self) -> int:
        return self.momentum_days + 1

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum indicators."""
        data = data.copy()

        # Calculate momentum
        data["momentum"] = data["close"].pct_change(self.momentum_days) * 100
        data["momentum_5d"] = data["close"].pct_change(5) * 100

        # Simple moving average for trend
        data["sma_20"] = data["close"].rolling(20).mean()
        data["above_sma"] = data["close"] > data["sma_20"]

        return data

    def generate_signal(
        self,
        data: pd.DataFrame,
        symbol: str,
        position: Position | None = None,
    ) -> Signal:
        """Generate combined signal."""
        if not self.should_generate_signal(data):
            return self.hold_signal(symbol, "Not enough data")

        current = data.iloc[-1]
        momentum = current["momentum"]

        if pd.isna(momentum):
            return self.hold_signal(symbol, "Momentum not ready")

        # Get sentiment signal
        sentiment_signal = self._sentiment_strategy.generate_signal(
            data, symbol, position
        )
        sentiment = self._sentiment_strategy.fetch_sentiment(symbol)

        # Combine signals
        momentum_bullish = momentum > 0
        momentum_bearish = momentum < -5  # More negative threshold for sell

        if position is None:
            # Looking to buy
            if momentum_bullish and sentiment_signal.action.name == "BUY":
                return self.buy_signal(
                    symbol,
                    f"Momentum ({momentum:.1f}%) + Sentiment ({sentiment.overall_score:.2f}) aligned",
                    confidence=sentiment_signal.confidence,
                )
            elif momentum_bullish:
                return self.hold_signal(
                    symbol,
                    f"Momentum positive ({momentum:.1f}%) but sentiment not bullish",
                )
            else:
                return self.hold_signal(
                    symbol,
                    f"Momentum negative ({momentum:.1f}%)",
                )
        else:
            # Looking to sell
            if momentum_bearish or sentiment_signal.action.name == "SELL":
                reason = []
                if momentum_bearish:
                    reason.append(f"momentum={momentum:.1f}%")
                if sentiment_signal.action.name == "SELL":
                    reason.append(f"sentiment={sentiment.overall_score:.2f}")
                return self.sell_signal(
                    symbol,
                    f"Exit signal: {', '.join(reason)}",
                )
            else:
                return self.hold_signal(
                    symbol,
                    f"Holding: momentum={momentum:.1f}%, sentiment={sentiment.overall_score:.2f}",
                )
