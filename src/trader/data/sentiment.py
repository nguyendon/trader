"""Sentiment data fetching from news and social media sources."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING

import httpx
from loguru import logger

if TYPE_CHECKING:
    from trader.config.settings import Settings


class SentimentLabel(str, Enum):
    """Sentiment classification labels."""

    BULLISH = "Bullish"
    SOMEWHAT_BULLISH = "Somewhat-Bullish"
    NEUTRAL = "Neutral"
    SOMEWHAT_BEARISH = "Somewhat-Bearish"
    BEARISH = "Bearish"


@dataclass
class NewsArticle:
    """A news article with sentiment data."""

    title: str
    summary: str
    source: str
    url: str
    published_at: datetime
    symbols: list[str]
    sentiment_score: float  # -1.0 (bearish) to 1.0 (bullish)
    sentiment_label: SentimentLabel
    relevance_score: float = 1.0  # 0.0 to 1.0


@dataclass
class SentimentData:
    """Aggregated sentiment data for a symbol."""

    symbol: str
    timestamp: datetime
    articles: list[NewsArticle] = field(default_factory=list)
    overall_score: float = 0.0  # -1.0 to 1.0
    overall_label: SentimentLabel = SentimentLabel.NEUTRAL
    bullish_count: int = 0
    bearish_count: int = 0
    neutral_count: int = 0
    article_count: int = 0

    @property
    def bullish_ratio(self) -> float:
        """Ratio of bullish articles."""
        if self.article_count == 0:
            return 0.0
        return self.bullish_count / self.article_count

    @property
    def bearish_ratio(self) -> float:
        """Ratio of bearish articles."""
        if self.article_count == 0:
            return 0.0
        return self.bearish_count / self.article_count

    @property
    def sentiment_strength(self) -> float:
        """Absolute strength of sentiment (0.0 to 1.0)."""
        return abs(self.overall_score)

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"{self.symbol}: {self.overall_label.value} "
            f"(score={self.overall_score:.2f}, "
            f"bullish={self.bullish_count}, "
            f"bearish={self.bearish_count}, "
            f"neutral={self.neutral_count})"
        )


class BaseSentimentFetcher(ABC):
    """Abstract base class for sentiment data fetchers."""

    @abstractmethod
    async def fetch_sentiment(
        self,
        symbol: str,
        lookback_hours: int = 24,
    ) -> SentimentData:
        """
        Fetch sentiment data for a symbol.

        Args:
            symbol: Stock symbol (e.g., "AAPL")
            lookback_hours: Hours of historical news to analyze

        Returns:
            Aggregated sentiment data
        """
        pass

    @abstractmethod
    async def fetch_news(
        self,
        symbol: str,
        limit: int = 50,
        lookback_hours: int = 24,
    ) -> list[NewsArticle]:
        """
        Fetch recent news articles for a symbol.

        Args:
            symbol: Stock symbol
            limit: Maximum number of articles
            lookback_hours: Hours of historical news

        Returns:
            List of news articles with sentiment
        """
        pass


class AlphaVantageSentimentFetcher(BaseSentimentFetcher):
    """
    Fetch sentiment data from Alpha Vantage News API.

    Alpha Vantage provides free news sentiment scores powered by AI/ML.
    Free tier: 25 requests/day, paid tiers available.
    """

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str) -> None:
        """
        Initialize with Alpha Vantage API key.

        Args:
            api_key: Alpha Vantage API key (get free at alphavantage.co)
        """
        self.api_key = api_key
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Lazy-initialize HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _parse_sentiment_label(self, label: str) -> SentimentLabel:
        """Parse Alpha Vantage sentiment label."""
        label_map = {
            "Bullish": SentimentLabel.BULLISH,
            "Somewhat-Bullish": SentimentLabel.SOMEWHAT_BULLISH,
            "Neutral": SentimentLabel.NEUTRAL,
            "Somewhat-Bearish": SentimentLabel.SOMEWHAT_BEARISH,
            "Bearish": SentimentLabel.BEARISH,
        }
        return label_map.get(label, SentimentLabel.NEUTRAL)

    def _calculate_overall_label(self, score: float) -> SentimentLabel:
        """Convert numeric score to sentiment label."""
        if score >= 0.35:
            return SentimentLabel.BULLISH
        elif score >= 0.15:
            return SentimentLabel.SOMEWHAT_BULLISH
        elif score >= -0.15:
            return SentimentLabel.NEUTRAL
        elif score >= -0.35:
            return SentimentLabel.SOMEWHAT_BEARISH
        else:
            return SentimentLabel.BEARISH

    async def fetch_news(
        self,
        symbol: str,
        limit: int = 50,
        lookback_hours: int = 24,
    ) -> list[NewsArticle]:
        """Fetch news articles from Alpha Vantage."""
        # Calculate time range
        time_to = datetime.now()
        time_from = time_to - timedelta(hours=lookback_hours)

        params: dict[str, str | int] = {
            "function": "NEWS_SENTIMENT",
            "tickers": symbol,
            "time_from": time_from.strftime("%Y%m%dT%H%M"),
            "time_to": time_to.strftime("%Y%m%dT%H%M"),
            "limit": limit,
            "apikey": self.api_key,
        }

        logger.debug(f"Fetching news for {symbol} from Alpha Vantage")

        try:
            response = await self.client.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as e:
            logger.error(f"Failed to fetch news from Alpha Vantage: {e}")
            return []

        # Check for API errors
        if "Error Message" in data or "Note" in data:
            error_msg = data.get("Error Message") or data.get("Note", "Unknown error")
            logger.warning(f"Alpha Vantage API error: {error_msg}")
            return []

        articles = []
        feed = data.get("feed", [])

        for item in feed:
            try:
                # Find sentiment for our specific symbol
                ticker_sentiment = None
                for ts in item.get("ticker_sentiment", []):
                    if ts.get("ticker") == symbol:
                        ticker_sentiment = ts
                        break

                if ticker_sentiment is None:
                    # Use overall sentiment if symbol-specific not found
                    sentiment_score = float(item.get("overall_sentiment_score", 0))
                    sentiment_label = self._parse_sentiment_label(
                        item.get("overall_sentiment_label", "Neutral")
                    )
                    relevance = 0.5
                else:
                    sentiment_score = float(
                        ticker_sentiment.get("ticker_sentiment_score", 0)
                    )
                    sentiment_label = self._parse_sentiment_label(
                        ticker_sentiment.get("ticker_sentiment_label", "Neutral")
                    )
                    relevance = float(ticker_sentiment.get("relevance_score", 0.5))

                # Parse published time
                time_published = item.get("time_published", "")
                if time_published:
                    published_at = datetime.strptime(time_published, "%Y%m%dT%H%M%S")
                else:
                    published_at = datetime.now()

                # Extract mentioned symbols
                symbols = [
                    ts.get("ticker", "")
                    for ts in item.get("ticker_sentiment", [])
                    if ts.get("ticker")
                ]

                articles.append(
                    NewsArticle(
                        title=item.get("title", ""),
                        summary=item.get("summary", ""),
                        source=item.get("source", "Unknown"),
                        url=item.get("url", ""),
                        published_at=published_at,
                        symbols=symbols,
                        sentiment_score=sentiment_score,
                        sentiment_label=sentiment_label,
                        relevance_score=relevance,
                    )
                )
            except (KeyError, ValueError) as e:
                logger.warning(f"Failed to parse news article: {e}")
                continue

        logger.info(f"Fetched {len(articles)} news articles for {symbol}")
        return articles

    async def fetch_sentiment(
        self,
        symbol: str,
        lookback_hours: int = 24,
    ) -> SentimentData:
        """Fetch and aggregate sentiment for a symbol."""
        articles = await self.fetch_news(symbol, lookback_hours=lookback_hours)

        if not articles:
            return SentimentData(
                symbol=symbol,
                timestamp=datetime.now(),
            )

        # Count by sentiment type
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        weighted_score_sum = 0.0
        weight_sum = 0.0

        for article in articles:
            # Weight by relevance
            weight = article.relevance_score
            weighted_score_sum += article.sentiment_score * weight
            weight_sum += weight

            if article.sentiment_label in (
                SentimentLabel.BULLISH,
                SentimentLabel.SOMEWHAT_BULLISH,
            ):
                bullish_count += 1
            elif article.sentiment_label in (
                SentimentLabel.BEARISH,
                SentimentLabel.SOMEWHAT_BEARISH,
            ):
                bearish_count += 1
            else:
                neutral_count += 1

        # Calculate weighted average score
        overall_score = weighted_score_sum / weight_sum if weight_sum > 0 else 0.0

        return SentimentData(
            symbol=symbol,
            timestamp=datetime.now(),
            articles=articles,
            overall_score=overall_score,
            overall_label=self._calculate_overall_label(overall_score),
            bullish_count=bullish_count,
            bearish_count=bearish_count,
            neutral_count=neutral_count,
            article_count=len(articles),
        )


class MockSentimentFetcher(BaseSentimentFetcher):
    """
    Mock sentiment fetcher for testing without API credentials.

    Generates random sentiment data based on symbol characteristics.
    """

    def __init__(self, seed: int | None = None) -> None:
        """Initialize with optional random seed."""
        import numpy as np

        self.rng = np.random.default_rng(seed)
        # Predefined biases for common symbols (for consistent testing)
        self._symbol_biases = {
            "AAPL": 0.15,  # Slightly bullish
            "TSLA": 0.25,  # More volatile/bullish
            "MSFT": 0.10,  # Slightly bullish
            "GOOGL": 0.05,  # Neutral-ish
            "AMZN": 0.12,  # Slightly bullish
            "META": -0.05,  # Slightly bearish
            "NVDA": 0.30,  # Bullish (AI hype)
        }

    def _get_bias(self, symbol: str) -> float:
        """Get sentiment bias for a symbol."""
        return self._symbol_biases.get(symbol, 0.0)

    def _generate_label(self, score: float) -> SentimentLabel:
        """Convert score to label."""
        if score >= 0.35:
            return SentimentLabel.BULLISH
        elif score >= 0.15:
            return SentimentLabel.SOMEWHAT_BULLISH
        elif score >= -0.15:
            return SentimentLabel.NEUTRAL
        elif score >= -0.35:
            return SentimentLabel.SOMEWHAT_BEARISH
        else:
            return SentimentLabel.BEARISH

    async def fetch_news(
        self,
        symbol: str,
        limit: int = 50,
        lookback_hours: int = 24,
    ) -> list[NewsArticle]:
        """Generate mock news articles."""
        bias = self._get_bias(symbol)
        num_articles = min(limit, int(self.rng.integers(5, 20)))

        articles = []
        now = datetime.now()

        headlines = [
            f"{symbol} Shows Strong Quarter Performance",
            f"Analysts Upgrade {symbol} Price Target",
            f"{symbol} Announces New Product Launch",
            f"Market Concerns Over {symbol} Growth",
            f"{symbol} Beats Earnings Expectations",
            f"Investors Cautious on {symbol} Outlook",
            f"{symbol} Partners with Tech Giant",
            f"Regulatory Scrutiny on {symbol} Intensifies",
            f"{symbol} Stock Reaches New Highs",
            f"Competition Heats Up for {symbol}",
        ]

        for i in range(num_articles):
            # Generate sentiment with bias
            score = float(self.rng.normal(bias, 0.3))
            score = max(-1.0, min(1.0, score))  # Clamp to [-1, 1]

            # Random time within lookback period
            hours_ago = self.rng.uniform(0, lookback_hours)
            published_at = now - timedelta(hours=hours_ago)

            articles.append(
                NewsArticle(
                    title=self.rng.choice(headlines),
                    summary=f"Mock article about {symbol} market activity.",
                    source=self.rng.choice(
                        ["Reuters", "Bloomberg", "CNBC", "WSJ", "MarketWatch"]
                    ),
                    url=f"https://example.com/news/{symbol.lower()}/{i}",
                    published_at=published_at,
                    symbols=[symbol],
                    sentiment_score=score,
                    sentiment_label=self._generate_label(score),
                    relevance_score=float(self.rng.uniform(0.5, 1.0)),
                )
            )

        return articles

    async def fetch_sentiment(
        self,
        symbol: str,
        lookback_hours: int = 24,
    ) -> SentimentData:
        """Generate mock sentiment data."""
        articles = await self.fetch_news(symbol, lookback_hours=lookback_hours)

        if not articles:
            return SentimentData(symbol=symbol, timestamp=datetime.now())

        bullish_count = sum(
            1
            for a in articles
            if a.sentiment_label
            in (SentimentLabel.BULLISH, SentimentLabel.SOMEWHAT_BULLISH)
        )
        bearish_count = sum(
            1
            for a in articles
            if a.sentiment_label
            in (SentimentLabel.BEARISH, SentimentLabel.SOMEWHAT_BEARISH)
        )
        neutral_count = len(articles) - bullish_count - bearish_count

        # Weighted average
        weight_sum = sum(a.relevance_score for a in articles)
        if weight_sum > 0:
            overall_score = (
                sum(a.sentiment_score * a.relevance_score for a in articles)
                / weight_sum
            )
        else:
            overall_score = 0.0

        return SentimentData(
            symbol=symbol,
            timestamp=datetime.now(),
            articles=articles,
            overall_score=overall_score,
            overall_label=self._generate_label(overall_score),
            bullish_count=bullish_count,
            bearish_count=bearish_count,
            neutral_count=neutral_count,
            article_count=len(articles),
        )


def get_sentiment_fetcher(settings: Settings | None = None) -> BaseSentimentFetcher:
    """
    Get the appropriate sentiment fetcher based on settings.

    Returns MockSentimentFetcher if no API credentials are configured,
    otherwise returns AlphaVantageSentimentFetcher.

    Args:
        settings: Application settings. If None, uses default settings.

    Returns:
        Sentiment fetcher instance
    """
    if settings is None:
        from trader.config.settings import get_settings

        settings = get_settings()

    api_key = getattr(settings, "alphavantage_api_key", None)
    if api_key:
        key_value = (
            api_key.get_secret_value() if hasattr(api_key, "get_secret_value") else api_key
        )
        if key_value:
            logger.info("Using Alpha Vantage sentiment fetcher")
            return AlphaVantageSentimentFetcher(api_key=key_value)

    logger.warning("No Alpha Vantage API key found, using mock sentiment fetcher")
    return MockSentimentFetcher()


async def fetch_multi_symbol_sentiment(
    symbols: list[str],
    fetcher: BaseSentimentFetcher | None = None,
    lookback_hours: int = 24,
) -> dict[str, SentimentData]:
    """
    Fetch sentiment for multiple symbols concurrently.

    Args:
        symbols: List of stock symbols
        fetcher: Sentiment fetcher to use (defaults to auto-detect)
        lookback_hours: Hours of historical news

    Returns:
        Dict mapping symbol to SentimentData
    """
    if fetcher is None:
        fetcher = get_sentiment_fetcher()

    tasks = [fetcher.fetch_sentiment(symbol, lookback_hours) for symbol in symbols]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    sentiment_data: dict[str, SentimentData] = {}
    for symbol, result in zip(symbols, results, strict=True):
        if isinstance(result, BaseException):
            logger.error(f"Failed to fetch sentiment for {symbol}: {result}")
            sentiment_data[symbol] = SentimentData(
                symbol=symbol, timestamp=datetime.now()
            )
        else:
            sentiment_data[symbol] = result

    return sentiment_data
