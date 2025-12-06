"""Tests for sentiment data fetching module."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trader.data.sentiment import (
    AlphaVantageSentimentFetcher,
    BaseSentimentFetcher,
    MockSentimentFetcher,
    NewsArticle,
    SentimentData,
    SentimentLabel,
    fetch_multi_symbol_sentiment,
    get_sentiment_fetcher,
)


class TestSentimentLabel:
    """Tests for SentimentLabel enum."""

    def test_label_values(self) -> None:
        """Test sentiment label values."""
        assert SentimentLabel.BULLISH.value == "Bullish"
        assert SentimentLabel.SOMEWHAT_BULLISH.value == "Somewhat-Bullish"
        assert SentimentLabel.NEUTRAL.value == "Neutral"
        assert SentimentLabel.SOMEWHAT_BEARISH.value == "Somewhat-Bearish"
        assert SentimentLabel.BEARISH.value == "Bearish"


class TestNewsArticle:
    """Tests for NewsArticle dataclass."""

    def test_news_article_creation(self) -> None:
        """Test creating a news article."""
        article = NewsArticle(
            title="Test Article",
            summary="This is a test summary",
            source="Reuters",
            url="https://example.com/article",
            published_at=datetime.now(),
            symbols=["AAPL", "MSFT"],
            sentiment_score=0.25,
            sentiment_label=SentimentLabel.SOMEWHAT_BULLISH,
            relevance_score=0.8,
        )

        assert article.title == "Test Article"
        assert article.source == "Reuters"
        assert len(article.symbols) == 2
        assert article.sentiment_score == 0.25
        assert article.sentiment_label == SentimentLabel.SOMEWHAT_BULLISH


class TestSentimentData:
    """Tests for SentimentData dataclass."""

    def test_sentiment_data_defaults(self) -> None:
        """Test default values for SentimentData."""
        data = SentimentData(
            symbol="AAPL",
            timestamp=datetime.now(),
        )

        assert data.symbol == "AAPL"
        assert data.overall_score == 0.0
        assert data.overall_label == SentimentLabel.NEUTRAL
        assert data.bullish_count == 0
        assert data.bearish_count == 0
        assert data.neutral_count == 0
        assert data.article_count == 0
        assert len(data.articles) == 0

    def test_bullish_ratio(self) -> None:
        """Test bullish ratio calculation."""
        data = SentimentData(
            symbol="AAPL",
            timestamp=datetime.now(),
            bullish_count=3,
            bearish_count=1,
            neutral_count=1,
            article_count=5,
        )

        assert data.bullish_ratio == 0.6  # 3/5

    def test_bearish_ratio(self) -> None:
        """Test bearish ratio calculation."""
        data = SentimentData(
            symbol="AAPL",
            timestamp=datetime.now(),
            bullish_count=1,
            bearish_count=2,
            neutral_count=2,
            article_count=5,
        )

        assert data.bearish_ratio == 0.4  # 2/5

    def test_ratios_with_no_articles(self) -> None:
        """Test ratios when no articles exist."""
        data = SentimentData(
            symbol="AAPL",
            timestamp=datetime.now(),
        )

        assert data.bullish_ratio == 0.0
        assert data.bearish_ratio == 0.0

    def test_sentiment_strength(self) -> None:
        """Test sentiment strength calculation."""
        bullish = SentimentData(
            symbol="AAPL",
            timestamp=datetime.now(),
            overall_score=0.5,
        )
        assert bullish.sentiment_strength == 0.5

        bearish = SentimentData(
            symbol="AAPL",
            timestamp=datetime.now(),
            overall_score=-0.3,
        )
        assert bearish.sentiment_strength == 0.3

    def test_summary(self) -> None:
        """Test summary string generation."""
        data = SentimentData(
            symbol="AAPL",
            timestamp=datetime.now(),
            overall_score=0.25,
            overall_label=SentimentLabel.SOMEWHAT_BULLISH,
            bullish_count=3,
            bearish_count=1,
            neutral_count=1,
        )

        summary = data.summary()
        assert "AAPL" in summary
        assert "Somewhat-Bullish" in summary
        assert "0.25" in summary


class TestMockSentimentFetcher:
    """Tests for MockSentimentFetcher."""

    @pytest.fixture
    def fetcher(self) -> MockSentimentFetcher:
        """Create a mock sentiment fetcher with fixed seed."""
        return MockSentimentFetcher(seed=42)

    @pytest.mark.asyncio
    async def test_fetch_news(self, fetcher: MockSentimentFetcher) -> None:
        """Test fetching mock news articles."""
        articles = await fetcher.fetch_news("AAPL", limit=10)

        assert len(articles) > 0
        assert len(articles) <= 10

        for article in articles:
            assert isinstance(article, NewsArticle)
            assert "AAPL" in article.symbols
            assert -1.0 <= article.sentiment_score <= 1.0
            assert article.relevance_score >= 0.5

    @pytest.mark.asyncio
    async def test_fetch_sentiment(self, fetcher: MockSentimentFetcher) -> None:
        """Test fetching aggregated sentiment."""
        sentiment = await fetcher.fetch_sentiment("AAPL", lookback_hours=24)

        assert isinstance(sentiment, SentimentData)
        assert sentiment.symbol == "AAPL"
        assert sentiment.article_count > 0
        assert -1.0 <= sentiment.overall_score <= 1.0

    @pytest.mark.asyncio
    async def test_symbol_bias(self, fetcher: MockSentimentFetcher) -> None:
        """Test that symbols have predefined biases."""
        # NVDA has a positive bias (AI hype)
        nvda_sentiment = await fetcher.fetch_sentiment("NVDA")

        # Unknown symbol should be neutral
        unknown_sentiment = await fetcher.fetch_sentiment("UNKNOWN")

        # Check biases are applied (NVDA should trend bullish)
        assert fetcher._get_bias("NVDA") > fetcher._get_bias("UNKNOWN")

    @pytest.mark.asyncio
    async def test_reproducibility_with_seed(self) -> None:
        """Test that same seed produces same results."""
        fetcher1 = MockSentimentFetcher(seed=123)
        fetcher2 = MockSentimentFetcher(seed=123)

        sentiment1 = await fetcher1.fetch_sentiment("AAPL")
        sentiment2 = await fetcher2.fetch_sentiment("AAPL")

        assert sentiment1.article_count == sentiment2.article_count
        assert abs(sentiment1.overall_score - sentiment2.overall_score) < 0.001


class TestAlphaVantageSentimentFetcher:
    """Tests for AlphaVantageSentimentFetcher."""

    @pytest.fixture
    def fetcher(self) -> AlphaVantageSentimentFetcher:
        """Create an Alpha Vantage fetcher with test key."""
        return AlphaVantageSentimentFetcher(api_key="test_key")

    def test_initialization(self, fetcher: AlphaVantageSentimentFetcher) -> None:
        """Test fetcher initialization."""
        assert fetcher.api_key == "test_key"
        assert fetcher._client is None

    def test_parse_sentiment_label(
        self, fetcher: AlphaVantageSentimentFetcher
    ) -> None:
        """Test parsing sentiment labels."""
        assert fetcher._parse_sentiment_label("Bullish") == SentimentLabel.BULLISH
        assert (
            fetcher._parse_sentiment_label("Somewhat-Bullish")
            == SentimentLabel.SOMEWHAT_BULLISH
        )
        assert fetcher._parse_sentiment_label("Neutral") == SentimentLabel.NEUTRAL
        assert (
            fetcher._parse_sentiment_label("Unknown") == SentimentLabel.NEUTRAL
        )

    def test_calculate_overall_label(
        self, fetcher: AlphaVantageSentimentFetcher
    ) -> None:
        """Test calculating overall label from score."""
        assert fetcher._calculate_overall_label(0.5) == SentimentLabel.BULLISH
        assert fetcher._calculate_overall_label(0.2) == SentimentLabel.SOMEWHAT_BULLISH
        assert fetcher._calculate_overall_label(0.0) == SentimentLabel.NEUTRAL
        assert fetcher._calculate_overall_label(-0.2) == SentimentLabel.SOMEWHAT_BEARISH
        assert fetcher._calculate_overall_label(-0.5) == SentimentLabel.BEARISH

    @pytest.mark.asyncio
    async def test_fetch_news_api_error(
        self, fetcher: AlphaVantageSentimentFetcher
    ) -> None:
        """Test handling API errors."""
        import httpx

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.HTTPError("API error"))
        fetcher._client = mock_client

        articles = await fetcher.fetch_news("AAPL")
        assert articles == []

    @pytest.mark.asyncio
    async def test_fetch_news_rate_limit(
        self, fetcher: AlphaVantageSentimentFetcher
    ) -> None:
        """Test handling rate limit response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"Note": "Rate limit exceeded"}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        fetcher._client = mock_client

        articles = await fetcher.fetch_news("AAPL")
        assert articles == []

    @pytest.mark.asyncio
    async def test_fetch_news_success(
        self, fetcher: AlphaVantageSentimentFetcher
    ) -> None:
        """Test successful news fetch."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "feed": [
                {
                    "title": "Apple Beats Earnings",
                    "summary": "Strong quarter for Apple",
                    "source": "Reuters",
                    "url": "https://example.com/1",
                    "time_published": "20231215T120000",
                    "ticker_sentiment": [
                        {
                            "ticker": "AAPL",
                            "ticker_sentiment_score": "0.35",
                            "ticker_sentiment_label": "Bullish",
                            "relevance_score": "0.9",
                        }
                    ],
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        fetcher._client = mock_client

        articles = await fetcher.fetch_news("AAPL")

        assert len(articles) == 1
        assert articles[0].title == "Apple Beats Earnings"
        assert articles[0].sentiment_score == 0.35
        assert articles[0].sentiment_label == SentimentLabel.BULLISH


class TestGetSentimentFetcher:
    """Tests for get_sentiment_fetcher factory function."""

    def test_returns_mock_without_credentials(self) -> None:
        """Test that mock fetcher is returned without API key."""
        mock_settings = MagicMock()
        mock_settings.alphavantage_api_key = MagicMock()
        mock_settings.alphavantage_api_key.get_secret_value.return_value = ""

        fetcher = get_sentiment_fetcher(mock_settings)
        assert isinstance(fetcher, MockSentimentFetcher)

    def test_returns_alphavantage_with_credentials(self) -> None:
        """Test that Alpha Vantage fetcher is returned with API key."""
        mock_settings = MagicMock()
        mock_settings.alphavantage_api_key = MagicMock()
        mock_settings.alphavantage_api_key.get_secret_value.return_value = "test_key"

        fetcher = get_sentiment_fetcher(mock_settings)
        assert isinstance(fetcher, AlphaVantageSentimentFetcher)


class TestFetchMultiSymbolSentiment:
    """Tests for fetch_multi_symbol_sentiment function."""

    @pytest.mark.asyncio
    async def test_fetch_multiple_symbols(self) -> None:
        """Test fetching sentiment for multiple symbols."""
        fetcher = MockSentimentFetcher(seed=42)
        symbols = ["AAPL", "MSFT", "GOOGL"]

        result = await fetch_multi_symbol_sentiment(symbols, fetcher)

        assert len(result) == 3
        assert all(s in result for s in symbols)
        assert all(isinstance(v, SentimentData) for v in result.values())

    @pytest.mark.asyncio
    async def test_handles_exceptions(self) -> None:
        """Test that exceptions are handled gracefully."""

        class FailingFetcher(BaseSentimentFetcher):
            async def fetch_sentiment(
                self, symbol: str, lookback_hours: int = 24
            ) -> SentimentData:
                if symbol == "FAIL":
                    raise ValueError("Failed to fetch")
                return SentimentData(symbol=symbol, timestamp=datetime.now())

            async def fetch_news(
                self, symbol: str, limit: int = 50, lookback_hours: int = 24
            ) -> list[NewsArticle]:
                return []

        fetcher = FailingFetcher()
        symbols = ["AAPL", "FAIL", "MSFT"]

        result = await fetch_multi_symbol_sentiment(symbols, fetcher)

        # All symbols should have entries
        assert len(result) == 3
        # FAIL should have default empty SentimentData
        assert result["FAIL"].article_count == 0
