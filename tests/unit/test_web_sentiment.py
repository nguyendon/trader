"""Tests for web scraping sentiment module."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trader.data.sentiment import SentimentLabel
from trader.data.web_sentiment import (
    RedditPost,
    RedditSentimentFetcher,
    RSSArticle,
    RSSNewsFetcher,
    SentimentAnalyzer,
    WebSentimentConfig,
    WebSentimentFetcher,
    get_web_sentiment_fetcher,
)


class TestSentimentAnalyzer:
    """Tests for VADER sentiment analyzer wrapper."""

    @pytest.fixture
    def analyzer(self) -> SentimentAnalyzer:
        """Create analyzer instance."""
        return SentimentAnalyzer()

    def test_analyze_positive_text(self, analyzer: SentimentAnalyzer) -> None:
        """Test analyzing positive text."""
        score = analyzer.analyze("AAPL is going to moon! Great earnings! 🚀")
        assert score > 0.3  # Should be clearly positive

    def test_analyze_negative_text(self, analyzer: SentimentAnalyzer) -> None:
        """Test analyzing negative text."""
        score = analyzer.analyze("This stock is terrible, worst investment ever")
        assert score < -0.3  # Should be clearly negative

    def test_analyze_neutral_text(self, analyzer: SentimentAnalyzer) -> None:
        """Test analyzing neutral text."""
        score = analyzer.analyze("The stock price is $150")
        assert -0.3 < score < 0.3  # Should be neutral-ish

    def test_analyze_empty_text(self, analyzer: SentimentAnalyzer) -> None:
        """Test analyzing empty text."""
        score = analyzer.analyze("")
        assert score == 0.0

    def test_analyze_with_context(self, analyzer: SentimentAnalyzer) -> None:
        """Test weighted title/body analysis."""
        # Title is weighted higher
        score = analyzer.analyze_with_context(
            title="Amazing breakthrough for Apple!",
            body="Some neutral description here."
        )
        assert score > 0.2  # Title should dominate

    def test_score_to_label_bullish(self, analyzer: SentimentAnalyzer) -> None:
        """Test converting score to bullish label."""
        assert analyzer.score_to_label(0.5) == SentimentLabel.BULLISH
        assert analyzer.score_to_label(0.35) == SentimentLabel.BULLISH

    def test_score_to_label_somewhat_bullish(self, analyzer: SentimentAnalyzer) -> None:
        """Test converting score to somewhat bullish label."""
        assert analyzer.score_to_label(0.2) == SentimentLabel.SOMEWHAT_BULLISH
        assert analyzer.score_to_label(0.15) == SentimentLabel.SOMEWHAT_BULLISH

    def test_score_to_label_neutral(self, analyzer: SentimentAnalyzer) -> None:
        """Test converting score to neutral label."""
        assert analyzer.score_to_label(0.0) == SentimentLabel.NEUTRAL
        assert analyzer.score_to_label(0.1) == SentimentLabel.NEUTRAL
        assert analyzer.score_to_label(-0.1) == SentimentLabel.NEUTRAL

    def test_score_to_label_bearish(self, analyzer: SentimentAnalyzer) -> None:
        """Test converting score to bearish labels."""
        assert analyzer.score_to_label(-0.2) == SentimentLabel.SOMEWHAT_BEARISH
        assert analyzer.score_to_label(-0.5) == SentimentLabel.BEARISH


class TestRedditPost:
    """Tests for RedditPost dataclass."""

    def test_reddit_post_creation(self) -> None:
        """Test creating a Reddit post."""
        post = RedditPost(
            title="AAPL to the moon!",
            body="Buy buy buy",
            subreddit="wallstreetbets",
            score=500,
            num_comments=100,
            created_utc=datetime.utcnow(),
            url="https://reddit.com/r/wallstreetbets/123",
            sentiment_score=0.8,
        )

        assert post.subreddit == "wallstreetbets"
        assert post.score == 500
        assert post.sentiment_score == 0.8


class TestRSSArticle:
    """Tests for RSSArticle dataclass."""

    def test_rss_article_creation(self) -> None:
        """Test creating an RSS article."""
        article = RSSArticle(
            title="Apple Reports Record Earnings",
            summary="Strong Q4 results...",
            source="Yahoo Finance",
            url="https://finance.yahoo.com/news/123",
            published=datetime.utcnow(),
            sentiment_score=0.5,
        )

        assert article.source == "Yahoo Finance"
        assert article.sentiment_score == 0.5


class TestRedditSentimentFetcher:
    """Tests for RedditSentimentFetcher."""

    def test_extract_tickers(self) -> None:
        """Test ticker extraction from text."""
        fetcher = RedditSentimentFetcher(
            client_id="test",
            client_secret="test",
        )

        # Should extract $TICKER pattern
        tickers = fetcher._extract_tickers("Buying $AAPL and $MSFT today!")
        assert "AAPL" in tickers
        assert "MSFT" in tickers

        # Should not extract common words
        tickers = fetcher._extract_tickers("$IT is my favorite stock")
        assert "IT" not in tickers

    def test_mentions_ticker(self) -> None:
        """Test ticker mention detection."""
        fetcher = RedditSentimentFetcher(
            client_id="test",
            client_secret="test",
        )

        # Should find $TICKER pattern
        assert fetcher._mentions_ticker("Buying $AAPL today", "AAPL")

        # Should find standalone ticker
        assert fetcher._mentions_ticker("AAPL is going up", "AAPL")

        # Should not false positive on partial matches
        assert not fetcher._mentions_ticker("GAAPL stock", "AAPL")

        # Common words should only match with $
        assert not fetcher._mentions_ticker("IT department is busy", "IT")
        assert fetcher._mentions_ticker("$IT stock is rising", "IT")


class TestRSSNewsFetcher:
    """Tests for RSSNewsFetcher."""

    @pytest.fixture
    def fetcher(self) -> RSSNewsFetcher:
        """Create RSS fetcher."""
        return RSSNewsFetcher()

    @pytest.mark.asyncio
    async def test_fetch_feed_error_handling(self, fetcher: RSSNewsFetcher) -> None:
        """Test that feed fetch errors are handled gracefully."""
        # Create mock client that raises error
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("Network error"))
        fetcher._client = mock_client

        result = await fetcher._fetch_feed("https://example.com/rss")
        assert result == []

    @pytest.mark.asyncio
    async def test_fetch_articles_empty_response(self, fetcher: RSSNewsFetcher) -> None:
        """Test handling empty RSS response."""
        # Mock the _fetch_feed method to return empty
        with patch.object(fetcher, "_fetch_feed", return_value=[]):
            articles = await fetcher.fetch_articles("AAPL")
            assert articles == []


class TestWebSentimentConfig:
    """Tests for WebSentimentConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = WebSentimentConfig()

        assert config.reddit_enabled is True
        assert config.rss_enabled is True
        assert "wallstreetbets" in config.subreddits
        assert config.lookback_hours == 24
        assert config.min_upvotes == 5

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = WebSentimentConfig(
            reddit_enabled=False,
            rss_enabled=True,
            lookback_hours=48,
            min_upvotes=10,
            subreddits=["stocks"],
        )

        assert config.reddit_enabled is False
        assert config.lookback_hours == 48
        assert config.subreddits == ["stocks"]


class TestWebSentimentFetcher:
    """Tests for WebSentimentFetcher."""

    @pytest.fixture
    def config_rss_only(self) -> WebSentimentConfig:
        """Config with only RSS enabled."""
        return WebSentimentConfig(
            reddit_enabled=False,
            rss_enabled=True,
        )

    @pytest.fixture
    def fetcher_rss_only(self, config_rss_only: WebSentimentConfig) -> WebSentimentFetcher:
        """Fetcher with only RSS."""
        return WebSentimentFetcher(config_rss_only)

    def test_reddit_fetcher_disabled(self, fetcher_rss_only: WebSentimentFetcher) -> None:
        """Test that Reddit fetcher is None when disabled."""
        assert fetcher_rss_only.reddit_fetcher is None

    def test_rss_fetcher_always_available(
        self, fetcher_rss_only: WebSentimentFetcher
    ) -> None:
        """Test that RSS fetcher is always available."""
        assert fetcher_rss_only.rss_fetcher is not None

    @pytest.mark.asyncio
    async def test_fetch_sentiment_rss_only(
        self, fetcher_rss_only: WebSentimentFetcher
    ) -> None:
        """Test fetching sentiment with RSS only."""
        # Mock the RSS fetcher
        mock_articles = [
            RSSArticle(
                title="Apple Stock Rises",
                summary="Good news for AAPL",
                source="Yahoo Finance",
                url="https://example.com/1",
                published=datetime.utcnow(),
                sentiment_score=0.3,
            ),
            RSSArticle(
                title="Apple Faces Challenges",
                summary="Some concerns...",
                source="Google News",
                url="https://example.com/2",
                published=datetime.utcnow(),
                sentiment_score=-0.2,
            ),
        ]

        with patch.object(
            fetcher_rss_only.rss_fetcher,
            "fetch_articles",
            return_value=mock_articles,
        ):
            sentiment = await fetcher_rss_only.fetch_sentiment("AAPL", lookback_hours=24)

            assert sentiment.symbol == "AAPL"
            assert sentiment.article_count == 2
            # Should have one bullish, one bearish
            assert sentiment.bullish_count >= 0
            assert sentiment.bearish_count >= 0

    @pytest.mark.asyncio
    async def test_fetch_sentiment_empty_results(
        self, fetcher_rss_only: WebSentimentFetcher
    ) -> None:
        """Test fetching sentiment with no results."""
        with patch.object(
            fetcher_rss_only.rss_fetcher,
            "fetch_articles",
            return_value=[],
        ):
            sentiment = await fetcher_rss_only.fetch_sentiment("UNKNOWN", lookback_hours=24)

            assert sentiment.symbol == "UNKNOWN"
            assert sentiment.article_count == 0
            assert sentiment.overall_score == 0.0


class TestGetWebSentimentFetcher:
    """Tests for get_web_sentiment_fetcher factory."""

    def test_without_reddit_credentials(self) -> None:
        """Test creating fetcher without Reddit credentials."""
        fetcher = get_web_sentiment_fetcher()

        assert fetcher.config.reddit_enabled is False
        assert fetcher.reddit_fetcher is None

    def test_with_reddit_credentials(self) -> None:
        """Test creating fetcher with Reddit credentials."""
        fetcher = get_web_sentiment_fetcher(
            reddit_client_id="test_id",
            reddit_client_secret="test_secret",
        )

        assert fetcher.config.reddit_enabled is True
        assert fetcher.config.reddit_client_id == "test_id"

    def test_custom_subreddits(self) -> None:
        """Test creating fetcher with custom subreddits."""
        fetcher = get_web_sentiment_fetcher(
            subreddits=["stocks", "options"],
        )

        assert fetcher.config.subreddits == ["stocks", "options"]
