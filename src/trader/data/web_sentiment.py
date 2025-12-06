"""Web scraping sentiment fetcher using Reddit, RSS feeds, and local NLP."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import feedparser
import httpx
from loguru import logger
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from trader.data.sentiment import (
    BaseSentimentFetcher,
    NewsArticle,
    SentimentData,
    SentimentLabel,
)

if TYPE_CHECKING:
    import praw


# Common stock tickers to avoid false positives (e.g., "IT" the movie, "A" grade)
COMMON_WORD_TICKERS = {"A", "I", "IT", "AT", "BE", "GO", "SO", "AN", "AS", "OR", "ON"}


@dataclass
class RedditPost:
    """A Reddit post or comment with sentiment."""

    title: str
    body: str
    subreddit: str
    score: int  # upvotes - downvotes
    num_comments: int
    created_utc: datetime
    url: str
    sentiment_score: float = 0.0


@dataclass
class RSSArticle:
    """An RSS feed article."""

    title: str
    summary: str
    source: str
    url: str
    published: datetime
    sentiment_score: float = 0.0


@dataclass
class WebSentimentConfig:
    """Configuration for web sentiment fetching."""

    # Reddit settings
    reddit_enabled: bool = True
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "trader-sentiment-bot/1.0"
    subreddits: list[str] = field(
        default_factory=lambda: ["wallstreetbets", "stocks", "investing", "stockmarket"]
    )
    reddit_post_limit: int = 50

    # RSS settings
    rss_enabled: bool = True
    rss_feeds: list[str] = field(default_factory=list)

    # General settings
    lookback_hours: int = 24
    min_upvotes: int = 5  # Filter low-quality posts


class SentimentAnalyzer:
    """Local sentiment analysis using VADER."""

    def __init__(self) -> None:
        """Initialize VADER sentiment analyzer."""
        self._analyzer: SentimentIntensityAnalyzer | None = None

    @property
    def analyzer(self) -> SentimentIntensityAnalyzer:
        """Lazy-initialize analyzer."""
        if self._analyzer is None:
            self._analyzer = SentimentIntensityAnalyzer()
        return self._analyzer

    def analyze(self, text: str) -> float:
        """
        Analyze sentiment of text.

        Args:
            text: Text to analyze

        Returns:
            Compound score from -1.0 (bearish) to 1.0 (bullish)
        """
        if not text:
            return 0.0

        scores = self.analyzer.polarity_scores(text)
        return float(scores["compound"])

    def analyze_with_context(self, title: str, body: str = "") -> float:
        """
        Analyze sentiment with title weighted more heavily.

        Args:
            title: Title/headline (weighted 2x)
            body: Body text

        Returns:
            Weighted compound score
        """
        title_score = self.analyze(title)
        body_score = self.analyze(body) if body else 0.0

        # Title is usually more informative, weight it higher
        if body:
            return (title_score * 2 + body_score) / 3
        return title_score

    def score_to_label(self, score: float) -> SentimentLabel:
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


class RedditSentimentFetcher:
    """Fetch sentiment from Reddit using PRAW."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        user_agent: str = "trader-sentiment-bot/1.0",
    ) -> None:
        """
        Initialize Reddit fetcher.

        Args:
            client_id: Reddit API client ID
            client_secret: Reddit API client secret
            user_agent: User agent string for API requests
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        self._reddit: praw.Reddit | None = None
        self._analyzer = SentimentAnalyzer()

    @property
    def reddit(self) -> praw.Reddit:
        """Lazy-initialize Reddit client."""
        if self._reddit is None:
            import praw

            self._reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.user_agent,
            )
        return self._reddit

    def _extract_tickers(self, text: str) -> set[str]:
        """
        Extract stock tickers from text.

        Looks for $TICKER pattern and TICKER mentions.
        """
        tickers = set()

        # Match $TICKER pattern (most reliable)
        dollar_pattern = r"\$([A-Z]{1,5})\b"
        for match in re.finditer(dollar_pattern, text.upper()):
            ticker = match.group(1)
            if ticker not in COMMON_WORD_TICKERS:
                tickers.add(ticker)

        return tickers

    def _mentions_ticker(self, text: str, symbol: str) -> bool:
        """Check if text mentions the given ticker."""
        text_upper = text.upper()

        # Check for $TICKER pattern
        if f"${symbol}" in text_upper:
            return True

        # Check for ticker as standalone word (avoid partial matches)
        pattern = rf"\b{symbol}\b"
        if re.search(pattern, text_upper):
            # Avoid false positives for common words
            if symbol in COMMON_WORD_TICKERS:
                # Only match if preceded by $ for common words
                return f"${symbol}" in text_upper
            return True

        return False

    async def fetch_posts(
        self,
        symbol: str,
        subreddits: list[str],
        limit: int = 50,
        lookback_hours: int = 24,
        min_upvotes: int = 5,
    ) -> list[RedditPost]:
        """
        Fetch Reddit posts mentioning a symbol.

        Args:
            symbol: Stock ticker to search for
            subreddits: Subreddits to search
            limit: Max posts per subreddit
            lookback_hours: Only include posts from this many hours ago
            min_upvotes: Minimum upvotes to include

        Returns:
            List of RedditPost objects
        """
        posts = []
        cutoff = datetime.utcnow() - timedelta(hours=lookback_hours)

        for subreddit_name in subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)

                # Search for ticker mentions
                search_queries = [f"${symbol}", symbol]

                for query in search_queries:
                    try:
                        # Use search for better results
                        for submission in subreddit.search(
                            query, sort="new", time_filter="day", limit=limit
                        ):
                            # Check if within time window
                            created = datetime.utcfromtimestamp(submission.created_utc)
                            if created < cutoff:
                                continue

                            # Check upvotes
                            if submission.score < min_upvotes:
                                continue

                            # Verify ticker is actually mentioned
                            full_text = f"{submission.title} {submission.selftext}"
                            if not self._mentions_ticker(full_text, symbol):
                                continue

                            # Analyze sentiment
                            sentiment = self._analyzer.analyze_with_context(
                                submission.title, submission.selftext
                            )

                            posts.append(
                                RedditPost(
                                    title=submission.title,
                                    body=submission.selftext[:500],  # Truncate
                                    subreddit=subreddit_name,
                                    score=submission.score,
                                    num_comments=submission.num_comments,
                                    created_utc=created,
                                    url=f"https://reddit.com{submission.permalink}",
                                    sentiment_score=sentiment,
                                )
                            )
                    except Exception as e:
                        logger.warning(f"Error searching r/{subreddit_name}: {e}")
                        continue

            except Exception as e:
                logger.warning(f"Error fetching from r/{subreddit_name}: {e}")
                continue

        # Sort by score (popularity) and deduplicate by URL
        seen_urls = set()
        unique_posts = []
        for post in sorted(posts, key=lambda p: p.score, reverse=True):
            if post.url not in seen_urls:
                seen_urls.add(post.url)
                unique_posts.append(post)

        logger.info(f"Fetched {len(unique_posts)} Reddit posts for {symbol}")
        return unique_posts


class RSSNewsFetcher:
    """Fetch news from RSS feeds."""

    # Default RSS feeds for financial news
    DEFAULT_FEEDS = [
        # Google News (search-based, most reliable)
        "https://news.google.com/rss/search?q={symbol}+stock&hl=en-US&gl=US&ceid=US:en",
        # Seeking Alpha (if available)
        "https://seekingalpha.com/api/sa/combined/{symbol}.xml",
    ]

    # General market news feeds
    MARKET_FEEDS = [
        "https://www.investing.com/rss/news.rss",
        "https://feeds.marketwatch.com/marketwatch/topstories/",
    ]

    def __init__(self, custom_feeds: list[str] | None = None) -> None:
        """
        Initialize RSS fetcher.

        Args:
            custom_feeds: Additional RSS feed URLs (use {symbol} placeholder)
        """
        self.custom_feeds = custom_feeds or []
        self._analyzer = SentimentAnalyzer()
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Lazy-initialize HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=15.0)
        return self._client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _fetch_feed(self, url: str) -> list[dict]:  # type: ignore[type-arg]
        """Fetch and parse a single RSS feed."""
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            feed = feedparser.parse(response.text)
            return list(feed.entries)  # type: ignore[return-value]
        except Exception as e:
            logger.warning(f"Error fetching RSS feed {url}: {e}")
            return []

    async def fetch_articles(
        self,
        symbol: str,
        lookback_hours: int = 24,
        limit: int = 50,
    ) -> list[RSSArticle]:
        """
        Fetch news articles for a symbol.

        Args:
            symbol: Stock ticker
            lookback_hours: Only include articles from this time window
            limit: Max articles to return

        Returns:
            List of RSSArticle objects
        """
        articles = []
        cutoff = datetime.utcnow() - timedelta(hours=lookback_hours)

        # Build feed URLs for this symbol
        feed_urls = [
            feed.format(symbol=symbol)
            for feed in self.DEFAULT_FEEDS + self.custom_feeds
        ]

        # Fetch all feeds concurrently
        tasks = [self._fetch_feed(url) for url in feed_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for entries in results:
            if isinstance(entries, BaseException):
                continue

            for entry in entries:
                try:
                    # Parse published date
                    published = None
                    if hasattr(entry, "published_parsed") and entry.published_parsed:
                        published = datetime(*entry.published_parsed[:6])
                    elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                        published = datetime(*entry.updated_parsed[:6])
                    else:
                        published = datetime.utcnow()

                    # Check if within time window
                    if published < cutoff:
                        continue

                    title = entry.get("title", "")
                    summary = entry.get("summary", entry.get("description", ""))

                    # Clean HTML from summary
                    summary = re.sub(r"<[^>]+>", "", summary)[:500]

                    # Check if article mentions the symbol
                    full_text = f"{title} {summary}"
                    if symbol.upper() not in full_text.upper():
                        continue

                    # Analyze sentiment
                    sentiment = self._analyzer.analyze_with_context(title, summary)

                    # Extract source from feed
                    source = entry.get("source", {}).get("title", "Unknown")
                    if source == "Unknown":
                        # Try to get from URL
                        link = entry.get("link", "")
                        if "yahoo" in link:
                            source = "Yahoo Finance"
                        elif "google" in link:
                            source = "Google News"
                        elif "reuters" in link:
                            source = "Reuters"
                        elif "bloomberg" in link:
                            source = "Bloomberg"

                    articles.append(
                        RSSArticle(
                            title=title,
                            summary=summary,
                            source=source,
                            url=entry.get("link", ""),
                            published=published,
                            sentiment_score=sentiment,
                        )
                    )

                except Exception as e:
                    logger.debug(f"Error parsing RSS entry: {e}")
                    continue

        # Sort by date and deduplicate
        seen_titles = set()
        unique_articles = []
        for article in sorted(articles, key=lambda a: a.published, reverse=True):
            # Simple dedup by title similarity
            title_key = article.title.lower()[:50]
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_articles.append(article)

        logger.info(f"Fetched {len(unique_articles)} RSS articles for {symbol}")
        return unique_articles[:limit]


class WebSentimentFetcher(BaseSentimentFetcher):
    """
    Combined web sentiment fetcher using Reddit and RSS feeds.

    This fetcher scrapes publicly available sources and uses VADER
    for local sentiment analysis - no API keys required for RSS feeds.
    Reddit requires free API credentials.
    """

    def __init__(self, config: WebSentimentConfig | None = None) -> None:
        """
        Initialize web sentiment fetcher.

        Args:
            config: Configuration options
        """
        self.config = config or WebSentimentConfig()
        self._analyzer = SentimentAnalyzer()
        self._reddit_fetcher: RedditSentimentFetcher | None = None
        self._rss_fetcher: RSSNewsFetcher | None = None

    @property
    def reddit_fetcher(self) -> RedditSentimentFetcher | None:
        """Get Reddit fetcher if configured."""
        if not self.config.reddit_enabled:
            return None
        if not self.config.reddit_client_id or not self.config.reddit_client_secret:
            return None
        if self._reddit_fetcher is None:
            self._reddit_fetcher = RedditSentimentFetcher(
                client_id=self.config.reddit_client_id,
                client_secret=self.config.reddit_client_secret,
                user_agent=self.config.reddit_user_agent,
            )
        return self._reddit_fetcher

    @property
    def rss_fetcher(self) -> RSSNewsFetcher:
        """Get RSS fetcher."""
        if self._rss_fetcher is None:
            self._rss_fetcher = RSSNewsFetcher(self.config.rss_feeds)
        return self._rss_fetcher

    async def fetch_news(
        self,
        symbol: str,
        limit: int = 50,
        lookback_hours: int = 24,
    ) -> list[NewsArticle]:
        """
        Fetch news from all configured sources.

        Combines Reddit posts and RSS articles into unified NewsArticle format.
        """
        articles = []
        lookback = lookback_hours or self.config.lookback_hours

        # Fetch from Reddit
        if self.reddit_fetcher:
            try:
                reddit_posts = await self.reddit_fetcher.fetch_posts(
                    symbol=symbol,
                    subreddits=self.config.subreddits,
                    limit=self.config.reddit_post_limit,
                    lookback_hours=lookback,
                    min_upvotes=self.config.min_upvotes,
                )

                for post in reddit_posts:
                    articles.append(
                        NewsArticle(
                            title=post.title,
                            summary=post.body[:200] if post.body else "",
                            source=f"Reddit r/{post.subreddit}",
                            url=post.url,
                            published_at=post.created_utc,
                            symbols=[symbol],
                            sentiment_score=post.sentiment_score,
                            sentiment_label=self._analyzer.score_to_label(
                                post.sentiment_score
                            ),
                            relevance_score=min(1.0, post.score / 100),  # Normalize
                        )
                    )
            except Exception as e:
                logger.warning(f"Error fetching Reddit sentiment: {e}")

        # Fetch from RSS feeds
        if self.config.rss_enabled:
            try:
                rss_articles = await self.rss_fetcher.fetch_articles(
                    symbol=symbol,
                    lookback_hours=lookback,
                    limit=limit,
                )

                for article in rss_articles:
                    articles.append(
                        NewsArticle(
                            title=article.title,
                            summary=article.summary,
                            source=article.source,
                            url=article.url,
                            published_at=article.published,
                            symbols=[symbol],
                            sentiment_score=article.sentiment_score,
                            sentiment_label=self._analyzer.score_to_label(
                                article.sentiment_score
                            ),
                            relevance_score=0.8,  # RSS articles are generally relevant
                        )
                    )
            except Exception as e:
                logger.warning(f"Error fetching RSS sentiment: {e}")

        # Sort by recency
        articles.sort(key=lambda a: a.published_at, reverse=True)

        return articles[:limit]

    async def fetch_sentiment(
        self,
        symbol: str,
        lookback_hours: int = 24,
    ) -> SentimentData:
        """
        Fetch and aggregate sentiment for a symbol.

        Combines Reddit and RSS sources, weights by relevance/upvotes.
        """
        articles = await self.fetch_news(
            symbol, lookback_hours=lookback_hours
        )

        if not articles:
            return SentimentData(
                symbol=symbol,
                timestamp=datetime.now(),
            )

        # Count by sentiment type and calculate weighted score
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        weighted_score_sum = 0.0
        weight_sum = 0.0

        for article in articles:
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

        overall_score = weighted_score_sum / weight_sum if weight_sum > 0 else 0.0

        return SentimentData(
            symbol=symbol,
            timestamp=datetime.now(),
            articles=articles,
            overall_score=overall_score,
            overall_label=self._analyzer.score_to_label(overall_score),
            bullish_count=bullish_count,
            bearish_count=bearish_count,
            neutral_count=neutral_count,
            article_count=len(articles),
        )


def get_web_sentiment_fetcher(
    reddit_client_id: str = "",
    reddit_client_secret: str = "",
    subreddits: list[str] | None = None,
) -> WebSentimentFetcher:
    """
    Create a WebSentimentFetcher with optional Reddit credentials.

    Args:
        reddit_client_id: Reddit API client ID (optional)
        reddit_client_secret: Reddit API secret (optional)
        subreddits: Custom list of subreddits to scrape

    Returns:
        Configured WebSentimentFetcher
    """
    config = WebSentimentConfig(
        reddit_enabled=bool(reddit_client_id and reddit_client_secret),
        reddit_client_id=reddit_client_id,
        reddit_client_secret=reddit_client_secret,
        subreddits=subreddits or ["wallstreetbets", "stocks", "investing"],
    )
    return WebSentimentFetcher(config)
