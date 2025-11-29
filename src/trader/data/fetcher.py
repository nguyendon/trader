"""Data fetching from Alpaca and mock data for testing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger

from trader.core.models import Bar, TimeFrame

if TYPE_CHECKING:
    from trader.config.settings import Settings


class BaseDataFetcher(ABC):
    """Abstract base class for data fetchers."""

    @abstractmethod
    async def fetch_bars(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start: datetime,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> list[Bar]:
        """
        Fetch historical bars for a symbol.

        Args:
            symbol: Stock symbol (e.g., "AAPL")
            timeframe: Bar timeframe (e.g., TimeFrame.DAY)
            start: Start datetime
            end: End datetime (defaults to now)
            limit: Maximum number of bars to return

        Returns:
            List of Bar objects
        """
        pass

    async def fetch_bars_df(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start: datetime,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """
        Fetch historical bars as a pandas DataFrame.

        Args:
            symbol: Stock symbol
            timeframe: Bar timeframe
            start: Start datetime
            end: End datetime
            limit: Maximum number of bars

        Returns:
            DataFrame with columns: open, high, low, close, volume
            Index is datetime
        """
        bars = await self.fetch_bars(symbol, timeframe, start, end, limit)

        if not bars:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"]
            )

        data = {
            "open": [float(bar.open) for bar in bars],
            "high": [float(bar.high) for bar in bars],
            "low": [float(bar.low) for bar in bars],
            "close": [float(bar.close) for bar in bars],
            "volume": [bar.volume for bar in bars],
        }

        df = pd.DataFrame(data, index=[bar.timestamp for bar in bars])
        df.index.name = "timestamp"

        return df


class AlpacaDataFetcher(BaseDataFetcher):
    """Fetch market data from Alpaca API."""

    def __init__(self, api_key: str, secret_key: str, paper: bool = True) -> None:
        """Initialize with Alpaca credentials.

        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            paper: Whether to use paper trading API
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper
        self._client: "StockHistoricalDataClient | None" = None

    @property
    def client(self) -> "StockHistoricalDataClient":
        """Lazy-initialize the Alpaca client."""
        if self._client is None:
            from alpaca.data.historical import StockHistoricalDataClient

            self._client = StockHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
            )
        return self._client

    def _timeframe_to_alpaca(self, timeframe: TimeFrame) -> "TimeFrame":
        """Convert our TimeFrame to Alpaca TimeFrame."""
        from alpaca.data.timeframe import TimeFrame as AlpacaTimeFrame
        from alpaca.data.timeframe import TimeFrameUnit

        mapping = {
            TimeFrame.MINUTE_1: AlpacaTimeFrame(1, TimeFrameUnit.Minute),
            TimeFrame.MINUTE_5: AlpacaTimeFrame(5, TimeFrameUnit.Minute),
            TimeFrame.MINUTE_15: AlpacaTimeFrame(15, TimeFrameUnit.Minute),
            TimeFrame.MINUTE_30: AlpacaTimeFrame(30, TimeFrameUnit.Minute),
            TimeFrame.HOUR_1: AlpacaTimeFrame(1, TimeFrameUnit.Hour),
            TimeFrame.HOUR_4: AlpacaTimeFrame(4, TimeFrameUnit.Hour),
            TimeFrame.DAY: AlpacaTimeFrame(1, TimeFrameUnit.Day),
            TimeFrame.WEEK: AlpacaTimeFrame(1, TimeFrameUnit.Week),
        }
        return mapping[timeframe]

    async def fetch_bars(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start: datetime,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> list[Bar]:
        """Fetch bars from Alpaca."""
        from alpaca.data.requests import StockBarsRequest

        if end is None:
            end = datetime.now()

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=self._timeframe_to_alpaca(timeframe),
            start=start,
            end=end,
            limit=limit,
        )

        logger.debug(f"Fetching {symbol} bars from {start} to {end}")

        # Note: This is synchronous but we wrap it for consistency
        response = self.client.get_stock_bars(request)

        bars = []
        if symbol in response:
            for bar_data in response[symbol]:
                bars.append(
                    Bar(
                        symbol=symbol,
                        timestamp=bar_data.timestamp,
                        open=Decimal(str(bar_data.open)),
                        high=Decimal(str(bar_data.high)),
                        low=Decimal(str(bar_data.low)),
                        close=Decimal(str(bar_data.close)),
                        volume=int(bar_data.volume),
                        timeframe=timeframe,
                    )
                )

        logger.debug(f"Fetched {len(bars)} bars for {symbol}")
        return bars


class MockDataFetcher(BaseDataFetcher):
    """
    Mock data fetcher for testing without API credentials.

    Generates realistic-looking random price data based on a
    geometric Brownian motion model.
    """

    def __init__(
        self,
        base_price: float = 100.0,
        volatility: float = 0.02,
        drift: float = 0.0001,
        seed: int | None = None,
    ) -> None:
        """Initialize mock data generator.

        Args:
            base_price: Starting price for generated data
            volatility: Daily volatility (standard deviation of returns)
            drift: Daily drift (expected return)
            seed: Random seed for reproducibility
        """
        self.base_price = base_price
        self.volatility = volatility
        self.drift = drift
        self.rng = np.random.default_rng(seed)

    async def fetch_bars(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start: datetime,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> list[Bar]:
        """Generate mock bars."""
        if end is None:
            end = datetime.now()

        # Calculate number of bars based on timeframe
        delta = end - start
        if timeframe == TimeFrame.DAY:
            num_bars = delta.days
        elif timeframe == TimeFrame.HOUR_1:
            num_bars = int(delta.total_seconds() / 3600)
        elif timeframe == TimeFrame.MINUTE_1:
            num_bars = int(delta.total_seconds() / 60)
        else:
            # Default to daily
            num_bars = delta.days

        if limit:
            num_bars = min(num_bars, limit)

        if num_bars <= 0:
            return []

        # Generate prices using geometric Brownian motion
        returns = self.rng.normal(
            self.drift, self.volatility, num_bars
        )
        prices = self.base_price * np.exp(np.cumsum(returns))

        bars = []
        for i in range(num_bars):
            # Calculate bar timestamp
            if timeframe == TimeFrame.DAY:
                timestamp = start + timedelta(days=i)
            elif timeframe == TimeFrame.HOUR_1:
                timestamp = start + timedelta(hours=i)
            elif timeframe == TimeFrame.MINUTE_1:
                timestamp = start + timedelta(minutes=i)
            else:
                timestamp = start + timedelta(days=i)

            # Generate OHLC from close price
            close_price = prices[i]

            # Random intraday movement
            high_pct = 1 + abs(self.rng.normal(0, self.volatility / 2))
            low_pct = 1 - abs(self.rng.normal(0, self.volatility / 2))

            # Open is close of previous bar (with small gap)
            if i == 0:
                open_price = self.base_price
            else:
                open_price = prices[i - 1] * (1 + self.rng.normal(0, 0.001))

            high_price = max(open_price, close_price) * high_pct
            low_price = min(open_price, close_price) * low_pct

            # Generate volume (higher on volatile days)
            base_volume = 1_000_000
            volume_multiplier = 1 + abs(returns[i]) * 10
            volume = int(base_volume * volume_multiplier * self.rng.uniform(0.8, 1.2))

            bars.append(
                Bar(
                    symbol=symbol,
                    timestamp=timestamp,
                    open=Decimal(str(round(open_price, 2))),
                    high=Decimal(str(round(high_price, 2))),
                    low=Decimal(str(round(low_price, 2))),
                    close=Decimal(str(round(close_price, 2))),
                    volume=volume,
                    timeframe=timeframe,
                )
            )

        return bars


def get_data_fetcher(settings: "Settings | None" = None) -> BaseDataFetcher:
    """
    Get the appropriate data fetcher based on settings.

    Returns MockDataFetcher if no API credentials are configured,
    otherwise returns AlpacaDataFetcher.

    Args:
        settings: Application settings. If None, uses default settings.

    Returns:
        Data fetcher instance
    """
    if settings is None:
        from trader.config.settings import get_settings
        settings = get_settings()

    if settings.has_alpaca_credentials:
        logger.info("Using Alpaca data fetcher")
        return AlpacaDataFetcher(
            api_key=settings.alpaca_api_key.get_secret_value(),
            secret_key=settings.alpaca_secret_key.get_secret_value(),
            paper=settings.alpaca_paper,
        )
    else:
        logger.warning("No Alpaca credentials found, using mock data fetcher")
        return MockDataFetcher()
