"""Tests for data fetching."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from trader.core.models import TimeFrame
from trader.data.fetcher import MockDataFetcher


class TestMockDataFetcher:
    """Tests for MockDataFetcher."""

    @pytest.fixture
    def fetcher(self) -> MockDataFetcher:
        """Create a seeded mock fetcher for reproducible tests."""
        return MockDataFetcher(
            base_price=100.0,
            volatility=0.02,
            drift=0.0001,
            seed=42,
        )

    @pytest.mark.asyncio
    async def test_fetch_bars_returns_correct_count(
        self, fetcher: MockDataFetcher
    ) -> None:
        """Test that correct number of bars is returned."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 11)  # 10 days

        bars = await fetcher.fetch_bars(
            symbol="TEST",
            timeframe=TimeFrame.DAY,
            start=start,
            end=end,
        )

        assert len(bars) == 10

    @pytest.mark.asyncio
    async def test_fetch_bars_with_limit(self, fetcher: MockDataFetcher) -> None:
        """Test limit parameter."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)

        bars = await fetcher.fetch_bars(
            symbol="TEST",
            timeframe=TimeFrame.DAY,
            start=start,
            end=end,
            limit=5,
        )

        assert len(bars) == 5

    @pytest.mark.asyncio
    async def test_fetch_bars_symbol_correct(self, fetcher: MockDataFetcher) -> None:
        """Test that symbol is set correctly on bars."""
        bars = await fetcher.fetch_bars(
            symbol="AAPL",
            timeframe=TimeFrame.DAY,
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 5),
        )

        for bar in bars:
            assert bar.symbol == "AAPL"

    @pytest.mark.asyncio
    async def test_fetch_bars_timeframe_correct(self, fetcher: MockDataFetcher) -> None:
        """Test that timeframe is set correctly."""
        bars = await fetcher.fetch_bars(
            symbol="TEST",
            timeframe=TimeFrame.HOUR_1,
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 1, 10),
        )

        for bar in bars:
            assert bar.timeframe == TimeFrame.HOUR_1

    @pytest.mark.asyncio
    async def test_fetch_bars_ohlc_relationships(
        self, fetcher: MockDataFetcher
    ) -> None:
        """Test that OHLC relationships are valid."""
        bars = await fetcher.fetch_bars(
            symbol="TEST",
            timeframe=TimeFrame.DAY,
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31),
        )

        for bar in bars:
            # High should be >= open, close, low
            assert bar.high >= bar.open
            assert bar.high >= bar.close
            assert bar.high >= bar.low

            # Low should be <= open, close, high
            assert bar.low <= bar.open
            assert bar.low <= bar.close
            assert bar.low <= bar.high

            # All prices should be positive
            assert bar.open > 0
            assert bar.high > 0
            assert bar.low > 0
            assert bar.close > 0
            assert bar.volume > 0

    @pytest.mark.asyncio
    async def test_fetch_bars_deterministic_with_seed(self) -> None:
        """Test that results are reproducible with same seed."""
        fetcher1 = MockDataFetcher(seed=123)
        fetcher2 = MockDataFetcher(seed=123)

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)

        bars1 = await fetcher1.fetch_bars("TEST", TimeFrame.DAY, start, end)
        bars2 = await fetcher2.fetch_bars("TEST", TimeFrame.DAY, start, end)

        for b1, b2 in zip(bars1, bars2, strict=True):
            assert b1.close == b2.close
            assert b1.volume == b2.volume

    @pytest.mark.asyncio
    async def test_fetch_bars_different_with_different_seed(self) -> None:
        """Test that different seeds produce different results."""
        fetcher1 = MockDataFetcher(seed=123)
        fetcher2 = MockDataFetcher(seed=456)

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)

        bars1 = await fetcher1.fetch_bars("TEST", TimeFrame.DAY, start, end)
        bars2 = await fetcher2.fetch_bars("TEST", TimeFrame.DAY, start, end)

        # At least some prices should be different
        different = any(
            b1.close != b2.close for b1, b2 in zip(bars1, bars2, strict=True)
        )
        assert different

    @pytest.mark.asyncio
    async def test_fetch_bars_empty_for_invalid_range(
        self, fetcher: MockDataFetcher
    ) -> None:
        """Test empty result for invalid date range."""
        bars = await fetcher.fetch_bars(
            symbol="TEST",
            timeframe=TimeFrame.DAY,
            start=datetime(2024, 1, 10),
            end=datetime(2024, 1, 1),  # End before start
        )

        assert len(bars) == 0

    @pytest.mark.asyncio
    async def test_fetch_bars_df(self, fetcher: MockDataFetcher) -> None:
        """Test DataFrame output."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)

        df = await fetcher.fetch_bars_df(
            symbol="TEST",
            timeframe=TimeFrame.DAY,
            start=start,
            end=end,
        )

        assert len(df) == 9
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns
        assert df.index.name == "timestamp"

    @pytest.mark.asyncio
    async def test_fetch_bars_df_empty(self, fetcher: MockDataFetcher) -> None:
        """Test empty DataFrame for invalid range."""
        df = await fetcher.fetch_bars_df(
            symbol="TEST",
            timeframe=TimeFrame.DAY,
            start=datetime(2024, 1, 10),
            end=datetime(2024, 1, 1),
        )

        assert len(df) == 0
        assert "close" in df.columns

    @pytest.mark.asyncio
    async def test_different_base_prices(self) -> None:
        """Test that base_price parameter works."""
        fetcher_low = MockDataFetcher(base_price=10.0, seed=42)
        fetcher_high = MockDataFetcher(base_price=1000.0, seed=42)

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        bars_low = await fetcher_low.fetch_bars("TEST", TimeFrame.DAY, start, end)
        bars_high = await fetcher_high.fetch_bars("TEST", TimeFrame.DAY, start, end)

        # High base price should produce higher prices
        avg_low = sum(float(b.close) for b in bars_low) / len(bars_low)
        avg_high = sum(float(b.close) for b in bars_high) / len(bars_high)

        assert avg_high > avg_low * 50  # Should be ~100x difference

    @pytest.mark.asyncio
    async def test_hourly_timeframe(self, fetcher: MockDataFetcher) -> None:
        """Test hourly bar generation."""
        start = datetime(2024, 1, 1, 0, 0)
        end = datetime(2024, 1, 1, 10, 0)  # 10 hours

        bars = await fetcher.fetch_bars(
            symbol="TEST",
            timeframe=TimeFrame.HOUR_1,
            start=start,
            end=end,
        )

        assert len(bars) == 10

        # Check timestamps are hourly
        for i, bar in enumerate(bars):
            expected_time = start + timedelta(hours=i)
            assert bar.timestamp == expected_time
