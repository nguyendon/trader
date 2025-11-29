"""Simple Moving Average Crossover Strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from ta.trend import SMAIndicator

from trader.core.models import Signal
from trader.strategies.base import BaseStrategy

if TYPE_CHECKING:
    from trader.core.models import Position


class SMACrossover(BaseStrategy):
    """
    Simple Moving Average Crossover Strategy.

    Generates buy signals when the fast SMA crosses above the slow SMA,
    and sell signals when the fast SMA crosses below the slow SMA.

    This is a classic trend-following strategy that works well in
    trending markets but may generate false signals in sideways markets.

    Parameters:
        fast_period: Period for the fast moving average (default: 10)
        slow_period: Period for the slow moving average (default: 50)

    Example:
        strategy = SMACrossover(fast_period=10, slow_period=50)
        data = strategy.calculate_indicators(ohlcv_data)
        signal = strategy.generate_signal(data, "AAPL")
    """

    def __init__(self, fast_period: int = 10, slow_period: int = 50) -> None:
        """Initialize the strategy with SMA periods.

        Args:
            fast_period: Period for the fast SMA. Must be less than slow_period.
            slow_period: Period for the slow SMA.

        Raises:
            ValueError: If fast_period >= slow_period or periods are not positive.
        """
        if fast_period <= 0 or slow_period <= 0:
            raise ValueError("Periods must be positive integers")
        if fast_period >= slow_period:
            raise ValueError("fast_period must be less than slow_period")

        self.fast_period = fast_period
        self.slow_period = slow_period

    @property
    def name(self) -> str:
        """Unique identifier for this strategy instance."""
        return f"sma_crossover_{self.fast_period}_{self.slow_period}"

    @property
    def description(self) -> str:
        """Human-readable description."""
        return (
            f"SMA Crossover ({self.fast_period}/{self.slow_period}): "
            f"Buy when SMA{self.fast_period} crosses above SMA{self.slow_period}"
        )

    @property
    def min_bars_required(self) -> int:
        """Minimum bars needed to calculate the slow SMA."""
        return self.slow_period + 1  # Need one extra for crossover detection

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate fast and slow SMAs.

        Args:
            data: DataFrame with 'close' column.

        Returns:
            DataFrame with added 'sma_fast' and 'sma_slow' columns.
        """
        data = data.copy()

        # Use ta library for indicator calculation
        data["sma_fast"] = SMAIndicator(
            close=data["close"], window=self.fast_period
        ).sma_indicator()

        data["sma_slow"] = SMAIndicator(
            close=data["close"], window=self.slow_period
        ).sma_indicator()

        return data

    def generate_signal(
        self,
        data: pd.DataFrame,
        symbol: str,
        position: Position | None = None,
    ) -> Signal:
        """
        Generate trading signal based on SMA crossover.

        Args:
            data: DataFrame with OHLCV and SMA indicator columns.
            symbol: Symbol to generate signal for.
            position: Current position (unused in this strategy).

        Returns:
            Signal with buy/sell/hold action.
        """
        if not self.should_generate_signal(data):
            return self.hold_signal(symbol, "Not enough data")

        # Get current and previous values
        current = data.iloc[-1]
        previous = data.iloc[-2]

        # Check for NaN values (not enough data for indicators)
        if pd.isna(current["sma_fast"]) or pd.isna(current["sma_slow"]):
            return self.hold_signal(symbol, "Indicators not ready")

        if pd.isna(previous["sma_fast"]) or pd.isna(previous["sma_slow"]):
            return self.hold_signal(symbol, "Previous indicators not ready")

        # Detect crossover
        fast_above_slow_now = current["sma_fast"] > current["sma_slow"]
        fast_above_slow_prev = previous["sma_fast"] > previous["sma_slow"]

        # Bullish crossover: fast crosses above slow
        if fast_above_slow_now and not fast_above_slow_prev:
            return self.buy_signal(
                symbol=symbol,
                reason=(
                    f"Bullish SMA crossover: "
                    f"SMA{self.fast_period}={current['sma_fast']:.2f} > "
                    f"SMA{self.slow_period}={current['sma_slow']:.2f}"
                ),
            )

        # Bearish crossover: fast crosses below slow
        if not fast_above_slow_now and fast_above_slow_prev:
            return self.sell_signal(
                symbol=symbol,
                reason=(
                    f"Bearish SMA crossover: "
                    f"SMA{self.fast_period}={current['sma_fast']:.2f} < "
                    f"SMA{self.slow_period}={current['sma_slow']:.2f}"
                ),
            )

        # No crossover
        return self.hold_signal(
            symbol=symbol,
            reason=(
                f"No crossover: "
                f"SMA{self.fast_period}={current['sma_fast']:.2f}, "
                f"SMA{self.slow_period}={current['sma_slow']:.2f}"
            ),
        )
