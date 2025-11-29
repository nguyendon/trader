"""MACD Crossover Strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from ta.trend import MACD

from trader.core.models import Signal
from trader.strategies.base import BaseStrategy

if TYPE_CHECKING:
    from trader.core.models import Position


class MACDStrategy(BaseStrategy):
    """
    MACD (Moving Average Convergence Divergence) Crossover Strategy.

    Buy when MACD line crosses above the signal line (bullish momentum).
    Sell when MACD line crosses below the signal line (bearish momentum).

    MACD = EMA(fast) - EMA(slow)
    Signal = EMA(MACD, signal_period)

    The histogram shows the difference between MACD and signal line.

    Parameters:
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line period (default: 9)
    """

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> None:
        """Initialize MACD strategy.

        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line smoothing period

        Raises:
            ValueError: If parameters are invalid
        """
        if fast_period <= 0 or slow_period <= 0 or signal_period <= 0:
            raise ValueError("All periods must be positive")
        if fast_period >= slow_period:
            raise ValueError("fast_period must be less than slow_period")

        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    @property
    def name(self) -> str:
        return f"macd_{self.fast_period}_{self.slow_period}_{self.signal_period}"

    @property
    def description(self) -> str:
        return (
            f"MACD ({self.fast_period}/{self.slow_period}/{self.signal_period}): "
            f"Buy on bullish crossover, sell on bearish"
        )

    @property
    def min_bars_required(self) -> int:
        return self.slow_period + self.signal_period + 1

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD, signal line, and histogram."""
        data = data.copy()

        macd = MACD(
            close=data["close"],
            window_slow=self.slow_period,
            window_fast=self.fast_period,
            window_sign=self.signal_period,
        )

        data["macd"] = macd.macd()
        data["macd_signal"] = macd.macd_signal()
        data["macd_histogram"] = macd.macd_diff()

        return data

    def generate_signal(
        self,
        data: pd.DataFrame,
        symbol: str,
        position: Position | None = None,
    ) -> Signal:
        """Generate signal based on MACD crossover."""
        if not self.should_generate_signal(data):
            return self.hold_signal(symbol, "Not enough data")

        current = data.iloc[-1]
        previous = data.iloc[-2]

        # Check for NaN
        if pd.isna(current["macd"]) or pd.isna(current["macd_signal"]):
            return self.hold_signal(symbol, "MACD not ready")
        if pd.isna(previous["macd"]) or pd.isna(previous["macd_signal"]):
            return self.hold_signal(symbol, "Previous MACD not ready")

        # Current and previous positions relative to signal line
        macd_above_signal_now = current["macd"] > current["macd_signal"]
        macd_above_signal_prev = previous["macd"] > previous["macd_signal"]

        # Bullish crossover
        if macd_above_signal_now and not macd_above_signal_prev:
            histogram = current["macd_histogram"]
            return self.buy_signal(
                symbol=symbol,
                reason=(
                    f"Bullish MACD crossover: "
                    f"MACD={current['macd']:.3f} > Signal={current['macd_signal']:.3f} "
                    f"(histogram={histogram:.3f})"
                ),
            )

        # Bearish crossover
        if not macd_above_signal_now and macd_above_signal_prev:
            histogram = current["macd_histogram"]
            return self.sell_signal(
                symbol=symbol,
                reason=(
                    f"Bearish MACD crossover: "
                    f"MACD={current['macd']:.3f} < Signal={current['macd_signal']:.3f} "
                    f"(histogram={histogram:.3f})"
                ),
            )

        return self.hold_signal(
            symbol=symbol,
            reason=(
                f"No crossover: MACD={current['macd']:.3f}, "
                f"Signal={current['macd_signal']:.3f}"
            ),
        )
