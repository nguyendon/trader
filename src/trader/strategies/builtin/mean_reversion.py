"""Mean Reversion Strategy using Z-Score."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from trader.core.models import Signal
from trader.strategies.base import BaseStrategy

if TYPE_CHECKING:
    from trader.core.models import Position


class MeanReversionStrategy(BaseStrategy):
    """
    Statistical Mean Reversion Strategy using Z-Score.

    This strategy assumes prices tend to revert to their historical mean.
    It uses z-score (number of standard deviations from mean) to identify
    extreme price movements that are likely to reverse.

    Trading logic:
    - BUY: Z-score < -threshold (price significantly below mean)
    - SELL: Z-score > +threshold (price significantly above mean)

    The strategy works well when prices oscillate around a stable mean,
    but can fail during regime changes or strong trends.

    Parameters:
        lookback: Period for calculating mean and std dev (default: 20)
        entry_zscore: Z-score threshold for entry (default: 2.0)
        exit_zscore: Z-score threshold for exit (default: 0.5)
    """

    def __init__(
        self,
        lookback: int = 20,
        entry_zscore: float = 2.0,
        exit_zscore: float = 0.5,
    ) -> None:
        """Initialize Mean Reversion strategy.

        Args:
            lookback: Period for mean/std calculation
            entry_zscore: Z-score magnitude required to enter
            exit_zscore: Z-score to close position (near mean)

        Raises:
            ValueError: If parameters are invalid
        """
        if lookback <= 1:
            raise ValueError("Lookback must be greater than 1")
        if entry_zscore <= 0:
            raise ValueError("Entry z-score must be positive")
        if exit_zscore < 0:
            raise ValueError("Exit z-score must be non-negative")
        if exit_zscore >= entry_zscore:
            raise ValueError("Exit z-score must be less than entry z-score")

        self.lookback = lookback
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore

    @property
    def name(self) -> str:
        return f"mean_reversion_{self.lookback}_{self.entry_zscore}"

    @property
    def description(self) -> str:
        return (
            f"Mean Reversion ({self.lookback}d): "
            f"Enter at {self.entry_zscore}σ, exit at {self.exit_zscore}σ"
        )

    @property
    def min_bars_required(self) -> int:
        return self.lookback + 5

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate mean reversion indicators."""
        data = data.copy()

        # Rolling mean and standard deviation
        data["rolling_mean"] = data["close"].rolling(window=self.lookback).mean()
        data["rolling_std"] = data["close"].rolling(window=self.lookback).std()

        # Z-score: how many std deviations from mean
        data["zscore"] = (data["close"] - data["rolling_mean"]) / data["rolling_std"]

        # Distance from mean as percentage
        data["mean_distance_pct"] = (
            (data["close"] - data["rolling_mean"]) / data["rolling_mean"] * 100
        )

        return data

    def generate_signal(
        self,
        data: pd.DataFrame,
        symbol: str,
        position: Position | None = None,
    ) -> Signal:
        """Generate signal based on z-score."""
        if not self.should_generate_signal(data):
            return self.hold_signal(symbol, "Not enough data")

        current = data.iloc[-1]

        if pd.isna(current["zscore"]):
            return self.hold_signal(symbol, "Z-score not ready")

        zscore = current["zscore"]
        close = current["close"]
        mean = current["rolling_mean"]
        distance_pct = current["mean_distance_pct"]

        # Entry: Buy when significantly below mean (negative z-score)
        if position is None and zscore < -self.entry_zscore:
            confidence = min(1.0, abs(zscore) / (self.entry_zscore * 2))
            return self.buy_signal(
                symbol=symbol,
                reason=f"Oversold: z-score={zscore:.2f} ({distance_pct:.1f}% below mean)",
                confidence=confidence,
                take_profit=mean,  # Target the mean
                stop_loss=close
                * (1 - abs(distance_pct) / 100 * 1.5),  # 1.5x the distance
            )

        # Exit long position
        if position is not None:
            # Take profit: price returned to mean
            if abs(zscore) < self.exit_zscore:
                return self.sell_signal(
                    symbol=symbol,
                    reason=f"Mean reached: z-score={zscore:.2f}",
                    confidence=0.8,
                )

            # Stop loss: price went even further from mean (wrong direction)
            if zscore < -self.entry_zscore * 1.5:
                return self.sell_signal(
                    symbol=symbol,
                    reason=f"Stop loss: z-score={zscore:.2f} (extended move)",
                    confidence=0.9,
                )

            # Overbought: price moved above mean significantly
            if zscore > self.entry_zscore:
                return self.sell_signal(
                    symbol=symbol,
                    reason=f"Overbought: z-score={zscore:.2f}",
                    confidence=0.7,
                )

        return self.hold_signal(
            symbol=symbol,
            reason=f"Z-score={zscore:.2f} (entry threshold: ±{self.entry_zscore})",
        )


class MeanReversionPairsStrategy(BaseStrategy):
    """
    Mean Reversion Strategy with Price Channel.

    Uses a price channel (highest high / lowest low) to identify
    mean reversion opportunities. When price touches channel extremes,
    it signals potential reversals.

    Trading logic:
    - BUY: Price at or near lowest low of lookback period
    - SELL: Price at or near highest high of lookback period

    Parameters:
        lookback: Period for high/low channel (default: 20)
        entry_pct: Percentage from extreme to trigger entry (default: 5)
    """

    def __init__(
        self,
        lookback: int = 20,
        entry_pct: float = 5.0,
    ) -> None:
        """Initialize Price Channel Mean Reversion strategy.

        Args:
            lookback: Period for channel calculation
            entry_pct: Percentage from channel extreme to trigger

        Raises:
            ValueError: If parameters are invalid
        """
        if lookback <= 1:
            raise ValueError("Lookback must be greater than 1")
        if entry_pct <= 0 or entry_pct >= 50:
            raise ValueError("Entry percentage must be between 0 and 50")

        self.lookback = lookback
        self.entry_pct = entry_pct

    @property
    def name(self) -> str:
        return f"mean_reversion_channel_{self.lookback}"

    @property
    def description(self) -> str:
        return f"Price Channel Mean Reversion ({self.lookback}d): Buy low, sell high"

    @property
    def min_bars_required(self) -> int:
        return self.lookback + 5

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate price channel indicators."""
        data = data.copy()

        # Rolling highest high and lowest low
        data["channel_high"] = data["high"].rolling(window=self.lookback).max()
        data["channel_low"] = data["low"].rolling(window=self.lookback).min()

        # Channel midpoint
        data["channel_mid"] = (data["channel_high"] + data["channel_low"]) / 2

        # Channel width
        data["channel_width"] = data["channel_high"] - data["channel_low"]

        # Position within channel (0 = at low, 1 = at high)
        data["channel_position"] = (data["close"] - data["channel_low"]) / data[
            "channel_width"
        ]

        return data

    def generate_signal(
        self,
        data: pd.DataFrame,
        symbol: str,
        position: Position | None = None,
    ) -> Signal:
        """Generate signal based on price channel position."""
        if not self.should_generate_signal(data):
            return self.hold_signal(symbol, "Not enough data")

        current = data.iloc[-1]

        if pd.isna(current["channel_high"]) or pd.isna(current["channel_low"]):
            return self.hold_signal(symbol, "Channel not ready")

        close = current["close"]
        channel_high = current["channel_high"]
        channel_low = current["channel_low"]
        channel_mid = current["channel_mid"]
        channel_position = current["channel_position"]

        # Calculate threshold positions
        low_threshold = self.entry_pct / 100
        high_threshold = 1 - (self.entry_pct / 100)

        # Buy at channel low
        if position is None and channel_position <= low_threshold:
            distance_from_low = (close - channel_low) / channel_low * 100
            confidence = min(1.0, 0.5 + (low_threshold - channel_position) * 2)

            return self.buy_signal(
                symbol=symbol,
                reason=f"At channel low: ${close:.2f} ({distance_from_low:.1f}% from ${channel_low:.2f})",
                confidence=confidence,
                take_profit=channel_mid,
                stop_loss=channel_low * 0.98,  # 2% below channel low
            )

        # Sell at channel high or midpoint
        if position is not None:
            if channel_position >= high_threshold:
                distance_from_high = (channel_high - close) / channel_high * 100
                confidence = min(1.0, 0.5 + (channel_position - high_threshold) * 2)

                return self.sell_signal(
                    symbol=symbol,
                    reason=f"At channel high: ${close:.2f} ({distance_from_high:.1f}% from ${channel_high:.2f})",
                    confidence=confidence,
                )

            # Also sell at midpoint (partial target)
            if 0.45 <= channel_position <= 0.55:
                return self.sell_signal(
                    symbol=symbol,
                    reason=f"At channel midpoint: ${close:.2f} ≈ ${channel_mid:.2f}",
                    confidence=0.5,
                )

        return self.hold_signal(
            symbol=symbol,
            reason=f"Channel position: {channel_position:.0%} (buy <{low_threshold:.0%}, sell >{high_threshold:.0%})",
        )
