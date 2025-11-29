"""Momentum Ranking Strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from trader.core.models import Signal
from trader.strategies.base import BaseStrategy

if TYPE_CHECKING:
    from trader.core.models import Position


class MomentumStrategy(BaseStrategy):
    """
    Momentum Ranking Strategy.

    For single-symbol mode:
    - Buy when recent returns are positive and above threshold
    - Sell when momentum turns negative

    For multi-symbol mode (portfolio):
    - Rank stocks by momentum (returns over lookback period)
    - Hold top N stocks
    - Rebalance periodically

    The lookback skips the most recent week to avoid short-term reversal.

    Parameters:
        lookback_days: Period to calculate momentum (default: 126 = 6 months)
        skip_days: Recent days to skip (default: 5 = 1 week)
        hold_days: Days between rebalancing (default: 5 = weekly)
        momentum_threshold: Minimum momentum to buy (default: 0.0)
    """

    def __init__(
        self,
        lookback_days: int = 126,
        skip_days: int = 5,
        hold_days: int = 5,
        momentum_threshold: float = 0.0,
    ) -> None:
        """Initialize momentum strategy.

        Args:
            lookback_days: Days to look back for momentum calculation
            skip_days: Recent days to skip (avoids short-term reversal)
            hold_days: Days to hold before considering rebalance
            momentum_threshold: Minimum return to trigger buy

        Raises:
            ValueError: If parameters are invalid
        """
        if lookback_days <= 0:
            raise ValueError("lookback_days must be positive")
        if skip_days < 0:
            raise ValueError("skip_days cannot be negative")
        if hold_days <= 0:
            raise ValueError("hold_days must be positive")
        if lookback_days <= skip_days:
            raise ValueError("lookback_days must be greater than skip_days")

        self.lookback_days = lookback_days
        self.skip_days = skip_days
        self.hold_days = hold_days
        self.momentum_threshold = momentum_threshold

        # Track last signal date for rebalancing
        self._last_signal_date: pd.Timestamp | None = None

    @property
    def name(self) -> str:
        return f"momentum_{self.lookback_days}_{self.skip_days}"

    @property
    def description(self) -> str:
        return (
            f"Momentum ({self.lookback_days}d lookback, skip {self.skip_days}d): "
            f"Buy positive momentum, rebalance every {self.hold_days}d"
        )

    @property
    def min_bars_required(self) -> int:
        return self.lookback_days + 1

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum (returns over lookback period)."""
        data = data.copy()

        # Momentum = return from (lookback_days ago) to (skip_days ago)
        # This avoids short-term reversal effect

        if len(data) >= self.lookback_days:
            # Price at start of lookback (excluding skip period)
            lookback_start = self.lookback_days
            lookback_end = self.skip_days if self.skip_days > 0 else 1

            # Calculate rolling momentum
            data["momentum"] = (
                data["close"].shift(lookback_end) / data["close"].shift(lookback_start) - 1
            ) * 100  # As percentage

            # Also track simple recent return for context
            data["return_5d"] = data["close"].pct_change(5) * 100
            data["return_20d"] = data["close"].pct_change(20) * 100
        else:
            data["momentum"] = float("nan")
            data["return_5d"] = float("nan")
            data["return_20d"] = float("nan")

        return data

    def generate_signal(
        self,
        data: pd.DataFrame,
        symbol: str,
        position: Position | None = None,
    ) -> Signal:
        """Generate signal based on momentum."""
        if not self.should_generate_signal(data):
            return self.hold_signal(symbol, "Not enough data")

        current = data.iloc[-1]

        if pd.isna(current["momentum"]):
            return self.hold_signal(symbol, "Momentum not ready")

        momentum = current["momentum"]
        current_date = data.index[-1]

        # Check if we should rebalance
        should_rebalance = True
        if self._last_signal_date is not None:
            days_since_signal = (current_date - self._last_signal_date).days
            should_rebalance = days_since_signal >= self.hold_days

        if not should_rebalance:
            return self.hold_signal(
                symbol=symbol,
                reason=f"Holding (rebalance in {self.hold_days - days_since_signal}d)",
            )

        # No position - consider buying if momentum is strong
        if position is None:
            if momentum > self.momentum_threshold:
                self._last_signal_date = current_date
                # Confidence based on strength of momentum
                confidence = min(1.0, momentum / 50)  # 50% return = full confidence
                return self.buy_signal(
                    symbol=symbol,
                    reason=(
                        f"Positive momentum: {momentum:.1f}% over {self.lookback_days}d "
                        f"(5d: {current['return_5d']:.1f}%, 20d: {current['return_20d']:.1f}%)"
                    ),
                    confidence=max(0.5, confidence),
                )
            else:
                return self.hold_signal(
                    symbol=symbol,
                    reason=f"Momentum too weak: {momentum:.1f}% < {self.momentum_threshold}%",
                )

        # Have position - consider selling if momentum turns negative
        if momentum < 0:
            self._last_signal_date = current_date
            return self.sell_signal(
                symbol=symbol,
                reason=f"Negative momentum: {momentum:.1f}% over {self.lookback_days}d",
            )

        return self.hold_signal(
            symbol=symbol,
            reason=f"Holding: momentum still positive at {momentum:.1f}%",
        )

    def rank_symbols(self, symbol_data: dict[str, pd.DataFrame]) -> list[tuple[str, float]]:
        """
        Rank multiple symbols by momentum.

        This is used for portfolio-level momentum strategies where
        you want to hold the top N stocks.

        Args:
            symbol_data: Dict mapping symbol to its OHLCV DataFrame

        Returns:
            List of (symbol, momentum) tuples, sorted by momentum descending
        """
        rankings = []

        for symbol, data in symbol_data.items():
            if len(data) < self.min_bars_required:
                continue

            data_with_indicators = self.calculate_indicators(data)
            momentum = data_with_indicators["momentum"].iloc[-1]

            if not pd.isna(momentum):
                rankings.append((symbol, momentum))

        # Sort by momentum descending (highest first)
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings
