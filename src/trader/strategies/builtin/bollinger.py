"""Bollinger Bands Strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from ta.volatility import BollingerBands

from trader.core.models import Signal
from trader.strategies.base import BaseStrategy

if TYPE_CHECKING:
    from trader.core.models import Position


class BollingerBandsStrategy(BaseStrategy):
    """
    Bollinger Bands Mean Reversion Strategy.

    This strategy uses Bollinger Bands to identify overbought and oversold
    conditions. Bollinger Bands consist of:
    - Middle band: Simple Moving Average (SMA)
    - Upper band: SMA + (std_dev * num_std)
    - Lower band: SMA - (std_dev * num_std)

    Trading signals:
    - BUY: Price touches or crosses below lower band (oversold)
    - SELL: Price touches or crosses above upper band (overbought)

    The strategy works best in range-bound markets where price tends to
    revert to the mean. In strong trends, prices can "walk the bands"
    leading to false signals.

    Parameters:
        period: Lookback period for SMA and std dev (default: 20)
        num_std: Number of standard deviations for bands (default: 2.0)
    """

    def __init__(
        self,
        period: int = 20,
        num_std: float = 2.0,
    ) -> None:
        """Initialize Bollinger Bands strategy.

        Args:
            period: Lookback period for calculations
            num_std: Number of standard deviations for band width

        Raises:
            ValueError: If parameters are invalid
        """
        if period <= 0:
            raise ValueError("Period must be positive")
        if num_std <= 0:
            raise ValueError("Number of standard deviations must be positive")

        self.period = period
        self.num_std = num_std

    @property
    def name(self) -> str:
        return f"bollinger_{self.period}_{self.num_std}"

    @property
    def description(self) -> str:
        return f"Bollinger Bands ({self.period}, {self.num_std}σ): Buy at lower band, sell at upper"

    @property
    def min_bars_required(self) -> int:
        return self.period + 1

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands indicators."""
        data = data.copy()

        bb = BollingerBands(
            close=data["close"],
            window=self.period,
            window_dev=self.num_std,
        )

        data["bb_upper"] = bb.bollinger_hband()
        data["bb_middle"] = bb.bollinger_mavg()
        data["bb_lower"] = bb.bollinger_lband()
        data["bb_width"] = bb.bollinger_wband()
        data["bb_pct"] = bb.bollinger_pband()  # %B indicator (0-1 scale)

        return data

    def generate_signal(
        self,
        data: pd.DataFrame,
        symbol: str,
        position: Position | None = None,
    ) -> Signal:
        """Generate signal based on Bollinger Bands position."""
        if not self.should_generate_signal(data):
            return self.hold_signal(symbol, "Not enough data")

        current = data.iloc[-1]
        prev = data.iloc[-2]

        # Check if indicators are ready
        if pd.isna(current["bb_upper"]) or pd.isna(current["bb_lower"]):
            return self.hold_signal(symbol, "Bollinger Bands not ready")

        close = current["close"]
        bb_upper = current["bb_upper"]
        bb_lower = current["bb_lower"]
        bb_middle = current["bb_middle"]
        bb_pct = current["bb_pct"]

        prev_close = prev["close"]
        prev_bb_lower = prev["bb_lower"]
        prev_bb_upper = prev["bb_upper"]

        # Buy signal: Price crosses below or touches lower band
        if position is None:
            # Price crossed below lower band
            if close <= bb_lower or (prev_close > prev_bb_lower and close <= bb_lower):
                # Confidence based on how far below the band
                distance_pct = (bb_lower - close) / bb_middle * 100
                confidence = min(1.0, 0.6 + distance_pct / 5)

                return self.buy_signal(
                    symbol=symbol,
                    reason=f"Price at lower band: ${close:.2f} <= ${bb_lower:.2f}",
                    confidence=confidence,
                    stop_loss=close * 0.97,  # 3% stop loss below entry
                    take_profit=bb_middle,  # Target middle band
                )

            # Price approaching lower band (%B < 0.1)
            if bb_pct < 0.1:
                return self.buy_signal(
                    symbol=symbol,
                    reason=f"Price near lower band: %B={bb_pct:.2f}",
                    confidence=0.5,
                )

        # Sell signal: Price crosses above or touches upper band
        if position is not None:
            # Price crossed above upper band
            if close >= bb_upper or (prev_close < prev_bb_upper and close >= bb_upper):
                distance_pct = (close - bb_upper) / bb_middle * 100
                confidence = min(1.0, 0.6 + distance_pct / 5)

                return self.sell_signal(
                    symbol=symbol,
                    reason=f"Price at upper band: ${close:.2f} >= ${bb_upper:.2f}",
                    confidence=confidence,
                )

            # Price approaching upper band (%B > 0.9)
            if bb_pct > 0.9:
                return self.sell_signal(
                    symbol=symbol,
                    reason=f"Price near upper band: %B={bb_pct:.2f}",
                    confidence=0.5,
                )

            # Also sell if price dropped back to middle after being high
            if bb_pct < 0.5 and prev["bb_pct"] > 0.7:
                return self.sell_signal(
                    symbol=symbol,
                    reason="Price retreating from upper band",
                    confidence=0.4,
                )

        return self.hold_signal(
            symbol=symbol,
            reason=f"Price within bands: %B={bb_pct:.2f}",
        )


class BollingerBreakoutStrategy(BaseStrategy):
    """
    Bollinger Bands Breakout Strategy.

    Unlike the mean reversion strategy, this strategy trades breakouts
    from the bands, betting that a strong move outside the bands will
    continue in that direction (momentum).

    Trading signals:
    - BUY: Price closes above upper band (bullish breakout)
    - SELL: Price closes below lower band OR retraces to middle band

    This strategy works best in trending markets where breakouts lead
    to sustained moves.

    Parameters:
        period: Lookback period for SMA and std dev (default: 20)
        num_std: Number of standard deviations for bands (default: 2.0)
    """

    def __init__(
        self,
        period: int = 20,
        num_std: float = 2.0,
    ) -> None:
        """Initialize Bollinger Breakout strategy."""
        if period <= 0:
            raise ValueError("Period must be positive")
        if num_std <= 0:
            raise ValueError("Number of standard deviations must be positive")

        self.period = period
        self.num_std = num_std

    @property
    def name(self) -> str:
        return f"bollinger_breakout_{self.period}_{self.num_std}"

    @property
    def description(self) -> str:
        return (
            f"Bollinger Breakout ({self.period}, {self.num_std}σ): Buy upper breakout"
        )

    @property
    def min_bars_required(self) -> int:
        return self.period + 1

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands indicators."""
        data = data.copy()

        bb = BollingerBands(
            close=data["close"],
            window=self.period,
            window_dev=self.num_std,
        )

        data["bb_upper"] = bb.bollinger_hband()
        data["bb_middle"] = bb.bollinger_mavg()
        data["bb_lower"] = bb.bollinger_lband()
        data["bb_pct"] = bb.bollinger_pband()

        return data

    def generate_signal(
        self,
        data: pd.DataFrame,
        symbol: str,
        position: Position | None = None,
    ) -> Signal:
        """Generate signal based on Bollinger Band breakouts."""
        if not self.should_generate_signal(data):
            return self.hold_signal(symbol, "Not enough data")

        current = data.iloc[-1]
        prev = data.iloc[-2]

        if pd.isna(current["bb_upper"]):
            return self.hold_signal(symbol, "Bollinger Bands not ready")

        close = current["close"]
        bb_upper = current["bb_upper"]
        bb_lower = current["bb_lower"]
        bb_middle = current["bb_middle"]

        prev_close = prev["close"]
        prev_bb_upper = prev["bb_upper"]

        # Buy on upper band breakout
        if position is None and prev_close <= prev_bb_upper and close > bb_upper:
            return self.buy_signal(
                symbol=symbol,
                reason=f"Bullish breakout: ${close:.2f} > upper ${bb_upper:.2f}",
                confidence=0.7,
                stop_loss=bb_middle,  # Stop at middle band
            )

        # Sell signals for existing position
        if position is not None:
            # Stop loss: price falls below middle band
            if close < bb_middle:
                return self.sell_signal(
                    symbol=symbol,
                    reason=f"Price below middle band: ${close:.2f} < ${bb_middle:.2f}",
                    confidence=0.8,
                )

            # Take profit or stop: price falls below lower band
            if close < bb_lower:
                return self.sell_signal(
                    symbol=symbol,
                    reason=f"Price at lower band: ${close:.2f}",
                    confidence=0.9,
                )

        return self.hold_signal(
            symbol=symbol,
            reason=f"No breakout: price ${close:.2f} within bands",
        )
