"""VWAP (Volume Weighted Average Price) Strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from trader.core.models import Signal
from trader.strategies.base import BaseStrategy

if TYPE_CHECKING:
    from trader.core.models import Position


class VWAPStrategy(BaseStrategy):
    """
    VWAP Mean Reversion Strategy.

    VWAP (Volume Weighted Average Price) represents the average price
    weighted by volume. Institutional traders often use VWAP as a
    benchmark for execution quality.

    Trading logic:
    - BUY: Price significantly below VWAP (discount to fair value)
    - SELL: Price significantly above VWAP (premium to fair value)

    The strategy assumes price will revert to VWAP over time.

    Parameters:
        deviation_pct: Minimum % deviation from VWAP to trigger signal (default: 1.0)
        use_bands: Whether to use standard deviation bands (default: True)
        band_std: Number of standard deviations for bands (default: 2.0)
    """

    def __init__(
        self,
        deviation_pct: float = 1.0,
        use_bands: bool = True,
        band_std: float = 2.0,
    ) -> None:
        """Initialize VWAP strategy.

        Args:
            deviation_pct: Minimum % deviation from VWAP to signal
            use_bands: Whether to calculate VWAP standard deviation bands
            band_std: Number of standard deviations for bands

        Raises:
            ValueError: If parameters are invalid
        """
        if deviation_pct <= 0:
            raise ValueError("Deviation percentage must be positive")
        if band_std <= 0:
            raise ValueError("Band standard deviation must be positive")

        self.deviation_pct = deviation_pct
        self.use_bands = use_bands
        self.band_std = band_std

    @property
    def name(self) -> str:
        return f"vwap_{self.deviation_pct}"

    @property
    def description(self) -> str:
        return f"VWAP Mean Reversion: Buy {self.deviation_pct}% below, sell {self.deviation_pct}% above"

    @property
    def min_bars_required(self) -> int:
        # VWAP needs some bars to establish meaningful average
        return 20

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate VWAP and related indicators."""
        data = data.copy()

        # Calculate VWAP: cumulative(price * volume) / cumulative(volume)
        # Using typical price = (high + low + close) / 3
        typical_price = (data["high"] + data["low"] + data["close"]) / 3
        cum_vol = data["volume"].cumsum()
        cum_vol_price = (typical_price * data["volume"]).cumsum()

        data["vwap"] = cum_vol_price / cum_vol

        # Calculate deviation from VWAP
        data["vwap_deviation"] = (data["close"] - data["vwap"]) / data["vwap"] * 100

        if self.use_bands:
            # Calculate VWAP standard deviation bands
            # Rolling std of (price - vwap)
            squared_diff = (typical_price - data["vwap"]) ** 2
            cum_squared_diff = (squared_diff * data["volume"]).cumsum()
            variance = cum_squared_diff / cum_vol
            std = variance**0.5

            data["vwap_upper"] = data["vwap"] + (std * self.band_std)
            data["vwap_lower"] = data["vwap"] - (std * self.band_std)

        return data

    def generate_signal(
        self,
        data: pd.DataFrame,
        symbol: str,
        position: Position | None = None,
    ) -> Signal:
        """Generate signal based on VWAP deviation."""
        if not self.should_generate_signal(data):
            return self.hold_signal(symbol, "Not enough data")

        current = data.iloc[-1]

        if pd.isna(current["vwap"]):
            return self.hold_signal(symbol, "VWAP not ready")

        close = current["close"]
        vwap = current["vwap"]
        deviation = current["vwap_deviation"]

        # Buy when price is significantly below VWAP
        if position is None and deviation < -self.deviation_pct:
            confidence = min(1.0, abs(deviation) / (self.deviation_pct * 3))

            # Extra confidence if below lower band
            if (
                self.use_bands
                and not pd.isna(current.get("vwap_lower"))
                and close < current["vwap_lower"]
            ):
                confidence = min(1.0, confidence + 0.2)

            return self.buy_signal(
                symbol=symbol,
                reason=f"Price {deviation:.1f}% below VWAP (${close:.2f} vs ${vwap:.2f})",
                confidence=confidence,
                take_profit=vwap,  # Target VWAP
            )

        # Sell when price is significantly above VWAP
        if position is not None and deviation > self.deviation_pct:
            confidence = min(1.0, abs(deviation) / (self.deviation_pct * 3))

            if (
                self.use_bands
                and not pd.isna(current.get("vwap_upper"))
                and close > current["vwap_upper"]
            ):
                confidence = min(1.0, confidence + 0.2)

            return self.sell_signal(
                symbol=symbol,
                reason=f"Price {deviation:.1f}% above VWAP (${close:.2f} vs ${vwap:.2f})",
                confidence=confidence,
            )

        # Also sell if price returned to VWAP (take profit)
        if position is not None and abs(deviation) < 0.2:
            return self.sell_signal(
                symbol=symbol,
                reason=f"Price at VWAP: ${close:.2f} ≈ ${vwap:.2f}",
                confidence=0.6,
            )

        return self.hold_signal(
            symbol=symbol,
            reason=f"VWAP deviation: {deviation:+.1f}% (threshold: ±{self.deviation_pct}%)",
        )


class VWAPTrendStrategy(BaseStrategy):
    """
    VWAP Trend Following Strategy.

    Instead of mean reversion, this strategy uses VWAP as a trend filter.
    Price consistently above VWAP indicates bullish sentiment, while
    price below VWAP indicates bearish sentiment.

    Trading logic:
    - BUY: Price crosses above VWAP (bullish)
    - SELL: Price crosses below VWAP (bearish)

    Parameters:
        confirmation_bars: Number of bars above/below VWAP to confirm (default: 3)
    """

    def __init__(
        self,
        confirmation_bars: int = 3,
    ) -> None:
        """Initialize VWAP Trend strategy.

        Args:
            confirmation_bars: Bars needed to confirm trend

        Raises:
            ValueError: If parameters are invalid
        """
        if confirmation_bars < 1:
            raise ValueError("Confirmation bars must be at least 1")

        self.confirmation_bars = confirmation_bars

    @property
    def name(self) -> str:
        return f"vwap_trend_{self.confirmation_bars}"

    @property
    def description(self) -> str:
        return f"VWAP Trend: Buy above VWAP, sell below ({self.confirmation_bars} bar confirm)"

    @property
    def min_bars_required(self) -> int:
        return max(20, self.confirmation_bars + 5)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate VWAP indicator."""
        data = data.copy()

        typical_price = (data["high"] + data["low"] + data["close"]) / 3
        cum_vol = data["volume"].cumsum()
        cum_vol_price = (typical_price * data["volume"]).cumsum()

        data["vwap"] = cum_vol_price / cum_vol
        data["above_vwap"] = data["close"] > data["vwap"]

        return data

    def generate_signal(
        self,
        data: pd.DataFrame,
        symbol: str,
        position: Position | None = None,
    ) -> Signal:
        """Generate signal based on VWAP trend."""
        if not self.should_generate_signal(data):
            return self.hold_signal(symbol, "Not enough data")

        current = data.iloc[-1]

        if pd.isna(current["vwap"]):
            return self.hold_signal(symbol, "VWAP not ready")

        close = current["close"]
        vwap = current["vwap"]

        # Check confirmation bars
        recent = data.tail(self.confirmation_bars)
        bars_above = recent["above_vwap"].sum()
        bars_below = self.confirmation_bars - bars_above

        # Buy: All recent bars above VWAP
        if position is None and bars_above == self.confirmation_bars:
            return self.buy_signal(
                symbol=symbol,
                reason=f"Price above VWAP for {self.confirmation_bars} bars (${close:.2f} > ${vwap:.2f})",
                confidence=0.7,
                stop_loss=vwap * 0.99,  # Stop just below VWAP
            )

        # Sell: All recent bars below VWAP
        if position is not None and bars_below == self.confirmation_bars:
            return self.sell_signal(
                symbol=symbol,
                reason=f"Price below VWAP for {self.confirmation_bars} bars (${close:.2f} < ${vwap:.2f})",
                confidence=0.7,
            )

        return self.hold_signal(
            symbol=symbol,
            reason=f"VWAP trend unclear: {bars_above}/{self.confirmation_bars} bars above",
        )
