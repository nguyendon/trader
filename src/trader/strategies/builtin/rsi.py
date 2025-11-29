"""RSI Mean Reversion Strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from ta.momentum import RSIIndicator

from trader.core.models import Signal
from trader.strategies.base import BaseStrategy

if TYPE_CHECKING:
    from trader.core.models import Position


class RSIStrategy(BaseStrategy):
    """
    RSI Mean Reversion Strategy.

    Buy when RSI drops below oversold level (default 30), indicating
    the stock may be undervalued and due for a bounce.

    Sell when RSI rises above overbought level (default 70), indicating
    the stock may be overvalued and due for a pullback.

    This is a classic mean reversion strategy that works well in
    range-bound markets but can get crushed in strong trends.

    Parameters:
        period: RSI calculation period (default: 14)
        oversold: Buy threshold (default: 30)
        overbought: Sell threshold (default: 70)
    """

    def __init__(
        self,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
    ) -> None:
        """Initialize RSI strategy.

        Args:
            period: RSI lookback period
            oversold: RSI level below which to buy
            overbought: RSI level above which to sell

        Raises:
            ValueError: If parameters are invalid
        """
        if period <= 0:
            raise ValueError("Period must be positive")
        if not 0 < oversold < overbought < 100:
            raise ValueError("Must have 0 < oversold < overbought < 100")

        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    @property
    def name(self) -> str:
        return f"rsi_{self.period}_{int(self.oversold)}_{int(self.overbought)}"

    @property
    def description(self) -> str:
        return (
            f"RSI Mean Reversion ({self.period}): "
            f"Buy <{self.oversold}, Sell >{self.overbought}"
        )

    @property
    def min_bars_required(self) -> int:
        return self.period + 1

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI indicator."""
        data = data.copy()
        data["rsi"] = RSIIndicator(close=data["close"], window=self.period).rsi()
        return data

    def generate_signal(
        self,
        data: pd.DataFrame,
        symbol: str,
        position: Position | None = None,
    ) -> Signal:
        """Generate signal based on RSI levels."""
        if not self.should_generate_signal(data):
            return self.hold_signal(symbol, "Not enough data")

        current = data.iloc[-1]

        if pd.isna(current["rsi"]):
            return self.hold_signal(symbol, "RSI not ready")

        rsi = current["rsi"]

        # Buy when oversold (only if no position)
        if rsi < self.oversold and position is None:
            return self.buy_signal(
                symbol=symbol,
                reason=f"RSI oversold: {rsi:.1f} < {self.oversold}",
                confidence=min(1.0, (self.oversold - rsi) / 10),  # Higher confidence when more oversold
            )

        # Sell when overbought (only if we have position)
        if rsi > self.overbought and position is not None:
            return self.sell_signal(
                symbol=symbol,
                reason=f"RSI overbought: {rsi:.1f} > {self.overbought}",
                confidence=min(1.0, (rsi - self.overbought) / 10),
            )

        return self.hold_signal(
            symbol=symbol,
            reason=f"RSI neutral: {rsi:.1f} (wait for <{self.oversold} or >{self.overbought})",
        )
