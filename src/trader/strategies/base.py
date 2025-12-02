"""Base strategy class and interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

from trader.core.models import Signal, SignalAction

if TYPE_CHECKING:
    from trader.core.models import Position


@dataclass
class StrategyConfig:
    """Configuration for a strategy instance."""

    name: str
    symbols: list[str]
    parameters: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.

    Strategies are stateless - all required state is passed via method arguments.
    This enables easy testing and allows the same strategy to work in both
    backtesting and live trading modes.

    Example:
        class MyStrategy(BaseStrategy):
            def __init__(self, fast_period: int = 10, slow_period: int = 20):
                self.fast_period = fast_period
                self.slow_period = slow_period

            @property
            def name(self) -> str:
                return f"my_strategy_{self.fast_period}_{self.slow_period}"

            def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
                data = data.copy()
                data['sma_fast'] = data['close'].rolling(self.fast_period).mean()
                data['sma_slow'] = data['close'].rolling(self.slow_period).mean()
                return data

            def generate_signal(self, data: pd.DataFrame, symbol: str) -> Signal:
                # Generate signal based on indicators
                ...
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this strategy instance."""
        pass

    @property
    def description(self) -> str:
        """Human-readable description of the strategy."""
        return f"{self.__class__.__name__} strategy"

    @property
    def min_bars_required(self) -> int:
        """Minimum number of bars needed before generating signals.

        Override this in subclasses based on indicator requirements.
        """
        return 1

    @abstractmethod
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators and add them to the data.

        Args:
            data: DataFrame with OHLCV columns (open, high, low, close, volume)
                  Index should be datetime.

        Returns:
            DataFrame with original data plus indicator columns.
            Must not modify the input DataFrame.
        """
        pass

    @abstractmethod
    def generate_signal(
        self,
        data: pd.DataFrame,
        symbol: str,
        position: Position | None = None,
    ) -> Signal:
        """
        Generate a trading signal based on the current data.

        Args:
            data: DataFrame with OHLCV and indicator columns.
                  The last row is the current bar.
            symbol: The symbol to generate signal for.
            position: Current position in this symbol, if any.

        Returns:
            Signal indicating the recommended action (buy/sell/hold).
        """
        pass

    def should_generate_signal(self, data: pd.DataFrame) -> bool:
        """Check if we have enough data to generate a signal.

        Args:
            data: DataFrame with OHLCV data.

        Returns:
            True if we have enough bars to generate a signal.
        """
        return len(data) >= self.min_bars_required

    def hold_signal(self, symbol: str, reason: str = "Holding") -> Signal:
        """Create a hold signal (convenience method)."""
        return Signal(
            action=SignalAction.HOLD,
            symbol=symbol,
            reason=reason,
        )

    def buy_signal(
        self,
        symbol: str,
        reason: str,
        confidence: float = 1.0,
        quantity: int | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> Signal:
        """Create a buy signal (convenience method)."""
        from decimal import Decimal

        return Signal(
            action=SignalAction.BUY,
            symbol=symbol,
            reason=reason,
            confidence=confidence,
            quantity=quantity,
            stop_loss=Decimal(str(stop_loss)) if stop_loss is not None else None,
            take_profit=Decimal(str(take_profit)) if take_profit is not None else None,
        )

    def sell_signal(
        self,
        symbol: str,
        reason: str,
        confidence: float = 1.0,
        quantity: int | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> Signal:
        """Create a sell signal (convenience method)."""
        from decimal import Decimal

        return Signal(
            action=SignalAction.SELL,
            symbol=symbol,
            reason=reason,
            confidence=confidence,
            quantity=quantity,
            stop_loss=Decimal(str(stop_loss)) if stop_loss is not None else None,
            take_profit=Decimal(str(take_profit)) if take_profit is not None else None,
        )
