"""Base broker interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trader.core.models import Order, Position


class BaseBroker(ABC):
    """
    Abstract base class for broker implementations.

    All brokers (Alpaca, paper, etc.) must implement this interface.
    This enables easy swapping between paper and live trading.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Broker identifier."""
        pass

    @property
    @abstractmethod
    def is_paper(self) -> bool:
        """Whether this is a paper trading account."""
        pass

    # Connection lifecycle
    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the broker."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the broker."""
        pass

    @abstractmethod
    async def is_connected(self) -> bool:
        """Check if connected to the broker."""
        pass

    # Account info
    @abstractmethod
    async def get_account_value(self) -> Decimal:
        """Get total account value (cash + positions)."""
        pass

    @abstractmethod
    async def get_buying_power(self) -> Decimal:
        """Get available buying power."""
        pass

    @abstractmethod
    async def get_cash(self) -> Decimal:
        """Get available cash balance."""
        pass

    # Positions
    @abstractmethod
    async def get_positions(self) -> list["Position"]:
        """Get all open positions."""
        pass

    @abstractmethod
    async def get_position(self, symbol: str) -> "Position | None":
        """Get position for a specific symbol."""
        pass

    # Orders
    @abstractmethod
    async def submit_order(self, order: "Order") -> "Order":
        """
        Submit an order to the broker.

        Args:
            order: Order to submit

        Returns:
            Updated order with broker_order_id and status
        """
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancelled successfully
        """
        pass

    @abstractmethod
    async def get_order(self, order_id: str) -> "Order | None":
        """Get order by ID."""
        pass

    @abstractmethod
    async def get_open_orders(self) -> list["Order"]:
        """Get all open orders."""
        pass

    # Market data
    @abstractmethod
    async def get_latest_price(self, symbol: str) -> Decimal:
        """Get latest price for a symbol."""
        pass

    # Context manager support
    async def __aenter__(self) -> "BaseBroker":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()
