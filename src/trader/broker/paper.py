"""Paper trading broker for testing without real money."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from loguru import logger

from trader.broker.base import BaseBroker
from trader.core.models import Order, OrderSide, OrderStatus, OrderType, Position

if TYPE_CHECKING:
    pass


class PaperBroker(BaseBroker):
    """
    Paper trading broker that simulates order execution.

    This broker maintains an in-memory portfolio and simulates
    market orders at the current price. Useful for testing
    strategies without risking real money.

    Features:
    - Simulated order execution at market price
    - Position tracking with P&L
    - Commission support
    - No API credentials needed
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission: float = 0.0,
    ) -> None:
        """Initialize paper broker.

        Args:
            initial_capital: Starting cash balance
            commission: Commission per trade in dollars
        """
        self._initial_capital = Decimal(str(initial_capital))
        self._commission = Decimal(str(commission))
        self._cash = self._initial_capital
        self._positions: dict[str, Position] = {}
        self._orders: dict[str, Order] = {}
        self._prices: dict[str, Decimal] = {}
        self._connected = False

    @property
    def name(self) -> str:
        return "paper"

    @property
    def is_paper(self) -> bool:
        return True

    async def connect(self) -> None:
        """Simulate connection."""
        self._connected = True
        logger.info("Paper broker connected")

    async def disconnect(self) -> None:
        """Simulate disconnection."""
        self._connected = False
        logger.info("Paper broker disconnected")

    async def is_connected(self) -> bool:
        return self._connected

    async def get_account_value(self) -> Decimal:
        """Get total account value."""
        positions_value = sum(
            (p.market_value or Decimal(0)) for p in self._positions.values()
        )
        return self._cash + positions_value

    async def get_buying_power(self) -> Decimal:
        """Get available buying power (same as cash for paper)."""
        return self._cash

    async def get_cash(self) -> Decimal:
        """Get cash balance."""
        return self._cash

    async def get_positions(self) -> list[Position]:
        """Get all open positions."""
        return list(self._positions.values())

    async def get_position(self, symbol: str) -> Position | None:
        """Get position for a symbol."""
        return self._positions.get(symbol)

    def set_price(self, symbol: str, price: float) -> None:
        """
        Set the current price for a symbol.

        This is used to simulate market prices for order execution.

        Args:
            symbol: Stock symbol
            price: Current price
        """
        self._prices[symbol] = Decimal(str(price))

        # Update position if exists
        if symbol in self._positions:
            self._positions[symbol].update_price(Decimal(str(price)))

    async def get_latest_price(self, symbol: str) -> Decimal:
        """Get latest price for a symbol."""
        if symbol not in self._prices:
            raise ValueError(f"No price set for {symbol}. Call set_price() first.")
        return self._prices[symbol]

    async def submit_order(self, order: Order) -> Order:
        """
        Submit and immediately execute a market order.

        For paper trading, orders are filled instantly at the current price.
        """
        if order.order_type != OrderType.MARKET:
            raise NotImplementedError("Paper broker only supports market orders")

        if order.symbol not in self._prices:
            raise ValueError(f"No price set for {order.symbol}")

        price = self._prices[order.symbol]
        order.broker_order_id = f"paper_{order.order_id}"

        if order.side == OrderSide.BUY:
            # Check buying power
            cost = price * order.quantity + self._commission
            if cost > self._cash:
                order.status = OrderStatus.REJECTED
                logger.warning(
                    f"Order rejected: insufficient funds. "
                    f"Need ${cost}, have ${self._cash}"
                )
                return order

            # Execute buy
            self._cash -= cost
            self._add_to_position(order.symbol, order.quantity, price)

            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.filled_avg_price = price
            order.updated_at = datetime.utcnow()

            logger.info(
                f"BUY {order.quantity} {order.symbol} @ ${price} (cash: ${self._cash})"
            )

        else:  # SELL
            position = self._positions.get(order.symbol)
            if not position or position.quantity < order.quantity:
                order.status = OrderStatus.REJECTED
                available = position.quantity if position else 0
                logger.warning(
                    f"Order rejected: insufficient shares. "
                    f"Need {order.quantity}, have {available}"
                )
                return order

            # Execute sell
            proceeds = price * order.quantity - self._commission
            self._cash += proceeds
            self._remove_from_position(order.symbol, order.quantity)

            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.filled_avg_price = price
            order.updated_at = datetime.utcnow()

            logger.info(
                f"SELL {order.quantity} {order.symbol} @ ${price} (cash: ${self._cash})"
            )

        self._orders[order.order_id] = order
        return order

    def _add_to_position(self, symbol: str, quantity: int, price: Decimal) -> None:
        """Add shares to a position."""
        if symbol in self._positions:
            pos = self._positions[symbol]
            # Calculate new average price
            total_cost = (pos.avg_entry_price * pos.quantity) + (price * quantity)
            new_quantity = pos.quantity + quantity
            pos.avg_entry_price = total_cost / new_quantity
            pos.quantity = new_quantity
            pos.update_price(price)
        else:
            self._positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                avg_entry_price=price,
                current_price=price,
                market_value=price * quantity,
            )

    def _remove_from_position(self, symbol: str, quantity: int) -> None:
        """Remove shares from a position."""
        pos = self._positions[symbol]
        pos.quantity -= quantity

        if pos.quantity == 0:
            del self._positions[symbol]
        else:
            pos.update_price(pos.current_price or pos.avg_entry_price)

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order (paper orders are instant, so nothing to cancel)."""
        return False

    async def get_order(self, order_id: str) -> Order | None:
        """Get order by ID."""
        return self._orders.get(order_id)

    async def get_open_orders(self) -> list[Order]:
        """Get open orders (paper orders are instant, so always empty)."""
        return []

    def reset(self) -> None:
        """Reset broker to initial state."""
        self._cash = self._initial_capital
        self._positions.clear()
        self._orders.clear()
        self._prices.clear()
        logger.info("Paper broker reset to initial state")
