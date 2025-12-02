"""Trailing stop manager using Alpaca WebSocket streaming."""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from alpaca.trading.stream import TradingStream

    from trader.broker.alpaca import AlpacaBroker


@dataclass
class TrailingStopOrder:
    """Represents a trailing stop order to be managed."""

    symbol: str
    quantity: int
    trail_percent: float | None = None  # e.g., 0.05 = 5%
    trail_price: Decimal | None = None  # Fixed dollar amount
    side: str = "sell"  # sell for long positions, buy for short
    entry_order_id: str | None = None  # ID of the entry order we're tracking
    trailing_stop_order_id: str | None = None  # ID of submitted trailing stop
    status: str = "pending"  # pending, active, filled, cancelled
    high_water_mark: Decimal | None = None  # Highest price seen (for sells)
    low_water_mark: Decimal | None = None  # Lowest price seen (for buys)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    activated_at: datetime | None = None


class TrailingStopManager:
    """
    Manages trailing stop orders using Alpaca's WebSocket streaming.

    This manager solves the limitation that Alpaca doesn't support trailing
    stops as legs of bracket orders. Instead, it:

    1. Monitors entry order fills via WebSocket
    2. When an entry fills, submits a separate trailing stop order
    3. Tracks the trailing stop status

    Usage:
        manager = TrailingStopManager(broker)
        await manager.start()

        # Register a trailing stop to be activated on fill
        manager.register_trailing_stop(
            entry_order_id="abc123",
            symbol="AAPL",
            quantity=100,
            trail_percent=0.05,  # 5% trailing stop
        )

        # When entry order fills, manager automatically submits trailing stop

        await manager.stop()
    """

    def __init__(self, broker: AlpacaBroker) -> None:
        """Initialize the trailing stop manager.

        Args:
            broker: AlpacaBroker instance for submitting orders
        """
        self.broker = broker
        self._pending_stops: dict[str, TrailingStopOrder] = {}  # entry_order_id -> stop
        self._active_stops: dict[str, TrailingStopOrder] = {}  # trailing_order_id -> stop
        self._stream: TradingStream | None = None
        self._stream_task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        """Start the WebSocket stream and begin monitoring."""
        if self._running:
            logger.warning("TrailingStopManager already running")
            return

        try:
            from alpaca.trading.stream import TradingStream

            # Get credentials from broker
            api_key = self.broker._api_key
            secret_key = self.broker._secret_key
            paper = self.broker._paper

            self._stream = TradingStream(
                api_key=api_key,
                secret_key=secret_key,
                paper=paper,
            )

            # Subscribe to trade updates
            self._stream.subscribe_trade_updates(self._handle_trade_update)

            self._running = True
            logger.info("TrailingStopManager started, listening for trade updates")

            # Run stream in background task
            self._stream_task = asyncio.create_task(self._run_stream())

        except ImportError:
            logger.error("alpaca-py not installed, trailing stop manager unavailable")
            raise
        except Exception as e:
            logger.error(f"Failed to start TrailingStopManager: {e}")
            raise

    async def _run_stream(self) -> None:
        """Run the WebSocket stream."""
        try:
            if self._stream:
                await self._stream._run_forever()
        except asyncio.CancelledError:
            logger.debug("Stream task cancelled")
        except Exception as e:
            logger.error(f"Stream error: {e}")
            self._running = False

    async def stop(self) -> None:
        """Stop the WebSocket stream."""
        self._running = False

        if self._stream_task:
            self._stream_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._stream_task

        if self._stream:
            try:
                await self._stream.close()
            except Exception as e:
                logger.debug(f"Error closing stream: {e}")

        self._stream = None
        self._stream_task = None
        logger.info("TrailingStopManager stopped")

    def register_trailing_stop(
        self,
        entry_order_id: str,
        symbol: str,
        quantity: int,
        trail_percent: float | None = None,
        trail_price: Decimal | None = None,
        side: str = "sell",
    ) -> TrailingStopOrder:
        """Register a trailing stop to be activated when entry order fills.

        Args:
            entry_order_id: The broker order ID to monitor for fills
            symbol: Symbol for the trailing stop
            quantity: Quantity to protect with trailing stop
            trail_percent: Trail by percentage (e.g., 0.05 = 5%)
            trail_price: Trail by fixed dollar amount
            side: "sell" for long positions, "buy" for short positions

        Returns:
            TrailingStopOrder tracking object
        """
        if trail_percent is None and trail_price is None:
            raise ValueError("Either trail_percent or trail_price must be specified")

        stop = TrailingStopOrder(
            symbol=symbol,
            quantity=quantity,
            trail_percent=trail_percent,
            trail_price=trail_price,
            side=side,
            entry_order_id=entry_order_id,
            status="pending",
        )

        self._pending_stops[entry_order_id] = stop
        logger.info(
            f"Registered trailing stop for {symbol}: "
            f"trail_pct={trail_percent}, trail_price={trail_price}, "
            f"waiting for order {entry_order_id} to fill"
        )

        return stop

    def unregister_trailing_stop(self, entry_order_id: str) -> bool:
        """Remove a pending trailing stop registration.

        Args:
            entry_order_id: The entry order ID to unregister

        Returns:
            True if found and removed, False otherwise
        """
        if entry_order_id in self._pending_stops:
            del self._pending_stops[entry_order_id]
            logger.info(f"Unregistered trailing stop for order {entry_order_id}")
            return True
        return False

    async def _handle_trade_update(self, data: Any) -> None:
        """Handle trade update events from WebSocket.

        Args:
            data: Trade update data from Alpaca
        """
        try:
            event = data.event
            order = data.order

            order_id = str(order.id)
            symbol = order.symbol

            logger.debug(f"Trade update: {event} for {symbol} (order {order_id})")

            if event == "fill":
                await self._handle_fill(order_id, order)
            elif event == "partial_fill":
                await self._handle_partial_fill(order_id, order)
            elif event == "canceled":
                self._handle_cancelled(order_id)
            elif event == "rejected":
                self._handle_rejected(order_id, order)

        except Exception as e:
            logger.error(f"Error handling trade update: {e}")

    async def _handle_fill(self, order_id: str, order: Any) -> None:
        """Handle order fill event - activate trailing stop if registered.

        Args:
            order_id: The filled order ID
            order: Order data from Alpaca
        """
        # Check if this is an entry order we're tracking
        if order_id in self._pending_stops:
            stop = self._pending_stops.pop(order_id)
            logger.info(
                f"Entry order {order_id} filled for {stop.symbol}, "
                f"activating trailing stop"
            )
            await self._activate_trailing_stop(stop, order)

        # Check if this is a trailing stop order that filled
        if order_id in self._active_stops:
            stop = self._active_stops.pop(order_id)
            stop.status = "filled"
            logger.info(
                f"Trailing stop filled for {stop.symbol} at "
                f"{order.filled_avg_price}"
            )

    async def _handle_partial_fill(self, order_id: str, order: Any) -> None:
        """Handle partial fill event.

        For entry orders, we might want to set up partial trailing stops.
        For now, we wait for full fill.
        """
        if order_id in self._pending_stops:
            filled_qty = int(order.filled_qty)
            total_qty = int(order.qty)
            logger.info(
                f"Partial fill {filled_qty}/{total_qty} for order {order_id}, "
                f"waiting for complete fill"
            )

    def _handle_cancelled(self, order_id: str) -> None:
        """Handle order cancellation."""
        if order_id in self._pending_stops:
            stop = self._pending_stops.pop(order_id)
            stop.status = "cancelled"
            logger.info(f"Entry order cancelled, removing trailing stop for {stop.symbol}")

        if order_id in self._active_stops:
            stop = self._active_stops.pop(order_id)
            stop.status = "cancelled"
            logger.info(f"Trailing stop cancelled for {stop.symbol}")

    def _handle_rejected(self, order_id: str, order: Any) -> None:
        """Handle order rejection."""
        if order_id in self._pending_stops:
            stop = self._pending_stops.pop(order_id)
            stop.status = "rejected"
            logger.warning(f"Entry order rejected, cannot set trailing stop for {stop.symbol}")

        if order_id in self._active_stops:
            stop = self._active_stops.pop(order_id)
            stop.status = "rejected"
            logger.warning(f"Trailing stop rejected for {stop.symbol}")

    async def _activate_trailing_stop(
        self,
        stop: TrailingStopOrder,
        entry_order: Any,
    ) -> None:
        """Submit the actual trailing stop order to Alpaca.

        Args:
            stop: TrailingStopOrder to activate
            entry_order: The filled entry order
        """
        try:
            from alpaca.trading.enums import OrderSide, TimeInForce
            from alpaca.trading.requests import TrailingStopOrderRequest

            # Determine side
            side = OrderSide.SELL if stop.side == "sell" else OrderSide.BUY

            # Build request
            if stop.trail_percent is not None:
                request = TrailingStopOrderRequest(
                    symbol=stop.symbol,
                    qty=stop.quantity,
                    side=side,
                    time_in_force=TimeInForce.GTC,
                    trail_percent=stop.trail_percent * 100,  # API expects percentage
                )
            else:
                request = TrailingStopOrderRequest(
                    symbol=stop.symbol,
                    qty=stop.quantity,
                    side=side,
                    time_in_force=TimeInForce.GTC,
                    trail_price=float(stop.trail_price) if stop.trail_price else 0.0,
                )

            # Submit via broker's client
            if self.broker._client is None:
                raise RuntimeError("Broker client not initialized")
            result = self.broker._client.submit_order(order_data=request)

            stop.trailing_stop_order_id = str(getattr(result, "id", result))
            stop.status = "active"
            stop.activated_at = datetime.now(UTC)

            # Track the active trailing stop
            self._active_stops[stop.trailing_stop_order_id] = stop

            logger.info(
                f"Trailing stop activated for {stop.symbol}: "
                f"order_id={stop.trailing_stop_order_id}, "
                f"trail_pct={stop.trail_percent}, trail_price={stop.trail_price}"
            )

        except Exception as e:
            logger.error(f"Failed to activate trailing stop for {stop.symbol}: {e}")
            stop.status = "error"

    def get_pending_stops(self) -> list[TrailingStopOrder]:
        """Get all pending trailing stops waiting for entry fills."""
        return list(self._pending_stops.values())

    def get_active_stops(self) -> list[TrailingStopOrder]:
        """Get all active trailing stop orders."""
        return list(self._active_stops.values())

    async def cancel_trailing_stop(self, symbol: str) -> bool:
        """Cancel an active trailing stop for a symbol.

        Args:
            symbol: Symbol to cancel trailing stop for

        Returns:
            True if cancelled, False if not found
        """
        for order_id, stop in list(self._active_stops.items()):
            if stop.symbol == symbol:
                try:
                    await self.broker.cancel_order(order_id)
                    del self._active_stops[order_id]
                    stop.status = "cancelled"
                    logger.info(f"Cancelled trailing stop for {symbol}")
                    return True
                except Exception as e:
                    logger.error(f"Failed to cancel trailing stop for {symbol}: {e}")
                    return False
        return False
