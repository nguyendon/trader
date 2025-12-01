"""Alpaca broker implementation."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from loguru import logger

from trader.broker.base import BaseBroker
from trader.core.models import (
    Order,
    OrderClass,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)

if TYPE_CHECKING:
    from alpaca.trading.client import TradingClient


class AlpacaBroker(BaseBroker):
    """
    Alpaca broker implementation for paper and live trading.

    This broker connects to Alpaca's trading API to submit orders
    and manage positions. Supports both paper and live trading.

    Requires Alpaca API credentials to be set in environment:
    - ALPACA_API_KEY
    - ALPACA_SECRET_KEY
    - ALPACA_PAPER (true/false)
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        paper: bool = True,
    ) -> None:
        """Initialize Alpaca broker.

        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            paper: Whether to use paper trading (default True)
        """
        self._api_key = api_key
        self._secret_key = secret_key
        self._paper = paper
        self._client: TradingClient | None = None

    @property
    def name(self) -> str:
        return "alpaca"

    @property
    def is_paper(self) -> bool:
        return self._paper

    @property
    def client(self) -> TradingClient:
        """Get or create the Alpaca client."""
        if self._client is None:
            raise RuntimeError("Broker not connected. Call connect() first.")
        return self._client

    async def connect(self) -> None:
        """Connect to Alpaca API."""
        from alpaca.trading.client import TradingClient

        self._client = TradingClient(
            api_key=self._api_key,
            secret_key=self._secret_key,
            paper=self._paper,
        )

        # Verify connection by getting account
        account = self._client.get_account()
        logger.info(
            f"Connected to Alpaca ({'paper' if self._paper else 'live'}). "
            f"Account: {account.account_number}, "  # type: ignore[union-attr]
            f"Equity: ${float(account.equity):,.2f}"  # type: ignore[union-attr,arg-type]
        )

    async def disconnect(self) -> None:
        """Disconnect from Alpaca (cleanup)."""
        self._client = None
        logger.info("Disconnected from Alpaca")

    async def is_connected(self) -> bool:
        """Check if connected."""
        if self._client is None:
            return False
        try:
            self._client.get_account()
            return True
        except Exception:
            return False

    async def get_account_value(self) -> Decimal:
        """Get total account equity."""
        account = self.client.get_account()
        return Decimal(str(account.equity))  # type: ignore[union-attr]

    async def get_buying_power(self) -> Decimal:
        """Get available buying power."""
        account = self.client.get_account()
        return Decimal(str(account.buying_power))  # type: ignore[union-attr]

    async def get_cash(self) -> Decimal:
        """Get cash balance."""
        account = self.client.get_account()
        return Decimal(str(account.cash))  # type: ignore[union-attr]

    async def get_positions(self) -> list[Position]:
        """Get all open positions."""
        alpaca_positions = self.client.get_all_positions()

        positions = []
        for ap in alpaca_positions:
            pos = Position(
                symbol=ap.symbol,  # type: ignore[union-attr]
                quantity=int(ap.qty),  # type: ignore[union-attr]
                avg_entry_price=Decimal(str(ap.avg_entry_price)),  # type: ignore[union-attr]
                current_price=Decimal(str(ap.current_price)),  # type: ignore[union-attr]
                market_value=Decimal(str(ap.market_value)),  # type: ignore[union-attr]
                unrealized_pnl=Decimal(str(ap.unrealized_pl)),  # type: ignore[union-attr]
                unrealized_pnl_pct=float(ap.unrealized_plpc or 0),  # type: ignore[union-attr]
                side=OrderSide.BUY if int(ap.qty) > 0 else OrderSide.SELL,  # type: ignore[union-attr]
            )
            positions.append(pos)

        return positions

    async def get_position(self, symbol: str) -> Position | None:
        """Get position for a symbol."""
        try:
            ap = self.client.get_open_position(symbol)
            return Position(
                symbol=ap.symbol,  # type: ignore[union-attr]
                quantity=int(ap.qty),  # type: ignore[union-attr]
                avg_entry_price=Decimal(str(ap.avg_entry_price)),  # type: ignore[union-attr]
                current_price=Decimal(str(ap.current_price)),  # type: ignore[union-attr]
                market_value=Decimal(str(ap.market_value)),  # type: ignore[union-attr]
                unrealized_pnl=Decimal(str(ap.unrealized_pl)),  # type: ignore[union-attr]
                unrealized_pnl_pct=float(ap.unrealized_plpc or 0),  # type: ignore[union-attr]
                side=OrderSide.BUY if int(ap.qty) > 0 else OrderSide.SELL,  # type: ignore[union-attr]
            )
        except Exception:
            return None

    async def submit_order(self, order: Order) -> Order:
        """Submit an order to Alpaca.

        Supports simple orders and bracket orders with stop loss/take profit.
        """
        from alpaca.trading.enums import OrderSide as AlpacaSide
        from alpaca.trading.enums import TimeInForce
        from alpaca.trading.requests import (
            LimitOrderRequest,
            MarketOrderRequest,
        )

        # Map order side
        side = AlpacaSide.BUY if order.side == OrderSide.BUY else AlpacaSide.SELL

        # Check if this is a bracket order
        if order.order_class == OrderClass.BRACKET:
            return await self._submit_bracket_order(order, side)

        # Create simple order request based on type
        request: MarketOrderRequest | LimitOrderRequest
        if order.order_type == OrderType.MARKET:
            request = MarketOrderRequest(
                symbol=order.symbol,
                qty=order.quantity,
                side=side,
                time_in_force=TimeInForce.DAY,
            )
        elif order.order_type == OrderType.LIMIT:
            if order.limit_price is None:
                raise ValueError("Limit price required for limit orders")
            request = LimitOrderRequest(
                symbol=order.symbol,
                qty=order.quantity,
                side=side,
                time_in_force=TimeInForce.DAY,
                limit_price=float(order.limit_price),
            )
        else:
            raise NotImplementedError(f"Order type {order.order_type} not supported")

        # Submit order
        alpaca_order = self.client.submit_order(request)

        # Update our order with Alpaca's response
        order.broker_order_id = str(alpaca_order.id)  # type: ignore[union-attr]
        order.status = self._map_status(alpaca_order.status.value)  # type: ignore[union-attr]
        order.filled_quantity = int(alpaca_order.filled_qty or 0)  # type: ignore[union-attr]
        if alpaca_order.filled_avg_price:  # type: ignore[union-attr]
            order.filled_avg_price = Decimal(str(alpaca_order.filled_avg_price))  # type: ignore[union-attr]
        order.updated_at = datetime.utcnow()

        logger.info(
            f"Submitted {order.side.value} order: {order.quantity} {order.symbol} "
            f"(id: {order.broker_order_id}, status: {order.status.value})"
        )

        return order

    async def _submit_bracket_order(self, order: Order, side: Any) -> Order:
        """Submit a bracket order with stop loss, take profit, or trailing stop.

        Bracket orders automatically place exit orders when the entry fills.
        Supports:
        - Fixed stop loss price
        - Fixed take profit price
        - Trailing stop (percentage or fixed dollar amount) - uses OTO order

        Note: Alpaca doesn't support trailing stops in bracket order legs directly.
        For trailing stops, we use OTO (One-Triggers-Other) orders instead.
        """
        # Check if we need to use OTO for trailing stop
        if order.trailing_stop_pct is not None or order.trailing_stop_price is not None:
            return await self._submit_oto_trailing_stop(order, side)

        from alpaca.trading.enums import OrderClass as AlpacaOrderClass
        from alpaca.trading.enums import TimeInForce
        from alpaca.trading.requests import (
            MarketOrderRequest,
            StopLossRequest,
            TakeProfitRequest,
        )

        # Build stop loss leg (fixed only - trailing uses OTO)
        stop_loss: StopLossRequest | None = None
        if order.stop_loss_price:
            if order.stop_loss_limit_price:
                stop_loss = StopLossRequest(
                    stop_price=float(order.stop_loss_price),
                    limit_price=float(order.stop_loss_limit_price),
                )
            else:
                stop_loss = StopLossRequest(stop_price=float(order.stop_loss_price))

        # Build take profit leg
        take_profit = None
        if order.take_profit_price:
            take_profit = TakeProfitRequest(limit_price=float(order.take_profit_price))

        # Create bracket order
        request = MarketOrderRequest(
            symbol=order.symbol,
            qty=order.quantity,
            side=side,
            time_in_force=TimeInForce.DAY,
            order_class=AlpacaOrderClass.BRACKET,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        # Submit order
        alpaca_order = self.client.submit_order(request)

        # Update our order with Alpaca's response
        order.broker_order_id = str(alpaca_order.id)  # type: ignore[union-attr]
        order.status = self._map_status(alpaca_order.status.value)  # type: ignore[union-attr]
        order.filled_quantity = int(alpaca_order.filled_qty or 0)  # type: ignore[union-attr]
        if alpaca_order.filled_avg_price:  # type: ignore[union-attr]
            order.filled_avg_price = Decimal(str(alpaca_order.filled_avg_price))  # type: ignore[union-attr]
        order.updated_at = datetime.utcnow()

        # Log bracket order details
        sl_str = f"SL@${float(order.stop_loss_price):.2f}" if order.stop_loss_price else ""
        tp_str = f"TP@${float(order.take_profit_price):.2f}" if order.take_profit_price else ""
        bracket_str = f" [{sl_str} {tp_str}]".strip()

        logger.info(
            f"Submitted bracket {order.side.value} order: {order.quantity} {order.symbol}"
            f"{bracket_str} (id: {order.broker_order_id})"
        )

        return order

    async def _submit_oto_trailing_stop(self, order: Order, side: Any) -> Order:
        """Submit a bracket order with trailing stop approximation.

        NOTE: Alpaca doesn't support trailing stops as legs of bracket/OTO orders.
        As a workaround, we convert the trailing stop to a fixed stop loss using
        the latest price. This provides initial protection but won't trail up.

        For true trailing stops, submit a separate TrailingStopOrderRequest
        after the primary order fills using the trading stream to monitor fills.
        """
        from alpaca.trading.enums import OrderClass as AlpacaOrderClass
        from alpaca.trading.enums import TimeInForce
        from alpaca.trading.requests import (
            MarketOrderRequest,
            StopLossRequest,
            TakeProfitRequest,
        )

        # Get current price to calculate initial stop
        current_price = await self.get_latest_price(order.symbol)

        # Convert trailing stop to fixed stop loss
        if order.trailing_stop_pct is not None:
            # Calculate stop price based on trailing percentage
            if side.value == "buy":
                stop_price = float(current_price * (1 - Decimal(str(order.trailing_stop_pct))))
            else:
                stop_price = float(current_price * (1 + Decimal(str(order.trailing_stop_pct))))
        else:
            # Fixed dollar trail amount
            trail_amount = float(order.trailing_stop_price)  # type: ignore[arg-type]
            if side.value == "buy":
                stop_price = float(current_price) - trail_amount
            else:
                stop_price = float(current_price) + trail_amount

        logger.warning(
            f"Alpaca doesn't support trailing stops in bracket orders. "
            f"Using fixed stop loss at ${stop_price:.2f} instead of trailing stop."
        )

        # Build fixed stop loss
        stop_loss = StopLossRequest(stop_price=stop_price)

        # Build take profit leg
        take_profit = None
        if order.take_profit_price:
            take_profit = TakeProfitRequest(limit_price=float(order.take_profit_price))

        # Create bracket order with fixed stop
        request = MarketOrderRequest(
            symbol=order.symbol,
            qty=order.quantity,
            side=side,
            time_in_force=TimeInForce.DAY,
            order_class=AlpacaOrderClass.BRACKET,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        # Submit order
        alpaca_order = self.client.submit_order(request)

        # Update our order with Alpaca's response
        order.broker_order_id = str(alpaca_order.id)  # type: ignore[union-attr]
        order.status = self._map_status(alpaca_order.status.value)  # type: ignore[union-attr]
        order.filled_quantity = int(alpaca_order.filled_qty or 0)  # type: ignore[union-attr]
        if alpaca_order.filled_avg_price:  # type: ignore[union-attr]
            order.filled_avg_price = Decimal(str(alpaca_order.filled_avg_price))  # type: ignore[union-attr]
        order.updated_at = datetime.utcnow()

        # Log bracket order details (note: shows original trailing stop config)
        tsl_str = (
            f"TSL@{order.trailing_stop_pct:.1%}→SL@${stop_price:.2f}"
            if order.trailing_stop_pct
            else f"TSL@${float(order.trailing_stop_price):.2f}→SL@${stop_price:.2f}"  # type: ignore[arg-type]
        )
        tp_str = f"TP@${float(order.take_profit_price):.2f}" if order.take_profit_price else ""
        bracket_str = f" [{tsl_str} {tp_str}]".strip()

        logger.info(
            f"Submitted bracket {order.side.value} order: {order.quantity} {order.symbol}"
            f"{bracket_str} (id: {order.broker_order_id})"
        )

        return order

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by ID."""
        try:
            self.client.cancel_order_by_id(order_id)
            logger.info(f"Cancelled order {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    async def get_order(self, order_id: str) -> Order | None:
        """Get order by ID."""
        try:
            ao = self.client.get_order_by_id(order_id)
            return self._alpaca_order_to_order(ao)
        except Exception:
            return None

    async def get_open_orders(self) -> list[Order]:
        """Get all open orders."""
        alpaca_orders = self.client.get_orders()
        return [self._alpaca_order_to_order(ao) for ao in alpaca_orders]

    async def get_latest_price(self, symbol: str) -> Decimal:
        """Get latest price for a symbol."""
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestQuoteRequest

        data_client = StockHistoricalDataClient(
            api_key=self._api_key,
            secret_key=self._secret_key,
        )

        request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        quotes = data_client.get_stock_latest_quote(request)

        if symbol in quotes:
            # Use mid price
            quote = quotes[symbol]
            mid = (float(quote.ask_price) + float(quote.bid_price)) / 2
            return Decimal(str(mid))

        raise ValueError(f"No quote available for {symbol}")

    def _map_status(self, alpaca_status: str) -> OrderStatus:
        """Map Alpaca order status to our OrderStatus."""
        mapping = {
            "new": OrderStatus.SUBMITTED,
            "accepted": OrderStatus.SUBMITTED,
            "pending_new": OrderStatus.PENDING,
            "accepted_for_bidding": OrderStatus.SUBMITTED,
            "filled": OrderStatus.FILLED,
            "partially_filled": OrderStatus.PARTIALLY_FILLED,
            "canceled": OrderStatus.CANCELLED,
            "expired": OrderStatus.CANCELLED,
            "rejected": OrderStatus.REJECTED,
            "pending_cancel": OrderStatus.SUBMITTED,
            "pending_replace": OrderStatus.SUBMITTED,
        }
        return mapping.get(alpaca_status.lower(), OrderStatus.PENDING)

    def _alpaca_order_to_order(self, ao: Any) -> Order:
        """Convert Alpaca order to our Order model."""
        return Order(
            symbol=ao.symbol,
            side=OrderSide.BUY if ao.side.value == "buy" else OrderSide.SELL,
            quantity=int(ao.qty),
            order_type=OrderType.MARKET
            if ao.type.value == "market"
            else OrderType.LIMIT,
            limit_price=Decimal(str(ao.limit_price)) if ao.limit_price else None,
            status=self._map_status(ao.status.value),
            order_id=str(ao.client_order_id or ao.id),
            broker_order_id=str(ao.id),
            filled_quantity=int(ao.filled_qty or 0),
            filled_avg_price=Decimal(str(ao.filled_avg_price))
            if ao.filled_avg_price
            else None,
        )
