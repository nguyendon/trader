"""Tests for trailing stop manager."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trader.broker.trailing_stop_manager import TrailingStopManager, TrailingStopOrder


class TestTrailingStopOrder:
    """Tests for TrailingStopOrder dataclass."""

    def test_initialization_with_trail_percent(self) -> None:
        """Test creating order with trail percent."""
        order = TrailingStopOrder(
            symbol="AAPL",
            quantity=100,
            trail_percent=0.05,
        )
        assert order.symbol == "AAPL"
        assert order.quantity == 100
        assert order.trail_percent == 0.05
        assert order.trail_price is None
        assert order.side == "sell"
        assert order.status == "pending"

    def test_initialization_with_trail_price(self) -> None:
        """Test creating order with trail price."""
        order = TrailingStopOrder(
            symbol="MSFT",
            quantity=50,
            trail_price=Decimal("5.00"),
            side="buy",
        )
        assert order.symbol == "MSFT"
        assert order.quantity == 50
        assert order.trail_price == Decimal("5.00")
        assert order.trail_percent is None
        assert order.side == "buy"

    def test_default_timestamps(self) -> None:
        """Test default timestamp is set."""
        order = TrailingStopOrder(
            symbol="AAPL",
            quantity=100,
            trail_percent=0.05,
        )
        assert order.created_at is not None
        assert isinstance(order.created_at, datetime)
        assert order.activated_at is None


class TestTrailingStopManager:
    """Tests for TrailingStopManager."""

    @pytest.fixture
    def mock_broker(self) -> MagicMock:
        """Create mock AlpacaBroker."""
        broker = MagicMock()
        broker._api_key = "test_key"
        broker._secret_key = "test_secret"
        broker._paper = True
        broker.cancel_order = AsyncMock()
        return broker

    @pytest.fixture
    def manager(self, mock_broker: MagicMock) -> TrailingStopManager:
        """Create TrailingStopManager instance."""
        return TrailingStopManager(mock_broker)

    def test_initialization(self, manager: TrailingStopManager) -> None:
        """Test manager initialization."""
        assert manager._pending_stops == {}
        assert manager._active_stops == {}
        assert manager._stream is None
        assert manager._running is False

    def test_register_trailing_stop_with_percent(
        self, manager: TrailingStopManager
    ) -> None:
        """Test registering trailing stop with percent."""
        stop = manager.register_trailing_stop(
            entry_order_id="order123",
            symbol="AAPL",
            quantity=100,
            trail_percent=0.05,
        )

        assert stop.symbol == "AAPL"
        assert stop.quantity == 100
        assert stop.trail_percent == 0.05
        assert stop.entry_order_id == "order123"
        assert stop.status == "pending"
        assert "order123" in manager._pending_stops

    def test_register_trailing_stop_with_price(
        self, manager: TrailingStopManager
    ) -> None:
        """Test registering trailing stop with fixed price."""
        stop = manager.register_trailing_stop(
            entry_order_id="order456",
            symbol="MSFT",
            quantity=50,
            trail_price=Decimal("3.00"),
            side="buy",
        )

        assert stop.symbol == "MSFT"
        assert stop.trail_price == Decimal("3.00")
        assert stop.side == "buy"
        assert "order456" in manager._pending_stops

    def test_register_trailing_stop_no_trail_value_raises(
        self, manager: TrailingStopManager
    ) -> None:
        """Test that missing trail value raises error."""
        with pytest.raises(ValueError, match="Either trail_percent or trail_price"):
            manager.register_trailing_stop(
                entry_order_id="order789",
                symbol="AAPL",
                quantity=100,
            )

    def test_unregister_trailing_stop(self, manager: TrailingStopManager) -> None:
        """Test unregistering a pending stop."""
        manager.register_trailing_stop(
            entry_order_id="order123",
            symbol="AAPL",
            quantity=100,
            trail_percent=0.05,
        )

        result = manager.unregister_trailing_stop("order123")
        assert result is True
        assert "order123" not in manager._pending_stops

    def test_unregister_nonexistent_stop(self, manager: TrailingStopManager) -> None:
        """Test unregistering non-existent stop returns False."""
        result = manager.unregister_trailing_stop("nonexistent")
        assert result is False

    def test_get_pending_stops(self, manager: TrailingStopManager) -> None:
        """Test getting pending stops."""
        manager.register_trailing_stop(
            entry_order_id="order1",
            symbol="AAPL",
            quantity=100,
            trail_percent=0.05,
        )
        manager.register_trailing_stop(
            entry_order_id="order2",
            symbol="MSFT",
            quantity=50,
            trail_percent=0.03,
        )

        pending = manager.get_pending_stops()
        assert len(pending) == 2
        symbols = [s.symbol for s in pending]
        assert "AAPL" in symbols
        assert "MSFT" in symbols

    def test_get_active_stops(self, manager: TrailingStopManager) -> None:
        """Test getting active stops."""
        # Manually add active stop for testing
        stop = TrailingStopOrder(
            symbol="AAPL",
            quantity=100,
            trail_percent=0.05,
            status="active",
            trailing_stop_order_id="ts123",
        )
        manager._active_stops["ts123"] = stop

        active = manager.get_active_stops()
        assert len(active) == 1
        assert active[0].symbol == "AAPL"

    @pytest.mark.asyncio
    async def test_handle_fill_activates_trailing_stop(
        self, manager: TrailingStopManager
    ) -> None:
        """Test that fill event activates pending trailing stop."""
        # Register a pending stop
        manager.register_trailing_stop(
            entry_order_id="order123",
            symbol="AAPL",
            quantity=100,
            trail_percent=0.05,
        )

        # Mock the _activate_trailing_stop method
        manager._activate_trailing_stop = AsyncMock()

        # Create mock trade update data
        mock_order = MagicMock()
        mock_order.id = "order123"
        mock_order.symbol = "AAPL"

        mock_data = MagicMock()
        mock_data.event = "fill"
        mock_data.order = mock_order

        await manager._handle_trade_update(mock_data)

        # Stop should be removed from pending
        assert "order123" not in manager._pending_stops

        # Activate should have been called
        manager._activate_trailing_stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_cancelled_removes_pending_stop(
        self, manager: TrailingStopManager
    ) -> None:
        """Test that cancel event removes pending stop."""
        stop = manager.register_trailing_stop(
            entry_order_id="order123",
            symbol="AAPL",
            quantity=100,
            trail_percent=0.05,
        )

        mock_order = MagicMock()
        mock_order.id = "order123"
        mock_order.symbol = "AAPL"

        mock_data = MagicMock()
        mock_data.event = "canceled"
        mock_data.order = mock_order

        await manager._handle_trade_update(mock_data)

        assert "order123" not in manager._pending_stops
        assert stop.status == "cancelled"

    @pytest.mark.asyncio
    async def test_handle_rejected_removes_pending_stop(
        self, manager: TrailingStopManager
    ) -> None:
        """Test that reject event removes pending stop."""
        stop = manager.register_trailing_stop(
            entry_order_id="order123",
            symbol="AAPL",
            quantity=100,
            trail_percent=0.05,
        )

        mock_order = MagicMock()
        mock_order.id = "order123"
        mock_order.symbol = "AAPL"

        mock_data = MagicMock()
        mock_data.event = "rejected"
        mock_data.order = mock_order

        await manager._handle_trade_update(mock_data)

        assert "order123" not in manager._pending_stops
        assert stop.status == "rejected"

    @pytest.mark.asyncio
    async def test_handle_partial_fill_keeps_pending(
        self, manager: TrailingStopManager
    ) -> None:
        """Test that partial fill keeps stop pending."""
        manager.register_trailing_stop(
            entry_order_id="order123",
            symbol="AAPL",
            quantity=100,
            trail_percent=0.05,
        )

        mock_order = MagicMock()
        mock_order.id = "order123"
        mock_order.symbol = "AAPL"
        mock_order.filled_qty = 50
        mock_order.qty = 100

        mock_data = MagicMock()
        mock_data.event = "partial_fill"
        mock_data.order = mock_order

        await manager._handle_trade_update(mock_data)

        # Stop should still be pending
        assert "order123" in manager._pending_stops

    @pytest.mark.asyncio
    async def test_handle_trailing_stop_filled(
        self, manager: TrailingStopManager
    ) -> None:
        """Test handling when trailing stop itself fills."""
        # Add an active trailing stop
        stop = TrailingStopOrder(
            symbol="AAPL",
            quantity=100,
            trail_percent=0.05,
            status="active",
            trailing_stop_order_id="ts123",
        )
        manager._active_stops["ts123"] = stop

        mock_order = MagicMock()
        mock_order.id = "ts123"
        mock_order.symbol = "AAPL"
        mock_order.filled_avg_price = "148.50"

        mock_data = MagicMock()
        mock_data.event = "fill"
        mock_data.order = mock_order

        await manager._handle_trade_update(mock_data)

        # Stop should be removed from active
        assert "ts123" not in manager._active_stops
        assert stop.status == "filled"

    @pytest.mark.asyncio
    async def test_cancel_trailing_stop(
        self, manager: TrailingStopManager, mock_broker: MagicMock
    ) -> None:
        """Test cancelling an active trailing stop."""
        # Add an active trailing stop
        stop = TrailingStopOrder(
            symbol="AAPL",
            quantity=100,
            trail_percent=0.05,
            status="active",
            trailing_stop_order_id="ts123",
        )
        manager._active_stops["ts123"] = stop

        result = await manager.cancel_trailing_stop("AAPL")

        assert result is True
        mock_broker.cancel_order.assert_called_once_with("ts123")
        assert "ts123" not in manager._active_stops
        assert stop.status == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_trailing_stop(
        self, manager: TrailingStopManager
    ) -> None:
        """Test cancelling non-existent trailing stop returns False."""
        result = await manager.cancel_trailing_stop("NONEXISTENT")
        assert result is False

    @pytest.mark.asyncio
    async def test_stop_cleans_up(self, manager: TrailingStopManager) -> None:
        """Test that stop() cleans up resources."""
        manager._running = True
        manager._stream = MagicMock()
        manager._stream.close = AsyncMock()
        manager._stream_task = None

        await manager.stop()

        assert manager._running is False
        assert manager._stream is None

    @pytest.mark.asyncio
    async def test_start_already_running(
        self, manager: TrailingStopManager, mock_broker: MagicMock
    ) -> None:
        """Test that start() warns if already running."""
        manager._running = True

        # Should return early without error
        await manager.start()

        # Running state should be preserved
        assert manager._running is True

    @pytest.mark.asyncio
    async def test_activate_trailing_stop_with_percent(
        self, manager: TrailingStopManager, mock_broker: MagicMock
    ) -> None:
        """Test activating a trailing stop with percent."""
        stop = TrailingStopOrder(
            symbol="AAPL",
            quantity=100,
            trail_percent=0.05,
            entry_order_id="order123",
        )

        mock_order = MagicMock()
        mock_order.id = "order123"

        mock_result = MagicMock()
        mock_result.id = "ts456"
        mock_broker._client = MagicMock()
        mock_broker._client.submit_order = MagicMock(return_value=mock_result)

        # Mock the alpaca imports inside the function
        with patch.dict(
            "sys.modules",
            {
                "alpaca.trading.enums": MagicMock(),
                "alpaca.trading.requests": MagicMock(),
            },
        ):
            await manager._activate_trailing_stop(stop, mock_order)

        assert stop.trailing_stop_order_id == "ts456"
        assert stop.status == "active"
        assert stop.activated_at is not None
        assert "ts456" in manager._active_stops

    @pytest.mark.asyncio
    async def test_activate_trailing_stop_with_price(
        self, manager: TrailingStopManager, mock_broker: MagicMock
    ) -> None:
        """Test activating a trailing stop with fixed price."""
        stop = TrailingStopOrder(
            symbol="MSFT",
            quantity=50,
            trail_price=Decimal("3.00"),
            entry_order_id="order789",
        )

        mock_order = MagicMock()
        mock_order.id = "order789"

        mock_result = MagicMock()
        mock_result.id = "ts999"
        mock_broker._client = MagicMock()
        mock_broker._client.submit_order = MagicMock(return_value=mock_result)

        with patch.dict(
            "sys.modules",
            {
                "alpaca.trading.enums": MagicMock(),
                "alpaca.trading.requests": MagicMock(),
            },
        ):
            await manager._activate_trailing_stop(stop, mock_order)

        assert stop.trailing_stop_order_id == "ts999"
        assert stop.status == "active"

    @pytest.mark.asyncio
    async def test_activate_trailing_stop_error_handling(
        self, manager: TrailingStopManager, mock_broker: MagicMock
    ) -> None:
        """Test error handling during activation."""
        stop = TrailingStopOrder(
            symbol="AAPL",
            quantity=100,
            trail_percent=0.05,
            entry_order_id="order123",
        )

        mock_order = MagicMock()
        mock_order.id = "order123"

        mock_broker._client = MagicMock()
        mock_broker._client.submit_order = MagicMock(
            side_effect=Exception("API error")
        )

        with patch.dict(
            "sys.modules",
            {
                "alpaca.trading.enums": MagicMock(),
                "alpaca.trading.requests": MagicMock(),
            },
        ):
            await manager._activate_trailing_stop(stop, mock_order)

        assert stop.status == "error"
        assert stop.trailing_stop_order_id is None
