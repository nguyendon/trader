"""Tests for broker implementations."""

from __future__ import annotations

from decimal import Decimal

import pytest

from trader.broker.paper import PaperBroker
from trader.core.models import Order, OrderSide, OrderStatus, OrderType


class TestPaperBroker:
    """Tests for PaperBroker."""

    @pytest.fixture
    def broker(self) -> PaperBroker:
        """Create a paper broker for testing."""
        return PaperBroker(initial_capital=100_000.0, commission=0.0)

    @pytest.mark.asyncio
    async def test_initial_state(self, broker: PaperBroker) -> None:
        """Test initial broker state."""
        await broker.connect()

        assert broker.name == "paper"
        assert broker.is_paper is True
        assert await broker.get_cash() == Decimal("100000")
        assert await broker.get_account_value() == Decimal("100000")
        assert await broker.get_positions() == []

    @pytest.mark.asyncio
    async def test_buy_order(self, broker: PaperBroker) -> None:
        """Test buying shares."""
        await broker.connect()
        broker.set_price("AAPL", 150.0)

        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
        )

        result = await broker.submit_order(order)

        assert result.status == OrderStatus.FILLED
        assert result.filled_quantity == 100
        assert result.filled_avg_price == Decimal("150")

        # Check position
        position = await broker.get_position("AAPL")
        assert position is not None
        assert position.quantity == 100
        assert position.avg_entry_price == Decimal("150")

        # Check cash
        expected_cash = Decimal("100000") - Decimal("15000")  # 100 * 150
        assert await broker.get_cash() == expected_cash

    @pytest.mark.asyncio
    async def test_sell_order(self, broker: PaperBroker) -> None:
        """Test selling shares."""
        await broker.connect()
        broker.set_price("AAPL", 150.0)

        # Buy first
        buy_order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
        )
        await broker.submit_order(buy_order)

        # Price goes up
        broker.set_price("AAPL", 160.0)

        # Sell
        sell_order = Order(
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=100,
            order_type=OrderType.MARKET,
        )
        result = await broker.submit_order(sell_order)

        assert result.status == OrderStatus.FILLED
        assert result.filled_quantity == 100
        assert result.filled_avg_price == Decimal("160")

        # Position should be closed
        position = await broker.get_position("AAPL")
        assert position is None

        # Cash should reflect profit
        # Started with 100k, bought at 150 (spent 15k), sold at 160 (got 16k)
        expected_cash = Decimal("100000") - Decimal("15000") + Decimal("16000")
        assert await broker.get_cash() == expected_cash

    @pytest.mark.asyncio
    async def test_buy_insufficient_funds(self, broker: PaperBroker) -> None:
        """Test buying with insufficient funds."""
        await broker.connect()
        broker.set_price("AAPL", 150.0)

        # Try to buy more than we can afford
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=1000,  # Would cost 150k, we only have 100k
            order_type=OrderType.MARKET,
        )

        result = await broker.submit_order(order)

        assert result.status == OrderStatus.REJECTED

    @pytest.mark.asyncio
    async def test_sell_insufficient_shares(self, broker: PaperBroker) -> None:
        """Test selling more shares than owned."""
        await broker.connect()
        broker.set_price("AAPL", 150.0)

        # Buy some shares
        buy_order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=50,
            order_type=OrderType.MARKET,
        )
        await broker.submit_order(buy_order)

        # Try to sell more than we own
        sell_order = Order(
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=100,
            order_type=OrderType.MARKET,
        )
        result = await broker.submit_order(sell_order)

        assert result.status == OrderStatus.REJECTED

    @pytest.mark.asyncio
    async def test_sell_no_position(self, broker: PaperBroker) -> None:
        """Test selling with no position."""
        await broker.connect()
        broker.set_price("AAPL", 150.0)

        order = Order(
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=100,
            order_type=OrderType.MARKET,
        )

        result = await broker.submit_order(order)

        assert result.status == OrderStatus.REJECTED

    @pytest.mark.asyncio
    async def test_commission(self) -> None:
        """Test commission is applied."""
        broker = PaperBroker(initial_capital=100_000.0, commission=10.0)
        await broker.connect()
        broker.set_price("AAPL", 100.0)

        # Buy 100 shares at $100 = $10,000 + $10 commission
        buy_order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
        )
        await broker.submit_order(buy_order)

        expected_cash = Decimal("100000") - Decimal("10000") - Decimal("10")
        assert await broker.get_cash() == expected_cash

        # Sell 100 shares at $100 = $10,000 - $10 commission
        sell_order = Order(
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=100,
            order_type=OrderType.MARKET,
        )
        await broker.submit_order(sell_order)

        expected_cash = Decimal("100000") - Decimal("20")  # Two commissions
        assert await broker.get_cash() == expected_cash

    @pytest.mark.asyncio
    async def test_multiple_positions(self, broker: PaperBroker) -> None:
        """Test managing multiple positions."""
        await broker.connect()
        broker.set_price("AAPL", 150.0)
        broker.set_price("MSFT", 300.0)

        # Buy AAPL
        await broker.submit_order(
            Order(symbol="AAPL", side=OrderSide.BUY, quantity=50, order_type=OrderType.MARKET)
        )

        # Buy MSFT
        await broker.submit_order(
            Order(symbol="MSFT", side=OrderSide.BUY, quantity=30, order_type=OrderType.MARKET)
        )

        positions = await broker.get_positions()
        assert len(positions) == 2

        symbols = {p.symbol for p in positions}
        assert symbols == {"AAPL", "MSFT"}

    @pytest.mark.asyncio
    async def test_add_to_position(self, broker: PaperBroker) -> None:
        """Test adding to existing position updates average price."""
        await broker.connect()
        broker.set_price("AAPL", 100.0)

        # Buy 100 at $100
        await broker.submit_order(
            Order(symbol="AAPL", side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET)
        )

        # Price goes up, buy more
        broker.set_price("AAPL", 120.0)
        await broker.submit_order(
            Order(symbol="AAPL", side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET)
        )

        position = await broker.get_position("AAPL")
        assert position is not None
        assert position.quantity == 200
        # Average: (100*100 + 100*120) / 200 = 110
        assert position.avg_entry_price == Decimal("110")

    @pytest.mark.asyncio
    async def test_partial_sell(self, broker: PaperBroker) -> None:
        """Test selling partial position."""
        await broker.connect()
        broker.set_price("AAPL", 100.0)

        # Buy 100 shares
        await broker.submit_order(
            Order(symbol="AAPL", side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET)
        )

        # Sell 50
        await broker.submit_order(
            Order(symbol="AAPL", side=OrderSide.SELL, quantity=50, order_type=OrderType.MARKET)
        )

        position = await broker.get_position("AAPL")
        assert position is not None
        assert position.quantity == 50

    @pytest.mark.asyncio
    async def test_context_manager(self, broker: PaperBroker) -> None:
        """Test async context manager."""
        async with broker:
            assert await broker.is_connected()
            broker.set_price("AAPL", 100.0)
            assert await broker.get_latest_price("AAPL") == Decimal("100")

    @pytest.mark.asyncio
    async def test_reset(self, broker: PaperBroker) -> None:
        """Test broker reset."""
        await broker.connect()
        broker.set_price("AAPL", 100.0)

        await broker.submit_order(
            Order(symbol="AAPL", side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET)
        )

        broker.reset()

        assert await broker.get_cash() == Decimal("100000")
        assert await broker.get_positions() == []
