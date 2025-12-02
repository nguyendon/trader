"""Tests for broker implementations."""

from __future__ import annotations

from decimal import Decimal

import pytest

from trader.broker.paper import PaperBroker
from trader.core.models import Order, OrderClass, OrderSide, OrderStatus, OrderType


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
            Order(
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=50,
                order_type=OrderType.MARKET,
            )
        )

        # Buy MSFT
        await broker.submit_order(
            Order(
                symbol="MSFT",
                side=OrderSide.BUY,
                quantity=30,
                order_type=OrderType.MARKET,
            )
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
            Order(
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=100,
                order_type=OrderType.MARKET,
            )
        )

        # Price goes up, buy more
        broker.set_price("AAPL", 120.0)
        await broker.submit_order(
            Order(
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=100,
                order_type=OrderType.MARKET,
            )
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
            Order(
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=100,
                order_type=OrderType.MARKET,
            )
        )

        # Sell 50
        await broker.submit_order(
            Order(
                symbol="AAPL",
                side=OrderSide.SELL,
                quantity=50,
                order_type=OrderType.MARKET,
            )
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
            Order(
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=100,
                order_type=OrderType.MARKET,
            )
        )

        broker.reset()

        assert await broker.get_cash() == Decimal("100000")
        assert await broker.get_positions() == []


class TestAlpacaBrokerBracketOrders:
    """Tests for Alpaca broker bracket order logic."""

    def test_bracket_order_detection(self) -> None:
        """Test that bracket orders are correctly identified."""
        # Simple order
        simple_order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )
        assert simple_order.order_class == OrderClass.SIMPLE

        # Bracket order with stop loss
        bracket_order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_class=OrderClass.BRACKET,
            stop_loss_price=Decimal("145.00"),
        )
        assert bracket_order.order_class == OrderClass.BRACKET

    def test_trailing_stop_order_fields(self) -> None:
        """Test trailing stop order field assignment."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_class=OrderClass.BRACKET,
            trailing_stop_pct=0.05,
            take_profit_price=Decimal("165.00"),
        )

        assert order.trailing_stop_pct == 0.05
        assert order.trailing_stop_price is None
        assert order.take_profit_price == Decimal("165.00")
        assert order.stop_loss_price is None

    def test_trailing_stop_fixed_dollar(self) -> None:
        """Test trailing stop with fixed dollar amount."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_class=OrderClass.BRACKET,
            trailing_stop_price=Decimal("5.00"),
        )

        assert order.trailing_stop_pct is None
        assert order.trailing_stop_price == Decimal("5.00")


class TestAlpacaBrokerMocked:
    """Mocked tests for Alpaca broker bracket order submission."""

    @pytest.fixture
    def mock_alpaca_broker(self, mocker):
        """Create Alpaca broker with mocked client."""
        from trader.broker.alpaca import AlpacaBroker

        broker = AlpacaBroker(
            api_key="test_key",
            secret_key="test_secret",
            paper=True,
        )

        # Mock the trading client
        mock_client = mocker.MagicMock()
        broker._client = mock_client

        # Mock order response
        mock_order = mocker.MagicMock()
        mock_order.id = "test-order-123"
        mock_order.status.value = "new"
        mock_order.filled_qty = 0
        mock_order.filled_avg_price = None
        mock_client.submit_order.return_value = mock_order

        # Mock get_latest_price for trailing stop tests
        async def mock_get_price(_symbol):
            return Decimal("150.00")

        mocker.patch.object(broker, "get_latest_price", side_effect=mock_get_price)

        return broker, mock_client

    @pytest.mark.asyncio
    async def test_submit_simple_market_order(self, mock_alpaca_broker, mocker) -> None:
        """Test submitting a simple market order."""
        broker, mock_client = mock_alpaca_broker

        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )

        result = await broker.submit_order(order)

        # Verify submit_order was called
        mock_client.submit_order.assert_called_once()
        call_args = mock_client.submit_order.call_args[0][0]

        assert call_args.symbol == "AAPL"
        assert call_args.qty == 100
        assert result.broker_order_id == "test-order-123"

    @pytest.mark.asyncio
    async def test_submit_bracket_order_with_stop_loss(
        self, mock_alpaca_broker, mocker
    ) -> None:
        """Test submitting a bracket order with stop loss."""
        broker, mock_client = mock_alpaca_broker

        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_class=OrderClass.BRACKET,
            stop_loss_price=Decimal("145.00"),
        )

        await broker.submit_order(order)

        # Verify submit_order was called with bracket order
        mock_client.submit_order.assert_called_once()
        call_args = mock_client.submit_order.call_args[0][0]

        assert call_args.symbol == "AAPL"
        assert call_args.qty == 100
        # Verify stop_loss was included
        assert call_args.stop_loss is not None

    @pytest.mark.asyncio
    async def test_submit_trailing_stop_converts_to_fixed(
        self, mock_alpaca_broker, mocker
    ) -> None:
        """Test that trailing stop is converted to fixed stop loss.

        Note: Alpaca doesn't support trailing stops in bracket order legs,
        so we convert to a fixed stop loss based on current price.
        """
        broker, mock_client = mock_alpaca_broker

        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_class=OrderClass.BRACKET,
            trailing_stop_pct=0.05,  # 5% trailing stop
        )

        result = await broker.submit_order(order)

        # Verify submit_order was called
        mock_client.submit_order.assert_called_once()
        call_args = mock_client.submit_order.call_args[0][0]

        assert call_args.symbol == "AAPL"
        assert call_args.qty == 100
        # Stop loss should be fixed at 5% below $150 = $142.50
        assert call_args.stop_loss is not None
        assert call_args.stop_loss.stop_price == 142.50
        assert result.broker_order_id == "test-order-123"

    @pytest.mark.asyncio
    async def test_submit_trailing_stop_with_take_profit(
        self, mock_alpaca_broker, mocker
    ) -> None:
        """Test trailing stop converted to fixed stop with take profit."""
        broker, mock_client = mock_alpaca_broker

        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_class=OrderClass.BRACKET,
            trailing_stop_pct=0.05,
            take_profit_price=Decimal("165.00"),
        )

        await broker.submit_order(order)

        mock_client.submit_order.assert_called_once()
        call_args = mock_client.submit_order.call_args[0][0]

        # Should have both fixed stop loss and take profit
        assert call_args.stop_loss is not None
        assert call_args.stop_loss.stop_price == 142.50  # 5% below $150
        assert call_args.take_profit is not None
        assert call_args.take_profit.limit_price == 165.00
