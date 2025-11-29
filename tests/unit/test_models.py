"""Tests for core domain models."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pytest

from trader.core.models import (
    Bar,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Signal,
    SignalAction,
    TimeFrame,
)


class TestBar:
    """Tests for Bar model."""

    def test_bar_creation(self, sample_bar: Bar) -> None:
        """Test basic bar creation."""
        assert sample_bar.symbol == "AAPL"
        assert sample_bar.open == Decimal("185.00")
        assert sample_bar.close == Decimal("186.25")
        assert sample_bar.volume == 1_000_000
        assert sample_bar.timeframe == TimeFrame.DAY

    def test_bar_to_dict(self, sample_bar: Bar) -> None:
        """Test bar serialization to dict."""
        data = sample_bar.to_dict()
        assert data["symbol"] == "AAPL"
        assert data["open"] == 185.0
        assert data["close"] == 186.25
        assert data["timeframe"] == "1Day"


class TestSignal:
    """Tests for Signal model."""

    def test_signal_creation(self) -> None:
        """Test basic signal creation."""
        signal = Signal(
            action=SignalAction.BUY,
            symbol="AAPL",
            quantity=100,
            reason="Test signal",
        )
        assert signal.action == SignalAction.BUY
        assert signal.symbol == "AAPL"
        assert signal.quantity == 100
        assert signal.confidence == 1.0  # Default

    def test_signal_with_confidence(self) -> None:
        """Test signal with custom confidence."""
        signal = Signal(
            action=SignalAction.SELL,
            symbol="MSFT",
            confidence=0.75,
        )
        assert signal.confidence == 0.75

    def test_signal_invalid_confidence_raises(self) -> None:
        """Test that invalid confidence raises error."""
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            Signal(action=SignalAction.BUY, symbol="AAPL", confidence=1.5)

    def test_signal_with_stop_loss_take_profit(self) -> None:
        """Test signal with risk management levels."""
        signal = Signal(
            action=SignalAction.BUY,
            symbol="AAPL",
            stop_loss=Decimal("145.00"),
            take_profit=Decimal("165.00"),
        )
        assert signal.stop_loss == Decimal("145.00")
        assert signal.take_profit == Decimal("165.00")


class TestOrder:
    """Tests for Order model."""

    def test_market_order_creation(self) -> None:
        """Test market order creation."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )
        assert order.symbol == "AAPL"
        assert order.side == OrderSide.BUY
        assert order.quantity == 100
        assert order.order_type == OrderType.MARKET
        assert order.status == OrderStatus.PENDING

    def test_limit_order_creation(self) -> None:
        """Test limit order creation."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("150.00"),
        )
        assert order.order_type == OrderType.LIMIT
        assert order.limit_price == Decimal("150.00")

    def test_limit_order_without_price_raises(self) -> None:
        """Test that limit order without price raises error."""
        with pytest.raises(ValueError, match="Limit price required"):
            Order(
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=100,
                order_type=OrderType.LIMIT,
            )

    def test_stop_order_without_price_raises(self) -> None:
        """Test that stop order without price raises error."""
        with pytest.raises(ValueError, match="Stop price required"):
            Order(
                symbol="AAPL",
                side=OrderSide.SELL,
                quantity=100,
                order_type=OrderType.STOP,
            )

    def test_order_with_zero_quantity_raises(self) -> None:
        """Test that zero quantity raises error."""
        with pytest.raises(ValueError, match="Quantity must be positive"):
            Order(symbol="AAPL", side=OrderSide.BUY, quantity=0)

    def test_order_with_negative_quantity_raises(self) -> None:
        """Test that negative quantity raises error."""
        with pytest.raises(ValueError, match="Quantity must be positive"):
            Order(symbol="AAPL", side=OrderSide.BUY, quantity=-10)

    def test_order_has_unique_id(self) -> None:
        """Test that orders have unique IDs."""
        order1 = Order(symbol="AAPL", side=OrderSide.BUY, quantity=100)
        order2 = Order(symbol="AAPL", side=OrderSide.BUY, quantity=100)
        assert order1.order_id != order2.order_id


class TestPosition:
    """Tests for Position model."""

    def test_position_creation(self, sample_position: Position) -> None:
        """Test basic position creation."""
        assert sample_position.symbol == "AAPL"
        assert sample_position.quantity == 100
        assert sample_position.avg_entry_price == Decimal("150.00")

    def test_position_is_long(self, sample_position: Position) -> None:
        """Test long position detection."""
        assert sample_position.is_long is True
        assert sample_position.is_short is False

    def test_position_is_short(self) -> None:
        """Test short position detection."""
        position = Position(
            symbol="AAPL",
            quantity=-100,
            avg_entry_price=Decimal("150.00"),
        )
        assert position.is_long is False
        assert position.is_short is True

    def test_position_update_price_long(self, sample_position: Position) -> None:
        """Test price update for long position."""
        sample_position.update_price(Decimal("160.00"))

        assert sample_position.current_price == Decimal("160.00")
        assert sample_position.market_value == Decimal("16000.00")
        # P&L: (160 - 150) * 100 = 1000
        assert sample_position.unrealized_pnl == Decimal("1000.00")
        # P&L %: 1000 / 15000 = 0.0667
        assert sample_position.unrealized_pnl_pct == pytest.approx(0.0667, rel=0.01)

    def test_position_update_price_with_loss(self) -> None:
        """Test price update with unrealized loss."""
        position = Position(
            symbol="AAPL",
            quantity=100,
            avg_entry_price=Decimal("150.00"),
        )
        position.update_price(Decimal("140.00"))

        assert position.unrealized_pnl == Decimal("-1000.00")
        assert position.unrealized_pnl_pct == pytest.approx(-0.0667, rel=0.01)
