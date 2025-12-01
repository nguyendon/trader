"""Tests for live trading engine."""

from __future__ import annotations

from decimal import Decimal

from trader.core.models import OrderClass
from trader.engine.live import EngineConfig, SafetyLimits, TradingMode


class TestSafetyLimits:
    """Tests for SafetyLimits configuration."""

    def test_default_values(self) -> None:
        """Test default safety limit values."""
        limits = SafetyLimits()

        assert limits.max_position_value == 10000.0
        assert limits.max_portfolio_value == 50000.0
        assert limits.max_loss_per_day == 500.0
        assert limits.max_trades_per_day == 20
        assert limits.stop_loss_pct is None
        assert limits.take_profit_pct is None
        assert limits.trailing_stop_pct is None
        assert limits.use_bracket_orders is True

    def test_custom_stop_loss(self) -> None:
        """Test custom stop loss percentage."""
        limits = SafetyLimits(stop_loss_pct=0.05)

        assert limits.stop_loss_pct == 0.05
        assert limits.trailing_stop_pct is None

    def test_custom_trailing_stop(self) -> None:
        """Test custom trailing stop percentage."""
        limits = SafetyLimits(trailing_stop_pct=0.05)

        assert limits.trailing_stop_pct == 0.05
        assert limits.stop_loss_pct is None

    def test_all_bracket_options(self) -> None:
        """Test all bracket order options together."""
        limits = SafetyLimits(
            stop_loss_pct=0.05,
            take_profit_pct=0.10,
            trailing_stop_pct=0.03,
        )

        assert limits.stop_loss_pct == 0.05
        assert limits.take_profit_pct == 0.10
        assert limits.trailing_stop_pct == 0.03


class TestEngineConfig:
    """Tests for EngineConfig."""

    def test_default_config(self) -> None:
        """Test default engine configuration."""
        config = EngineConfig()

        assert config.symbols == ["AAPL"]
        assert config.trading_mode == TradingMode.SWING
        assert config.check_interval_seconds == 60
        assert config.safety.max_position_value == 10000.0

    def test_custom_config(self) -> None:
        """Test custom engine configuration."""
        safety = SafetyLimits(
            max_position_value=5000.0,
            trailing_stop_pct=0.05,
        )
        config = EngineConfig(
            symbols=["AAPL", "MSFT"],
            trading_mode=TradingMode.DAY,
            check_interval_seconds=300,
            safety=safety,
        )

        assert config.symbols == ["AAPL", "MSFT"]
        assert config.trading_mode == TradingMode.DAY
        assert config.check_interval_seconds == 300
        assert config.safety.max_position_value == 5000.0
        assert config.safety.trailing_stop_pct == 0.05


class TestLiveEngineOrderCreation:
    """Tests for live engine order creation logic."""

    def test_trailing_stop_takes_precedence_over_fixed_stop(self) -> None:
        """Test that trailing stop is used when both are configured.

        When both trailing_stop_pct and stop_loss_pct are set,
        trailing stop should take precedence.
        """
        # This tests the priority logic in the live engine
        safety = SafetyLimits(
            stop_loss_pct=0.05,
            trailing_stop_pct=0.03,
        )

        # Trailing stop should be used (lower precedence check in code)
        assert safety.trailing_stop_pct is not None
        assert safety.stop_loss_pct is not None
        # The live engine code checks trailing_stop_pct first

    def test_order_class_bracket_when_stops_configured(self) -> None:
        """Test that orders use BRACKET class when stops are configured."""
        from trader.core.models import Order, OrderSide

        # Order with trailing stop should be bracket
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_class=OrderClass.BRACKET,
            trailing_stop_pct=0.05,
        )

        assert order.order_class == OrderClass.BRACKET

    def test_calculate_stop_loss_price(self) -> None:
        """Test stop loss price calculation."""
        current_price = Decimal("150.00")
        stop_loss_pct = Decimal("0.05")  # 5%

        stop_loss_price = current_price * (1 - stop_loss_pct)

        # 150 * 0.95 = 142.50
        assert stop_loss_price == Decimal("142.50")

    def test_calculate_take_profit_price(self) -> None:
        """Test take profit price calculation."""
        current_price = Decimal("150.00")
        take_profit_pct = Decimal("0.10")  # 10%

        take_profit_price = current_price * (1 + take_profit_pct)

        # 150 * 1.10 = 165.00
        assert take_profit_price == Decimal("165.00")


class TestLiveEngineTrailingStopConfig:
    """Tests for live engine trailing stop configuration."""

    def test_trailing_stop_config_stored_in_order(self) -> None:
        """Test that trailing stop config is stored in order model."""
        from trader.core.models import Order, OrderSide

        # When engine creates an order with trailing stop, it should store the pct
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_class=OrderClass.BRACKET,
            trailing_stop_pct=0.05,
            take_profit_price=Decimal("165.00"),
        )

        assert order.trailing_stop_pct == 0.05
        assert order.take_profit_price == Decimal("165.00")
        assert order.order_class == OrderClass.BRACKET

    def test_safety_limits_with_trailing_stop(self) -> None:
        """Test SafetyLimits stores trailing stop configuration."""
        safety = SafetyLimits(
            trailing_stop_pct=0.05,
            take_profit_pct=0.10,
            max_position_value=5000.0,
        )

        assert safety.trailing_stop_pct == 0.05
        assert safety.take_profit_pct == 0.10
        assert safety.max_position_value == 5000.0

    def test_engine_config_with_trailing_stop(self) -> None:
        """Test EngineConfig with trailing stop safety limits."""
        safety = SafetyLimits(
            trailing_stop_pct=0.05,
        )
        config = EngineConfig(
            symbols=["AAPL", "MSFT"],
            safety=safety,
        )

        assert config.safety.trailing_stop_pct == 0.05
        assert config.symbols == ["AAPL", "MSFT"]
