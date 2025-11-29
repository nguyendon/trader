"""Tests for risk management."""

from __future__ import annotations

from decimal import Decimal

import pytest

from trader.core.models import Position, Signal, SignalAction
from trader.risk.manager import RiskConfig, RiskManager


class TestRiskConfig:
    """Tests for RiskConfig."""

    def test_default_values(self) -> None:
        """Test default config values."""
        config = RiskConfig()

        assert config.max_position_size_pct == 0.10
        assert config.max_open_positions == 10
        assert config.max_daily_loss_pct == 0.02
        assert config.stop_loss_pct == 0.05

    def test_custom_values(self) -> None:
        """Test custom config values."""
        config = RiskConfig(
            max_position_size_pct=0.05,
            max_open_positions=5,
            max_daily_loss_pct=0.01,
        )

        assert config.max_position_size_pct == 0.05
        assert config.max_open_positions == 5
        assert config.max_daily_loss_pct == 0.01


class TestRiskManager:
    """Tests for RiskManager."""

    @pytest.fixture
    def manager(self) -> RiskManager:
        """Create risk manager with default config."""
        return RiskManager(RiskConfig())

    @pytest.fixture
    def strict_manager(self) -> RiskManager:
        """Create risk manager with strict limits."""
        return RiskManager(
            RiskConfig(
                max_position_size_pct=0.05,
                max_open_positions=3,
                max_daily_loss_pct=0.01,
                min_confidence=0.5,
            )
        )

    def test_hold_signal_always_approved(self, manager: RiskManager) -> None:
        """Test that hold signals are always approved."""
        signal = Signal(action=SignalAction.HOLD, symbol="AAPL")

        result = manager.check_signal(
            signal=signal,
            portfolio_value=Decimal("100000"),
            current_price=Decimal("150"),
        )

        assert result.approved is True

    def test_buy_signal_approved(self, manager: RiskManager) -> None:
        """Test buy signal approval."""
        signal = Signal(action=SignalAction.BUY, symbol="AAPL")

        result = manager.check_signal(
            signal=signal,
            portfolio_value=Decimal("100000"),
            current_price=Decimal("150"),
            open_positions=0,
        )

        assert result.approved is True
        assert result.adjusted_quantity > 0
        assert result.stop_loss is not None

    def test_buy_signal_with_position_size(self, manager: RiskManager) -> None:
        """Test position size calculation."""
        signal = Signal(action=SignalAction.BUY, symbol="AAPL")

        result = manager.check_signal(
            signal=signal,
            portfolio_value=Decimal("100000"),
            current_price=Decimal("150"),
            open_positions=0,
        )

        # Max position = 10% of 100k = 10k
        # At $150/share = 66 shares
        assert result.adjusted_quantity == 66

    def test_buy_signal_rejected_max_positions(
        self, strict_manager: RiskManager
    ) -> None:
        """Test rejection when max positions reached."""
        signal = Signal(action=SignalAction.BUY, symbol="AAPL")

        result = strict_manager.check_signal(
            signal=signal,
            portfolio_value=Decimal("100000"),
            current_price=Decimal("150"),
            open_positions=3,  # Max is 3
        )

        assert result.approved is False
        assert "Max positions" in result.reason

    def test_buy_signal_rejected_low_confidence(
        self, strict_manager: RiskManager
    ) -> None:
        """Test rejection for low confidence signals."""
        signal = Signal(action=SignalAction.BUY, symbol="AAPL", confidence=0.3)

        result = strict_manager.check_signal(
            signal=signal,
            portfolio_value=Decimal("100000"),
            current_price=Decimal("150"),
            open_positions=0,
        )

        assert result.approved is False
        assert "Confidence" in result.reason

    def test_buy_signal_rejected_daily_loss_limit(
        self, strict_manager: RiskManager
    ) -> None:
        """Test rejection when daily loss limit reached."""
        signal = Signal(action=SignalAction.BUY, symbol="AAPL", confidence=0.8)

        # Down 1.5% today (limit is 1%)
        result = strict_manager.check_signal(
            signal=signal,
            portfolio_value=Decimal("100000"),
            current_price=Decimal("150"),
            open_positions=0,
            daily_pnl=Decimal("-1500"),
        )

        assert result.approved is False
        assert "Daily loss limit" in result.reason

    def test_sell_signal_approved_with_position(
        self, manager: RiskManager
    ) -> None:
        """Test sell signal approval with existing position."""
        signal = Signal(action=SignalAction.SELL, symbol="AAPL")
        position = Position(
            symbol="AAPL",
            quantity=100,
            avg_entry_price=Decimal("150"),
        )

        result = manager.check_signal(
            signal=signal,
            portfolio_value=Decimal("100000"),
            current_price=Decimal("160"),
            existing_position=position,
        )

        assert result.approved is True
        assert result.adjusted_quantity == 100

    def test_sell_signal_rejected_no_position(self, manager: RiskManager) -> None:
        """Test sell signal rejection without position."""
        signal = Signal(action=SignalAction.SELL, symbol="AAPL")

        result = manager.check_signal(
            signal=signal,
            portfolio_value=Decimal("100000"),
            current_price=Decimal("150"),
            existing_position=None,
        )

        assert result.approved is False
        assert "No position" in result.reason

    def test_stop_loss_calculation(self, manager: RiskManager) -> None:
        """Test stop loss is calculated correctly."""
        signal = Signal(action=SignalAction.BUY, symbol="AAPL")

        result = manager.check_signal(
            signal=signal,
            portfolio_value=Decimal("100000"),
            current_price=Decimal("100"),
            open_positions=0,
        )

        # Default stop loss is 5%
        # 100 * (1 - 0.05) = 95
        assert result.stop_loss == Decimal("95")

    def test_signal_stop_loss_overrides(self, manager: RiskManager) -> None:
        """Test that signal's stop loss overrides default."""
        signal = Signal(
            action=SignalAction.BUY,
            symbol="AAPL",
            stop_loss=Decimal("90"),
        )

        result = manager.check_signal(
            signal=signal,
            portfolio_value=Decimal("100000"),
            current_price=Decimal("100"),
            open_positions=0,
        )

        assert result.stop_loss == Decimal("90")

    def test_take_profit_calculation(self) -> None:
        """Test take profit when configured."""
        manager = RiskManager(RiskConfig(take_profit_pct=0.10))
        signal = Signal(action=SignalAction.BUY, symbol="AAPL")

        result = manager.check_signal(
            signal=signal,
            portfolio_value=Decimal("100000"),
            current_price=Decimal("100"),
            open_positions=0,
        )

        # 10% take profit: 100 * 1.10 = 110
        assert result.take_profit == Decimal("110")

    def test_calculate_position_size_basic(self, manager: RiskManager) -> None:
        """Test basic position size calculation."""
        size = manager.calculate_position_size(
            portfolio_value=Decimal("100000"),
            current_price=Decimal("100"),
        )

        # 10% of 100k = 10k, at $100 = 100 shares
        assert size == 100

    def test_calculate_position_size_with_max_value(self) -> None:
        """Test position size with max value limit."""
        manager = RiskManager(
            RiskConfig(
                max_position_size_pct=0.10,
                max_position_value=5000.0,  # $5k max
            )
        )

        size = manager.calculate_position_size(
            portfolio_value=Decimal("100000"),
            current_price=Decimal("100"),
        )

        # Max would be 10k from percentage, but capped at 5k
        # 5k / 100 = 50 shares
        assert size == 50

    def test_calculate_position_size_with_existing_position(
        self, manager: RiskManager
    ) -> None:
        """Test position size accounts for existing position."""
        existing = Position(
            symbol="AAPL",
            quantity=50,
            avg_entry_price=Decimal("100"),
        )

        size = manager.calculate_position_size(
            portfolio_value=Decimal("100000"),
            current_price=Decimal("100"),
            existing_position=existing,
        )

        # Max is 100 shares (10k / 100)
        # Already have 50, so can add 50 more
        assert size == 50

    def test_calculate_position_size_existing_at_max(
        self, manager: RiskManager
    ) -> None:
        """Test position size when already at max."""
        existing = Position(
            symbol="AAPL",
            quantity=100,  # Already at max
            avg_entry_price=Decimal("100"),
        )

        size = manager.calculate_position_size(
            portfolio_value=Decimal("100000"),
            current_price=Decimal("100"),
            existing_position=existing,
        )

        assert size == 0

    def test_daily_trade_limit(self) -> None:
        """Test daily trade limit enforcement."""
        manager = RiskManager(RiskConfig(max_daily_trades=2))
        signal = Signal(action=SignalAction.BUY, symbol="AAPL")

        # Record 2 trades
        manager.record_trade()
        manager.record_trade()

        result = manager.check_signal(
            signal=signal,
            portfolio_value=Decimal("100000"),
            current_price=Decimal("100"),
        )

        assert result.approved is False
        assert "Daily trade limit" in result.reason

    def test_reset_daily_counters(self) -> None:
        """Test resetting daily counters."""
        manager = RiskManager(RiskConfig(max_daily_trades=2))

        manager.record_trade(pnl=Decimal("100"))
        manager.record_trade(pnl=Decimal("-50"))

        assert manager.daily_trades == 2
        assert manager.daily_pnl == Decimal("50")

        manager.reset_daily_counters()

        assert manager.daily_trades == 0
        assert manager.daily_pnl == Decimal("0")
