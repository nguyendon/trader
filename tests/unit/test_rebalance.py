"""Tests for portfolio rebalancing."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from trader.core.models import Order, OrderSide, OrderStatus, OrderType, Position
from trader.portfolio.rebalance import (
    PortfolioAllocation,
    RebalanceAction,
    RebalanceConfig,
    RebalanceEngine,
    RebalanceOrder,
    RebalanceResult,
    create_equal_weight_config,
    create_weighted_config,
)


class TestPortfolioAllocation:
    """Tests for PortfolioAllocation dataclass."""

    def test_create_allocation(self) -> None:
        """Test basic allocation creation."""
        alloc = PortfolioAllocation(symbol="AAPL", target_pct=0.30)
        assert alloc.symbol == "AAPL"
        assert alloc.target_pct == 0.30
        assert alloc.min_pct is None
        assert alloc.max_pct is None

    def test_allocation_with_bounds(self) -> None:
        """Test allocation with min/max bounds."""
        alloc = PortfolioAllocation(
            symbol="MSFT",
            target_pct=0.25,
            min_pct=0.20,
            max_pct=0.30,
        )
        assert alloc.min_pct == 0.20
        assert alloc.max_pct == 0.30

    def test_invalid_target_pct_too_high(self) -> None:
        """Test that target_pct > 1 raises error."""
        with pytest.raises(ValueError, match="target_pct must be between 0 and 1"):
            PortfolioAllocation(symbol="AAPL", target_pct=1.5)

    def test_invalid_target_pct_negative(self) -> None:
        """Test that negative target_pct raises error."""
        with pytest.raises(ValueError, match="target_pct must be between 0 and 1"):
            PortfolioAllocation(symbol="AAPL", target_pct=-0.1)

    def test_invalid_min_pct(self) -> None:
        """Test that invalid min_pct raises error."""
        with pytest.raises(ValueError, match="min_pct must be between 0 and 1"):
            PortfolioAllocation(symbol="AAPL", target_pct=0.30, min_pct=1.5)

    def test_invalid_max_pct(self) -> None:
        """Test that invalid max_pct raises error."""
        with pytest.raises(ValueError, match="max_pct must be between 0 and 1"):
            PortfolioAllocation(symbol="AAPL", target_pct=0.30, max_pct=-0.1)

    def test_min_greater_than_max(self) -> None:
        """Test that min_pct > max_pct raises error."""
        with pytest.raises(ValueError, match="min_pct cannot be greater than max_pct"):
            PortfolioAllocation(
                symbol="AAPL",
                target_pct=0.30,
                min_pct=0.40,
                max_pct=0.20,
            )


class TestRebalanceConfig:
    """Tests for RebalanceConfig dataclass."""

    def test_create_config(self) -> None:
        """Test basic config creation."""
        allocs = [
            PortfolioAllocation(symbol="AAPL", target_pct=0.50),
            PortfolioAllocation(symbol="MSFT", target_pct=0.50),
        ]
        config = RebalanceConfig(allocations=allocs)
        assert len(config.allocations) == 2
        assert config.drift_threshold == 0.05
        assert config.min_trade_value == 100.0
        assert config.sell_first is True
        assert config.dry_run is False

    def test_custom_config(self) -> None:
        """Test config with custom values."""
        allocs = [PortfolioAllocation(symbol="AAPL", target_pct=1.0)]
        config = RebalanceConfig(
            allocations=allocs,
            drift_threshold=0.10,
            min_trade_value=500.0,
            sell_first=False,
            dry_run=True,
        )
        assert config.drift_threshold == 0.10
        assert config.min_trade_value == 500.0
        assert config.sell_first is False
        assert config.dry_run is True

    def test_empty_allocations_raises(self) -> None:
        """Test that empty allocations raise error."""
        with pytest.raises(ValueError, match="Must have at least one allocation"):
            RebalanceConfig(allocations=[])

    def test_allocations_must_sum_to_one(self) -> None:
        """Test that allocations not summing to 1.0 raises error."""
        allocs = [
            PortfolioAllocation(symbol="AAPL", target_pct=0.30),
            PortfolioAllocation(symbol="MSFT", target_pct=0.30),
        ]
        with pytest.raises(ValueError, match="Allocations must sum to 1.0"):
            RebalanceConfig(allocations=allocs)

    def test_allocations_sum_tolerance(self) -> None:
        """Test that allocations within 1% tolerance are accepted."""
        allocs = [
            PortfolioAllocation(symbol="AAPL", target_pct=0.334),
            PortfolioAllocation(symbol="MSFT", target_pct=0.333),
            PortfolioAllocation(symbol="GOOGL", target_pct=0.333),
        ]
        # Should not raise - sum is 1.0 within tolerance
        config = RebalanceConfig(allocations=allocs)
        assert len(config.allocations) == 3

    def test_get_allocation(self) -> None:
        """Test getting allocation by symbol."""
        allocs = [
            PortfolioAllocation(symbol="AAPL", target_pct=0.50),
            PortfolioAllocation(symbol="MSFT", target_pct=0.50),
        ]
        config = RebalanceConfig(allocations=allocs)

        aapl = config.get_allocation("AAPL")
        assert aapl is not None
        assert aapl.target_pct == 0.50

        googl = config.get_allocation("GOOGL")
        assert googl is None

    def test_symbols_property(self) -> None:
        """Test symbols property returns all symbols."""
        allocs = [
            PortfolioAllocation(symbol="AAPL", target_pct=0.50),
            PortfolioAllocation(symbol="MSFT", target_pct=0.50),
        ]
        config = RebalanceConfig(allocations=allocs)
        assert config.symbols == ["AAPL", "MSFT"]


class TestRebalanceResult:
    """Tests for RebalanceResult dataclass."""

    def test_needs_rebalance_true(self) -> None:
        """Test needs_rebalance when action required."""
        orders = [
            RebalanceOrder(
                symbol="AAPL",
                action=RebalanceAction.BUY,
                quantity=10,
                current_pct=0.20,
                target_pct=0.30,
                drift_pct=-0.10,
                trade_value=1500.0,
            ),
        ]
        result = RebalanceResult(orders=orders)
        assert result.needs_rebalance is True

    def test_needs_rebalance_false(self) -> None:
        """Test needs_rebalance when no action needed."""
        orders = [
            RebalanceOrder(
                symbol="AAPL",
                action=RebalanceAction.HOLD,
                quantity=0,
                current_pct=0.30,
                target_pct=0.30,
                drift_pct=0.0,
                trade_value=0.0,
            ),
        ]
        result = RebalanceResult(orders=orders)
        assert result.needs_rebalance is False

    def test_buy_orders_filter(self) -> None:
        """Test buy_orders returns only buy orders."""
        orders = [
            RebalanceOrder(
                symbol="AAPL",
                action=RebalanceAction.BUY,
                quantity=10,
                current_pct=0.20,
                target_pct=0.30,
                drift_pct=-0.10,
                trade_value=1500.0,
            ),
            RebalanceOrder(
                symbol="MSFT",
                action=RebalanceAction.SELL,
                quantity=5,
                current_pct=0.40,
                target_pct=0.30,
                drift_pct=0.10,
                trade_value=1500.0,
            ),
            RebalanceOrder(
                symbol="GOOGL",
                action=RebalanceAction.HOLD,
                quantity=0,
                current_pct=0.30,
                target_pct=0.30,
                drift_pct=0.0,
                trade_value=0.0,
            ),
        ]
        result = RebalanceResult(orders=orders)
        assert len(result.buy_orders) == 1
        assert result.buy_orders[0].symbol == "AAPL"

    def test_sell_orders_filter(self) -> None:
        """Test sell_orders returns only sell orders."""
        orders = [
            RebalanceOrder(
                symbol="AAPL",
                action=RebalanceAction.BUY,
                quantity=10,
                current_pct=0.20,
                target_pct=0.30,
                drift_pct=-0.10,
                trade_value=1500.0,
            ),
            RebalanceOrder(
                symbol="MSFT",
                action=RebalanceAction.SELL,
                quantity=5,
                current_pct=0.40,
                target_pct=0.30,
                drift_pct=0.10,
                trade_value=1500.0,
            ),
        ]
        result = RebalanceResult(orders=orders)
        assert len(result.sell_orders) == 1
        assert result.sell_orders[0].symbol == "MSFT"


class TestRebalanceEngine:
    """Tests for RebalanceEngine."""

    @pytest.fixture
    def mock_broker(self) -> MagicMock:
        """Create a mock broker."""
        broker = MagicMock()
        broker.get_positions = AsyncMock(return_value=[])
        broker.get_account_value = AsyncMock(return_value=Decimal("100000"))
        broker.get_cash = AsyncMock(return_value=Decimal("100000"))
        broker.get_latest_price = AsyncMock(return_value=Decimal("150.00"))
        broker.submit_order = AsyncMock(
            return_value=Order(
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=10,
                order_type=OrderType.MARKET,
                order_id="test-123",
                status=OrderStatus.FILLED,
            )
        )
        return broker

    @pytest.fixture
    def basic_config(self) -> RebalanceConfig:
        """Create a basic rebalance config."""
        return RebalanceConfig(
            allocations=[
                PortfolioAllocation(symbol="AAPL", target_pct=0.40),
                PortfolioAllocation(symbol="MSFT", target_pct=0.30),
                PortfolioAllocation(symbol="GOOGL", target_pct=0.30),
            ],
            drift_threshold=0.05,
        )

    @pytest.mark.asyncio
    async def test_get_current_allocations_empty(
        self, mock_broker: MagicMock, basic_config: RebalanceConfig
    ) -> None:
        """Test getting allocations with no positions."""
        engine = RebalanceEngine(mock_broker, basic_config)
        allocations = await engine.get_current_allocations()
        assert allocations == {}

    @pytest.mark.asyncio
    async def test_get_current_allocations_with_positions(
        self, mock_broker: MagicMock, basic_config: RebalanceConfig
    ) -> None:
        """Test getting allocations with existing positions."""
        mock_broker.get_positions = AsyncMock(
            return_value=[
                Position(
                    symbol="AAPL",
                    quantity=100,
                    avg_entry_price=Decimal("150.00"),
                    market_value=Decimal("40000.00"),
                    current_price=Decimal("150.00"),
                ),
                Position(
                    symbol="MSFT",
                    quantity=80,
                    avg_entry_price=Decimal("375.00"),
                    market_value=Decimal("30000.00"),
                    current_price=Decimal("375.00"),
                ),
            ]
        )

        engine = RebalanceEngine(mock_broker, basic_config)
        allocations = await engine.get_current_allocations()

        assert "AAPL" in allocations
        assert "MSFT" in allocations
        assert allocations["AAPL"] == pytest.approx(0.40, rel=0.01)
        assert allocations["MSFT"] == pytest.approx(0.30, rel=0.01)

    @pytest.mark.asyncio
    async def test_get_cash_allocation(
        self, mock_broker: MagicMock, basic_config: RebalanceConfig
    ) -> None:
        """Test getting cash allocation percentage."""
        mock_broker.get_cash = AsyncMock(return_value=Decimal("25000"))
        mock_broker.get_account_value = AsyncMock(return_value=Decimal("100000"))

        engine = RebalanceEngine(mock_broker, basic_config)
        cash_pct = await engine.get_cash_allocation()

        assert cash_pct == pytest.approx(0.25, rel=0.01)

    @pytest.mark.asyncio
    async def test_calculate_rebalance_buy_needed(
        self, mock_broker: MagicMock, basic_config: RebalanceConfig
    ) -> None:
        """Test calculating rebalance when buys are needed."""
        # No positions, all cash - need to buy everything
        engine = RebalanceEngine(mock_broker, basic_config)
        result = await engine.calculate_rebalance()

        assert result.needs_rebalance is True
        assert len(result.buy_orders) == 3
        assert result.total_buys > 0
        assert result.total_sells == 0

    @pytest.mark.asyncio
    async def test_calculate_rebalance_sell_needed(
        self, mock_broker: MagicMock, basic_config: RebalanceConfig
    ) -> None:
        """Test calculating rebalance when sells are needed."""
        # AAPL is overweight
        mock_broker.get_positions = AsyncMock(
            return_value=[
                Position(
                    symbol="AAPL",
                    quantity=400,
                    avg_entry_price=Decimal("150.00"),
                    market_value=Decimal("60000.00"),
                    current_price=Decimal("150.00"),
                ),
                Position(
                    symbol="MSFT",
                    quantity=80,
                    avg_entry_price=Decimal("375.00"),
                    market_value=Decimal("30000.00"),
                    current_price=Decimal("375.00"),
                ),
                Position(
                    symbol="GOOGL",
                    quantity=67,
                    avg_entry_price=Decimal("150.00"),
                    market_value=Decimal("10000.00"),
                    current_price=Decimal("150.00"),
                ),
            ]
        )

        engine = RebalanceEngine(mock_broker, basic_config)
        result = await engine.calculate_rebalance()

        # AAPL should be sold (60% -> 40%)
        aapl_order = next((o for o in result.orders if o.symbol == "AAPL"), None)
        assert aapl_order is not None
        assert aapl_order.action == RebalanceAction.SELL

    @pytest.mark.asyncio
    async def test_calculate_rebalance_within_threshold(
        self, mock_broker: MagicMock, basic_config: RebalanceConfig
    ) -> None:
        """Test that positions within drift threshold result in HOLD."""
        # Positions are close to target
        mock_broker.get_positions = AsyncMock(
            return_value=[
                Position(
                    symbol="AAPL",
                    quantity=267,
                    avg_entry_price=Decimal("150.00"),
                    market_value=Decimal("40000.00"),
                    current_price=Decimal("150.00"),
                ),
                Position(
                    symbol="MSFT",
                    quantity=80,
                    avg_entry_price=Decimal("375.00"),
                    market_value=Decimal("30000.00"),
                    current_price=Decimal("375.00"),
                ),
                Position(
                    symbol="GOOGL",
                    quantity=200,
                    avg_entry_price=Decimal("150.00"),
                    market_value=Decimal("30000.00"),
                    current_price=Decimal("150.00"),
                ),
            ]
        )

        engine = RebalanceEngine(mock_broker, basic_config)
        result = await engine.calculate_rebalance()

        # All should be HOLD
        for order in result.orders:
            assert order.action == RebalanceAction.HOLD

    @pytest.mark.asyncio
    async def test_calculate_rebalance_min_trade_value(
        self, mock_broker: MagicMock
    ) -> None:
        """Test that trades below min_trade_value are skipped."""
        config = RebalanceConfig(
            allocations=[
                PortfolioAllocation(symbol="AAPL", target_pct=1.0),
            ],
            min_trade_value=10000.0,  # High min trade value
            drift_threshold=0.01,  # Low drift threshold
        )

        # Position slightly off target but trade value too small
        mock_broker.get_positions = AsyncMock(
            return_value=[
                Position(
                    symbol="AAPL",
                    quantity=600,
                    avg_entry_price=Decimal("150.00"),
                    market_value=Decimal("90000.00"),
                    current_price=Decimal("150.00"),
                ),
            ]
        )

        engine = RebalanceEngine(mock_broker, config)
        result = await engine.calculate_rebalance()

        # Trade should be skipped due to min trade value
        aapl_order = result.orders[0]
        assert aapl_order.action == RebalanceAction.HOLD

    @pytest.mark.asyncio
    async def test_calculate_rebalance_sells_unwanted_positions(
        self, mock_broker: MagicMock, basic_config: RebalanceConfig
    ) -> None:
        """Test that positions not in target allocations are sold."""
        # TSLA is not in target allocations
        mock_broker.get_positions = AsyncMock(
            return_value=[
                Position(
                    symbol="TSLA",
                    quantity=50,
                    avg_entry_price=Decimal("200.00"),
                    market_value=Decimal("10000.00"),
                    current_price=Decimal("200.00"),
                ),
            ]
        )

        engine = RebalanceEngine(mock_broker, basic_config)
        result = await engine.calculate_rebalance()

        # TSLA should be sold
        tsla_order = next((o for o in result.orders if o.symbol == "TSLA"), None)
        assert tsla_order is not None
        assert tsla_order.action == RebalanceAction.SELL
        assert tsla_order.target_pct == 0.0

    @pytest.mark.asyncio
    async def test_calculate_rebalance_zero_account_value(
        self, mock_broker: MagicMock, basic_config: RebalanceConfig
    ) -> None:
        """Test handling of zero account value."""
        mock_broker.get_account_value = AsyncMock(return_value=Decimal("0"))

        engine = RebalanceEngine(mock_broker, basic_config)
        result = await engine.calculate_rebalance()

        assert result.orders == []
        assert "zero or negative" in result.errors[0].lower()

    @pytest.mark.asyncio
    async def test_execute_rebalance_dry_run(
        self, mock_broker: MagicMock, basic_config: RebalanceConfig
    ) -> None:
        """Test that dry run mode doesn't execute orders."""
        basic_config.dry_run = True

        engine = RebalanceEngine(mock_broker, basic_config)
        result = await engine.calculate_rebalance()
        result = await engine.execute_rebalance(result)

        # Orders should not be submitted
        mock_broker.submit_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_rebalance_submits_orders(
        self, mock_broker: MagicMock, basic_config: RebalanceConfig
    ) -> None:
        """Test that orders are submitted during execution."""
        engine = RebalanceEngine(mock_broker, basic_config)
        result = await engine.calculate_rebalance()
        result = await engine.execute_rebalance(result)

        # Orders should be submitted
        assert mock_broker.submit_order.call_count > 0
        assert result.executed is True

    @pytest.mark.asyncio
    async def test_execute_rebalance_sell_first(self, mock_broker: MagicMock) -> None:
        """Test that sells are executed before buys when sell_first=True."""
        config = RebalanceConfig(
            allocations=[
                PortfolioAllocation(symbol="AAPL", target_pct=0.50),
                PortfolioAllocation(symbol="MSFT", target_pct=0.50),
            ],
            sell_first=True,
        )

        # Create a result with both buy and sell orders
        buy_order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET,
        )
        sell_order = Order(
            symbol="MSFT",
            side=OrderSide.SELL,
            quantity=5,
            order_type=OrderType.MARKET,
        )

        result = RebalanceResult(
            orders=[
                RebalanceOrder(
                    symbol="AAPL",
                    action=RebalanceAction.BUY,
                    quantity=10,
                    current_pct=0.30,
                    target_pct=0.50,
                    drift_pct=-0.20,
                    trade_value=1500.0,
                    order=buy_order,
                ),
                RebalanceOrder(
                    symbol="MSFT",
                    action=RebalanceAction.SELL,
                    quantity=5,
                    current_pct=0.70,
                    target_pct=0.50,
                    drift_pct=0.20,
                    trade_value=1500.0,
                    order=sell_order,
                ),
            ],
        )

        execution_order = []

        async def track_order(order: Order) -> Order:
            execution_order.append(order.side)
            return Order(
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                order_type=order.order_type,
                order_id="test",
                status=OrderStatus.FILLED,
            )

        mock_broker.submit_order = AsyncMock(side_effect=track_order)

        engine = RebalanceEngine(mock_broker, config)
        await engine.execute_rebalance(result)

        # Sell should come before buy
        assert execution_order[0] == OrderSide.SELL
        assert execution_order[1] == OrderSide.BUY

    @pytest.mark.asyncio
    async def test_execute_rebalance_handles_errors(
        self, mock_broker: MagicMock, basic_config: RebalanceConfig
    ) -> None:
        """Test that execution errors are captured."""
        mock_broker.submit_order = AsyncMock(side_effect=Exception("API Error"))

        engine = RebalanceEngine(mock_broker, basic_config)
        result = await engine.calculate_rebalance()
        result = await engine.execute_rebalance(result)

        assert result.executed is False
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_rebalance_all_in_one(
        self, mock_broker: MagicMock, basic_config: RebalanceConfig
    ) -> None:
        """Test the combined rebalance() method."""
        engine = RebalanceEngine(mock_broker, basic_config)
        result = await engine.rebalance()

        assert result is not None
        assert result.executed is True

    @pytest.mark.asyncio
    async def test_rebalance_no_action_needed(
        self, mock_broker: MagicMock, basic_config: RebalanceConfig
    ) -> None:
        """Test rebalance when portfolio is already balanced."""
        # Set positions to match targets exactly
        mock_broker.get_positions = AsyncMock(
            return_value=[
                Position(
                    symbol="AAPL",
                    quantity=267,
                    avg_entry_price=Decimal("150.00"),
                    market_value=Decimal("40000.00"),
                    current_price=Decimal("150.00"),
                ),
                Position(
                    symbol="MSFT",
                    quantity=80,
                    avg_entry_price=Decimal("375.00"),
                    market_value=Decimal("30000.00"),
                    current_price=Decimal("375.00"),
                ),
                Position(
                    symbol="GOOGL",
                    quantity=200,
                    avg_entry_price=Decimal("150.00"),
                    market_value=Decimal("30000.00"),
                    current_price=Decimal("150.00"),
                ),
            ]
        )

        engine = RebalanceEngine(mock_broker, basic_config)
        result = await engine.rebalance()

        # Should indicate no rebalance needed
        assert result.needs_rebalance is False


class TestCreateEqualWeightConfig:
    """Tests for create_equal_weight_config helper."""

    def test_two_symbols(self) -> None:
        """Test equal weight with two symbols."""
        config = create_equal_weight_config(["AAPL", "MSFT"])

        assert len(config.allocations) == 2
        assert config.allocations[0].target_pct == 0.50
        assert config.allocations[1].target_pct == 0.50

    def test_three_symbols(self) -> None:
        """Test equal weight with three symbols."""
        config = create_equal_weight_config(["AAPL", "MSFT", "GOOGL"])

        for alloc in config.allocations:
            assert alloc.target_pct == pytest.approx(1 / 3, rel=0.01)

    def test_custom_thresholds(self) -> None:
        """Test custom drift threshold and min trade value."""
        config = create_equal_weight_config(
            ["AAPL", "MSFT"],
            drift_threshold=0.10,
            min_trade_value=500.0,
        )

        assert config.drift_threshold == 0.10
        assert config.min_trade_value == 500.0


class TestCreateWeightedConfig:
    """Tests for create_weighted_config helper."""

    def test_normalized_weights(self) -> None:
        """Test that weights are normalized."""
        config = create_weighted_config(
            {
                "AAPL": 3,
                "MSFT": 1,
            }
        )

        aapl = config.get_allocation("AAPL")
        msft = config.get_allocation("MSFT")

        assert aapl is not None
        assert msft is not None
        assert aapl.target_pct == pytest.approx(0.75, rel=0.01)
        assert msft.target_pct == pytest.approx(0.25, rel=0.01)

    def test_percentage_weights(self) -> None:
        """Test weights that look like percentages."""
        config = create_weighted_config(
            {
                "AAPL": 30,
                "MSFT": 30,
                "GOOGL": 40,
            }
        )

        assert config.get_allocation("AAPL").target_pct == pytest.approx(0.30, rel=0.01)
        assert config.get_allocation("MSFT").target_pct == pytest.approx(0.30, rel=0.01)
        assert config.get_allocation("GOOGL").target_pct == pytest.approx(
            0.40, rel=0.01
        )

    def test_custom_thresholds(self) -> None:
        """Test custom drift threshold and min trade value."""
        config = create_weighted_config(
            {"AAPL": 1},
            drift_threshold=0.03,
            min_trade_value=200.0,
        )

        assert config.drift_threshold == 0.03
        assert config.min_trade_value == 200.0


class TestRebalanceAction:
    """Tests for RebalanceAction enum."""

    def test_action_values(self) -> None:
        """Test enum values."""
        assert RebalanceAction.BUY.value == "buy"
        assert RebalanceAction.SELL.value == "sell"
        assert RebalanceAction.HOLD.value == "hold"
