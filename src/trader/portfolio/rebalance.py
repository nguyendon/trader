"""Portfolio rebalancing engine for maintaining target allocations."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING

from loguru import logger

from trader.core.models import Order, OrderSide, OrderType

if TYPE_CHECKING:
    from trader.broker.base import BaseBroker
    from trader.core.models import Position


class RebalanceAction(str, Enum):
    """Type of rebalance action."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class PortfolioAllocation:
    """Target allocation for a symbol in the portfolio.

    Attributes:
        symbol: Stock symbol
        target_pct: Target percentage of portfolio (0.0-1.0)
        min_pct: Minimum acceptable percentage (optional)
        max_pct: Maximum acceptable percentage (optional)
    """

    symbol: str
    target_pct: float
    min_pct: float | None = None
    max_pct: float | None = None

    def __post_init__(self) -> None:
        """Validate allocation."""
        if not 0 <= self.target_pct <= 1:
            raise ValueError(
                f"target_pct must be between 0 and 1, got {self.target_pct}"
            )
        if self.min_pct is not None and not 0 <= self.min_pct <= 1:
            raise ValueError(f"min_pct must be between 0 and 1, got {self.min_pct}")
        if self.max_pct is not None and not 0 <= self.max_pct <= 1:
            raise ValueError(f"max_pct must be between 0 and 1, got {self.max_pct}")
        if (
            self.min_pct is not None
            and self.max_pct is not None
            and self.min_pct > self.max_pct
        ):
            raise ValueError("min_pct cannot be greater than max_pct")


@dataclass
class RebalanceConfig:
    """Configuration for portfolio rebalancing.

    Attributes:
        allocations: Target allocations for each symbol
        drift_threshold: Minimum drift from target to trigger rebalance (default 5%)
        min_trade_value: Minimum trade value in dollars (avoid tiny trades)
        sell_first: Whether to execute sells before buys (frees up cash)
        dry_run: If True, only calculate orders without executing
    """

    allocations: list[PortfolioAllocation]
    drift_threshold: float = 0.05  # 5% drift triggers rebalance
    min_trade_value: float = 100.0  # Minimum $100 trade
    sell_first: bool = True  # Execute sells first to free cash
    dry_run: bool = False  # If True, don't execute orders

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not self.allocations:
            raise ValueError("Must have at least one allocation")

        total = sum(a.target_pct for a in self.allocations)
        if abs(total - 1.0) > 0.01:  # Allow 1% tolerance
            raise ValueError(f"Allocations must sum to 1.0, got {total:.2f}")

    def get_allocation(self, symbol: str) -> PortfolioAllocation | None:
        """Get allocation for a symbol."""
        for alloc in self.allocations:
            if alloc.symbol == symbol:
                return alloc
        return None

    @property
    def symbols(self) -> list[str]:
        """Get all target symbols."""
        return [a.symbol for a in self.allocations]


@dataclass
class RebalanceOrder:
    """A single rebalance order with context.

    Attributes:
        symbol: Stock symbol
        action: BUY, SELL, or HOLD
        quantity: Number of shares
        current_pct: Current allocation percentage
        target_pct: Target allocation percentage
        drift_pct: Difference from target
        trade_value: Estimated trade value in dollars
        order: The Order object to submit (None if HOLD)
    """

    symbol: str
    action: RebalanceAction
    quantity: int
    current_pct: float
    target_pct: float
    drift_pct: float
    trade_value: float
    order: Order | None = None


@dataclass
class RebalanceResult:
    """Result of a rebalance operation.

    Attributes:
        orders: List of rebalance orders
        total_buys: Total value of buy orders
        total_sells: Total value of sell orders
        net_cash_change: Net change in cash (sells - buys)
        executed: Whether orders were executed
        errors: Any errors encountered
    """

    orders: list[RebalanceOrder]
    total_buys: float = 0.0
    total_sells: float = 0.0
    net_cash_change: float = 0.0
    executed: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def needs_rebalance(self) -> bool:
        """Check if any rebalancing is needed."""
        return any(o.action != RebalanceAction.HOLD for o in self.orders)

    @property
    def buy_orders(self) -> list[RebalanceOrder]:
        """Get all buy orders."""
        return [o for o in self.orders if o.action == RebalanceAction.BUY]

    @property
    def sell_orders(self) -> list[RebalanceOrder]:
        """Get all sell orders."""
        return [o for o in self.orders if o.action == RebalanceAction.SELL]


class RebalanceEngine:
    """
    Engine for calculating and executing portfolio rebalances.

    The engine compares current portfolio allocations against targets
    and generates orders to bring the portfolio back into balance.

    Example:
        config = RebalanceConfig(
            allocations=[
                PortfolioAllocation("AAPL", 0.30),
                PortfolioAllocation("MSFT", 0.30),
                PortfolioAllocation("GOOGL", 0.40),
            ],
            drift_threshold=0.05,
        )

        engine = RebalanceEngine(broker, config)
        result = await engine.calculate_rebalance()

        if result.needs_rebalance:
            await engine.execute_rebalance(result)
    """

    def __init__(self, broker: BaseBroker, config: RebalanceConfig) -> None:
        """Initialize rebalance engine.

        Args:
            broker: Broker for fetching positions and executing orders
            config: Rebalance configuration with target allocations
        """
        self.broker = broker
        self.config = config

    async def get_current_allocations(self) -> dict[str, float]:
        """Get current portfolio allocations as percentages.

        Returns:
            Dict mapping symbol to current allocation (0.0-1.0)
        """
        positions = await self.broker.get_positions()
        account_value = await self.broker.get_account_value()

        if account_value <= 0:
            return {}

        allocations = {}
        for position in positions:
            if position.market_value:
                allocations[position.symbol] = float(position.market_value) / float(
                    account_value
                )

        return allocations

    async def get_cash_allocation(self) -> float:
        """Get current cash allocation as percentage."""
        account_value = await self.broker.get_account_value()
        cash = await self.broker.get_cash()

        if account_value <= 0:
            return 1.0

        return float(cash) / float(account_value)

    async def calculate_rebalance(self) -> RebalanceResult:
        """Calculate orders needed to rebalance portfolio.

        Returns:
            RebalanceResult with calculated orders
        """
        positions = await self.broker.get_positions()
        account_value = await self.broker.get_account_value()

        if account_value <= 0:
            return RebalanceResult(
                orders=[],
                errors=["Account value is zero or negative"],
            )

        # Build position lookup
        position_map: dict[str, Position] = {p.symbol: p for p in positions}
        current_allocations = await self.get_current_allocations()

        orders: list[RebalanceOrder] = []
        total_buys = 0.0
        total_sells = 0.0

        # Process each target allocation
        for alloc in self.config.allocations:
            symbol = alloc.symbol
            target_pct = alloc.target_pct
            current_pct = current_allocations.get(symbol, 0.0)
            drift_pct = current_pct - target_pct

            # Get current price
            try:
                if symbol in position_map:
                    current_price = position_map[symbol].current_price
                    if current_price is None:
                        current_price = await self.broker.get_latest_price(symbol)
                else:
                    current_price = await self.broker.get_latest_price(symbol)
            except Exception as e:
                logger.error(f"Failed to get price for {symbol}: {e}")
                orders.append(
                    RebalanceOrder(
                        symbol=symbol,
                        action=RebalanceAction.HOLD,
                        quantity=0,
                        current_pct=current_pct,
                        target_pct=target_pct,
                        drift_pct=drift_pct,
                        trade_value=0.0,
                    )
                )
                continue

            # Calculate target value and difference
            target_value = float(account_value) * target_pct
            current_value = float(account_value) * current_pct
            value_difference = target_value - current_value

            # Calculate shares needed
            shares_needed = int(abs(value_difference) / float(current_price))
            trade_value = shares_needed * float(current_price)

            # Check if rebalance is needed
            if abs(drift_pct) < self.config.drift_threshold:
                # Within threshold, no action needed
                orders.append(
                    RebalanceOrder(
                        symbol=symbol,
                        action=RebalanceAction.HOLD,
                        quantity=0,
                        current_pct=current_pct,
                        target_pct=target_pct,
                        drift_pct=drift_pct,
                        trade_value=0.0,
                    )
                )
                continue

            # Check minimum trade value
            if trade_value < self.config.min_trade_value:
                logger.debug(
                    f"{symbol}: Trade value ${trade_value:.2f} below minimum "
                    f"${self.config.min_trade_value:.2f}, skipping"
                )
                orders.append(
                    RebalanceOrder(
                        symbol=symbol,
                        action=RebalanceAction.HOLD,
                        quantity=0,
                        current_pct=current_pct,
                        target_pct=target_pct,
                        drift_pct=drift_pct,
                        trade_value=0.0,
                    )
                )
                continue

            if value_difference > 0:
                # Need to buy more
                action = RebalanceAction.BUY
                side = OrderSide.BUY
                total_buys += trade_value
            else:
                # Need to sell some
                action = RebalanceAction.SELL
                side = OrderSide.SELL
                total_sells += trade_value

                # Don't sell more than we have
                if symbol in position_map:
                    max_shares = position_map[symbol].quantity
                    shares_needed = min(shares_needed, max_shares)
                    trade_value = shares_needed * float(current_price)
                else:
                    # No position to sell
                    shares_needed = 0
                    trade_value = 0.0

            if shares_needed > 0:
                order = Order(
                    symbol=symbol,
                    side=side,
                    quantity=shares_needed,
                    order_type=OrderType.MARKET,
                )
            else:
                order = None
                action = RebalanceAction.HOLD

            orders.append(
                RebalanceOrder(
                    symbol=symbol,
                    action=action,
                    quantity=shares_needed,
                    current_pct=current_pct,
                    target_pct=target_pct,
                    drift_pct=drift_pct,
                    trade_value=trade_value,
                    order=order,
                )
            )

        # Check for positions not in target allocations (to be sold)
        for symbol, position in position_map.items():
            if symbol not in self.config.symbols:
                current_pct = float(position.market_value or 0) / float(account_value)
                current_price = position.current_price or Decimal(0)
                trade_value = float(position.market_value or 0)

                order = Order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=position.quantity,
                    order_type=OrderType.MARKET,
                )

                orders.append(
                    RebalanceOrder(
                        symbol=symbol,
                        action=RebalanceAction.SELL,
                        quantity=position.quantity,
                        current_pct=current_pct,
                        target_pct=0.0,  # Not in target
                        drift_pct=current_pct,  # All is drift
                        trade_value=trade_value,
                        order=order,
                    )
                )
                total_sells += trade_value

        return RebalanceResult(
            orders=orders,
            total_buys=total_buys,
            total_sells=total_sells,
            net_cash_change=total_sells - total_buys,
        )

    async def execute_rebalance(self, result: RebalanceResult) -> RebalanceResult:
        """Execute rebalance orders.

        Args:
            result: RebalanceResult from calculate_rebalance()

        Returns:
            Updated RebalanceResult with execution status
        """
        if self.config.dry_run:
            logger.info("Dry run mode - orders not executed")
            return result

        if not result.needs_rebalance:
            logger.info("No rebalancing needed")
            result.executed = True
            return result

        executed_orders = []
        errors = []

        # Determine execution order
        if self.config.sell_first:
            order_sequence = result.sell_orders + result.buy_orders
        else:
            order_sequence = result.buy_orders + result.sell_orders

        for rebalance_order in order_sequence:
            if rebalance_order.order is None:
                continue

            try:
                logger.info(
                    f"Executing {rebalance_order.action.value.upper()} "
                    f"{rebalance_order.quantity} {rebalance_order.symbol} "
                    f"(${rebalance_order.trade_value:,.2f})"
                )

                submitted_order = await self.broker.submit_order(rebalance_order.order)
                executed_orders.append(rebalance_order)

                logger.info(
                    f"Order {submitted_order.order_id} submitted: "
                    f"{submitted_order.status.value}"
                )

            except Exception as e:
                error_msg = f"Failed to execute {rebalance_order.symbol}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

        result.executed = len(errors) == 0
        result.errors = errors
        return result

    async def rebalance(self) -> RebalanceResult:
        """Calculate and execute rebalance in one step.

        Returns:
            RebalanceResult with execution status
        """
        result = await self.calculate_rebalance()

        if result.needs_rebalance:
            result = await self.execute_rebalance(result)

        return result


def create_equal_weight_config(
    symbols: list[str],
    drift_threshold: float = 0.05,
    min_trade_value: float = 100.0,
) -> RebalanceConfig:
    """Create a config with equal weights for all symbols.

    Args:
        symbols: List of symbols to include
        drift_threshold: Rebalance threshold (default 5%)
        min_trade_value: Minimum trade value (default $100)

    Returns:
        RebalanceConfig with equal allocations
    """
    weight = 1.0 / len(symbols)
    allocations = [PortfolioAllocation(symbol, weight) for symbol in symbols]

    return RebalanceConfig(
        allocations=allocations,
        drift_threshold=drift_threshold,
        min_trade_value=min_trade_value,
    )


def create_weighted_config(
    weights: dict[str, float],
    drift_threshold: float = 0.05,
    min_trade_value: float = 100.0,
) -> RebalanceConfig:
    """Create a config with custom weights.

    Args:
        weights: Dict mapping symbol to weight (will be normalized)
        drift_threshold: Rebalance threshold (default 5%)
        min_trade_value: Minimum trade value (default $100)

    Returns:
        RebalanceConfig with specified allocations

    Example:
        config = create_weighted_config({
            "AAPL": 3,  # 30%
            "MSFT": 3,  # 30%
            "GOOGL": 4,  # 40%
        })
    """
    total_weight = sum(weights.values())
    allocations = [
        PortfolioAllocation(symbol, weight / total_weight)
        for symbol, weight in weights.items()
    ]

    return RebalanceConfig(
        allocations=allocations,
        drift_threshold=drift_threshold,
        min_trade_value=min_trade_value,
    )
