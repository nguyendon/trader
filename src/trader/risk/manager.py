"""Risk management for trading."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from loguru import logger

from trader.core.models import Signal, SignalAction

if TYPE_CHECKING:
    from trader.core.models import Position


@dataclass
class RiskConfig:
    """Risk management configuration."""

    # Position sizing
    max_position_size_pct: float = 0.10  # Max 10% of portfolio per position
    max_position_value: float | None = None  # Max dollar value per position

    # Portfolio limits
    max_open_positions: int = 10  # Maximum concurrent positions
    max_portfolio_risk_pct: float = 0.20  # Max 20% of portfolio at risk

    # Trade limits
    max_daily_trades: int = 50  # Maximum trades per day
    max_daily_loss_pct: float = 0.02  # Stop trading if down 2% today

    # Per-trade risk
    stop_loss_pct: float = 0.05  # Default 5% stop loss
    take_profit_pct: float | None = None  # Optional take profit

    # Filters
    min_confidence: float = 0.0  # Minimum signal confidence to trade


@dataclass
class RiskCheckResult:
    """Result of a risk check."""

    approved: bool
    reason: str
    adjusted_quantity: int | None = None
    stop_loss: Decimal | None = None
    take_profit: Decimal | None = None


class RiskManager:
    """
    Risk manager for controlling position sizing and trade limits.

    The risk manager validates signals against risk rules and calculates
    appropriate position sizes. It can reject signals that violate
    risk limits or adjust quantities to stay within limits.

    Example:
        risk_mgr = RiskManager(RiskConfig(max_position_size_pct=0.1))

        result = risk_mgr.check_signal(
            signal=signal,
            portfolio_value=100000,
            current_price=150,
            open_positions=3,
        )

        if result.approved:
            order = create_order(quantity=result.adjusted_quantity)
    """

    def __init__(self, config: RiskConfig | None = None) -> None:
        """Initialize risk manager.

        Args:
            config: Risk configuration. Uses defaults if not provided.
        """
        self.config = config or RiskConfig()
        self._daily_trades = 0
        self._daily_pnl = Decimal(0)
        self._starting_equity = Decimal(0)

    def check_signal(
        self,
        signal: Signal,
        portfolio_value: Decimal,
        current_price: Decimal,
        open_positions: int = 0,
        existing_position: Position | None = None,
        daily_pnl: Decimal | None = None,
    ) -> RiskCheckResult:
        """
        Check if a signal passes risk rules.

        Args:
            signal: Trading signal to check
            portfolio_value: Current total portfolio value
            current_price: Current price of the symbol
            open_positions: Number of currently open positions
            existing_position: Existing position in this symbol
            daily_pnl: Today's P&L so far

        Returns:
            RiskCheckResult with approval status and adjusted parameters
        """
        # Hold signals always pass
        if signal.action == SignalAction.HOLD:
            return RiskCheckResult(approved=True, reason="Hold signal")

        # Check minimum confidence
        if signal.confidence < self.config.min_confidence:
            return RiskCheckResult(
                approved=False,
                reason=f"Confidence {signal.confidence:.2f} below minimum {self.config.min_confidence:.2f}",
            )

        # Check daily loss limit
        if daily_pnl is not None:
            daily_loss_pct = (
                float(-daily_pnl / portfolio_value) if portfolio_value > 0 else 0
            )
            if daily_loss_pct >= self.config.max_daily_loss_pct:
                return RiskCheckResult(
                    approved=False,
                    reason=f"Daily loss limit reached: {daily_loss_pct:.2%}",
                )

        # Check daily trade limit
        if self._daily_trades >= self.config.max_daily_trades:
            return RiskCheckResult(
                approved=False,
                reason=f"Daily trade limit reached: {self._daily_trades}",
            )

        # For BUY signals
        if signal.action == SignalAction.BUY:
            return self._check_buy_signal(
                signal=signal,
                portfolio_value=portfolio_value,
                current_price=current_price,
                open_positions=open_positions,
                existing_position=existing_position,
            )

        # For SELL signals
        if signal.action == SignalAction.SELL:
            return self._check_sell_signal(
                signal=signal,
                existing_position=existing_position,
                current_price=current_price,
            )

        return RiskCheckResult(approved=False, reason="Unknown signal action")

    def _check_buy_signal(
        self,
        signal: Signal,
        portfolio_value: Decimal,
        current_price: Decimal,
        open_positions: int,
        existing_position: Position | None,
    ) -> RiskCheckResult:
        """Check buy signal against risk rules."""
        # Check max positions
        if (
            open_positions >= self.config.max_open_positions
            and existing_position is None
        ):
            return RiskCheckResult(
                approved=False,
                reason=f"Max positions reached: {open_positions}",
            )

        # Calculate position size
        quantity = self.calculate_position_size(
            portfolio_value=portfolio_value,
            current_price=current_price,
            existing_position=existing_position,
        )

        if quantity <= 0:
            return RiskCheckResult(
                approved=False,
                reason="Calculated position size is zero",
            )

        # Calculate stop loss and take profit
        stop_loss = None
        take_profit = None

        if self.config.stop_loss_pct:
            stop_loss = current_price * (1 - Decimal(str(self.config.stop_loss_pct)))

        if self.config.take_profit_pct:
            take_profit = current_price * (
                1 + Decimal(str(self.config.take_profit_pct))
            )

        # Use signal's stop/take if provided
        if signal.stop_loss is not None:
            stop_loss = signal.stop_loss
        if signal.take_profit is not None:
            take_profit = signal.take_profit

        return RiskCheckResult(
            approved=True,
            reason="Buy signal approved",
            adjusted_quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

    def _check_sell_signal(
        self,
        signal: Signal,
        existing_position: Position | None,
        current_price: Decimal,
    ) -> RiskCheckResult:
        """Check sell signal against risk rules."""
        if existing_position is None:
            return RiskCheckResult(
                approved=False,
                reason="No position to sell",
            )

        if existing_position.quantity <= 0:
            return RiskCheckResult(
                approved=False,
                reason="Position quantity is zero",
            )

        # Sell entire position
        quantity = existing_position.quantity

        return RiskCheckResult(
            approved=True,
            reason="Sell signal approved",
            adjusted_quantity=quantity,
        )

    def calculate_position_size(
        self,
        portfolio_value: Decimal,
        current_price: Decimal,
        existing_position: Position | None = None,
    ) -> int:
        """
        Calculate the number of shares to buy.

        Uses the smaller of:
        - max_position_size_pct of portfolio
        - max_position_value (if set)

        Args:
            portfolio_value: Total portfolio value
            current_price: Current share price
            existing_position: Existing position (to avoid over-buying)

        Returns:
            Number of shares to buy
        """
        if current_price <= 0:
            return 0

        # Calculate max position value based on percentage
        max_value_pct = portfolio_value * Decimal(
            str(self.config.max_position_size_pct)
        )

        # Apply absolute max if set
        if self.config.max_position_value is not None:
            max_value = min(max_value_pct, Decimal(str(self.config.max_position_value)))
        else:
            max_value = max_value_pct

        # Calculate shares
        shares = int(max_value / current_price)

        # Reduce by existing position if any
        if existing_position is not None:
            existing_value = current_price * existing_position.quantity
            remaining_allocation = max_value - existing_value
            if remaining_allocation <= 0:
                return 0
            shares = int(remaining_allocation / current_price)

        return max(0, shares)

    def record_trade(self, pnl: Decimal = Decimal(0)) -> None:
        """Record a trade for daily limits."""
        self._daily_trades += 1
        self._daily_pnl += pnl

    def reset_daily_counters(self, starting_equity: Decimal | None = None) -> None:
        """Reset daily trade counters (call at start of each day)."""
        self._daily_trades = 0
        self._daily_pnl = Decimal(0)
        if starting_equity is not None:
            self._starting_equity = starting_equity
        logger.debug("Daily risk counters reset")

    @property
    def daily_trades(self) -> int:
        """Number of trades today."""
        return self._daily_trades

    @property
    def daily_pnl(self) -> Decimal:
        """P&L for today."""
        return self._daily_pnl
