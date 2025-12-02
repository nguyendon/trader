"""Transaction cost modeling for realistic backtesting."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass
class CostModel:
    """
    Comprehensive transaction cost model.

    Models realistic trading costs including:
    - Commission: Fixed fee per trade
    - Spread: Bid-ask spread cost
    - Slippage: Price movement between signal and execution
    - Market impact: Price movement caused by your order

    Example:
        # Realistic retail costs
        costs = CostModel(
            commission_per_trade=1.00,
            spread_pct=0.05,  # 5 bps spread
            slippage_pct=0.02,  # 2 bps slippage
        )

        # Calculate total cost for a trade
        entry_cost = costs.calculate_entry_cost(price=150.0, quantity=100)
        exit_cost = costs.calculate_exit_cost(price=155.0, quantity=100)
    """

    # Fixed costs
    commission_per_trade: float = 0.0  # Fixed $ per trade
    commission_per_share: float = 0.0  # $ per share

    # Proportional costs (as decimal, e.g., 0.0005 = 5 bps)
    spread_pct: float = 0.0  # Bid-ask spread (pay half on each side)
    slippage_pct: float = 0.0  # Execution slippage

    # Market impact (for larger orders)
    market_impact_pct: float = 0.0  # Base market impact
    market_impact_power: float = 0.5  # Impact scales with sqrt(size) by default

    # Volume-based impact
    avg_daily_volume: int | None = None  # If set, enables volume-based impact
    volume_impact_coefficient: float = 0.1  # Impact per % of ADV

    def calculate_entry_cost(
        self,
        price: float | Decimal,
        quantity: int,
        daily_volume: int | None = None,
    ) -> Decimal:
        """
        Calculate total cost of entering a position.

        Args:
            price: Execution price
            quantity: Number of shares
            daily_volume: Optional daily volume for impact calculation

        Returns:
            Total cost in dollars (always positive)
        """
        price = Decimal(str(price))
        trade_value = price * quantity

        # Fixed costs
        cost = Decimal(str(self.commission_per_trade))
        cost += Decimal(str(self.commission_per_share)) * quantity

        # Spread cost (pay half the spread on entry)
        spread_cost = trade_value * Decimal(str(self.spread_pct / 2))
        cost += spread_cost

        # Slippage (assume adverse movement)
        slippage_cost = trade_value * Decimal(str(self.slippage_pct))
        cost += slippage_cost

        # Market impact
        impact_cost = self._calculate_market_impact(
            trade_value, quantity, daily_volume
        )
        cost += impact_cost

        return cost

    def calculate_exit_cost(
        self,
        price: float | Decimal,
        quantity: int,
        daily_volume: int | None = None,
    ) -> Decimal:
        """
        Calculate total cost of exiting a position.

        Args:
            price: Execution price
            quantity: Number of shares
            daily_volume: Optional daily volume for impact calculation

        Returns:
            Total cost in dollars (always positive)
        """
        # Exit costs are similar to entry costs
        return self.calculate_entry_cost(price, quantity, daily_volume)

    def calculate_round_trip_cost(
        self,
        entry_price: float | Decimal,
        exit_price: float | Decimal,
        quantity: int,
        daily_volume: int | None = None,
    ) -> Decimal:
        """
        Calculate total round-trip cost (entry + exit).

        Args:
            entry_price: Entry execution price
            exit_price: Exit execution price
            quantity: Number of shares
            daily_volume: Optional daily volume for impact calculation

        Returns:
            Total round-trip cost in dollars
        """
        entry_cost = self.calculate_entry_cost(entry_price, quantity, daily_volume)
        exit_cost = self.calculate_exit_cost(exit_price, quantity, daily_volume)
        return entry_cost + exit_cost

    def get_effective_entry_price(
        self,
        price: float | Decimal,
        quantity: int,
        side: str = "buy",
        daily_volume: int | None = None,
    ) -> Decimal:
        """
        Get effective entry price after all costs.

        For buys, the effective price is higher than market price.
        For sells (shorting), the effective price is lower.

        Args:
            price: Market price
            quantity: Number of shares
            side: "buy" or "sell"
            daily_volume: Optional daily volume for impact calculation

        Returns:
            Effective price per share
        """
        price = Decimal(str(price))
        cost = self.calculate_entry_cost(price, quantity, daily_volume)
        cost_per_share = cost / quantity

        if side == "buy":
            return price + cost_per_share
        else:
            return price - cost_per_share

    def get_effective_exit_price(
        self,
        price: float | Decimal,
        quantity: int,
        side: str = "sell",
        daily_volume: int | None = None,
    ) -> Decimal:
        """
        Get effective exit price after all costs.

        For sells (closing long), the effective price is lower than market.
        For buys (closing short), the effective price is higher.

        Args:
            price: Market price
            quantity: Number of shares
            side: "sell" or "buy" (for covering short)
            daily_volume: Optional daily volume for impact calculation

        Returns:
            Effective price per share
        """
        price = Decimal(str(price))
        cost = self.calculate_exit_cost(price, quantity, daily_volume)
        cost_per_share = cost / quantity

        if side == "sell":
            return price - cost_per_share
        else:
            return price + cost_per_share

    def _calculate_market_impact(
        self,
        trade_value: Decimal,
        quantity: int,
        daily_volume: int | None = None,
    ) -> Decimal:
        """Calculate market impact cost."""
        if self.market_impact_pct == 0 and self.volume_impact_coefficient == 0:
            return Decimal("0")

        impact = Decimal("0")

        # Base market impact (scales with trade size)
        if self.market_impact_pct > 0:
            # Impact scales with sqrt of trade value by default
            base_value = Decimal("10000")  # Reference trade size
            scale = (trade_value / base_value) ** Decimal(str(self.market_impact_power))
            impact += trade_value * Decimal(str(self.market_impact_pct)) * scale

        # Volume-based impact
        if self.volume_impact_coefficient > 0 and daily_volume:
            adv = self.avg_daily_volume or daily_volume
            participation_rate = quantity / adv
            volume_impact = (
                trade_value
                * Decimal(str(self.volume_impact_coefficient))
                * Decimal(str(participation_rate))
            )
            impact += volume_impact

        return impact

    def summary(self) -> dict:
        """Get cost model parameters as dictionary."""
        return {
            "commission_per_trade": self.commission_per_trade,
            "commission_per_share": self.commission_per_share,
            "spread_pct_bps": self.spread_pct * 10000,
            "slippage_pct_bps": self.slippage_pct * 10000,
            "market_impact_pct_bps": self.market_impact_pct * 10000,
        }

    @classmethod
    def zero_cost(cls) -> CostModel:
        """Create a zero-cost model (for comparison)."""
        return cls()

    @classmethod
    def retail_investor(cls) -> CostModel:
        """
        Typical retail investor costs.

        Assumes:
        - Commission-free broker (Robinhood, etc.)
        - ~5 bps spread on liquid stocks
        - ~2 bps slippage
        """
        return cls(
            commission_per_trade=0.0,
            spread_pct=0.0005,  # 5 bps
            slippage_pct=0.0002,  # 2 bps
        )

    @classmethod
    def active_trader(cls) -> CostModel:
        """
        Active trader costs.

        Assumes:
        - Small commission
        - Tighter spreads (better execution)
        - Some slippage
        """
        return cls(
            commission_per_trade=1.00,
            spread_pct=0.0003,  # 3 bps
            slippage_pct=0.0002,  # 2 bps
            market_impact_pct=0.0001,  # 1 bp
        )

    @classmethod
    def institutional(cls, avg_daily_volume: int = 1_000_000) -> CostModel:
        """
        Institutional investor costs.

        Assumes:
        - Low commission
        - Minimal spread
        - Significant market impact due to size
        """
        return cls(
            commission_per_trade=0.50,
            commission_per_share=0.001,
            spread_pct=0.0002,  # 2 bps
            slippage_pct=0.0001,  # 1 bp
            market_impact_pct=0.0002,  # 2 bps base
            market_impact_power=0.5,
            avg_daily_volume=avg_daily_volume,
            volume_impact_coefficient=0.1,
        )

    @classmethod
    def high_frequency(cls) -> CostModel:
        """
        High-frequency trading costs.

        Assumes:
        - Rebates (negative commission)
        - Zero spread (market maker)
        - Minimal slippage
        """
        return cls(
            commission_per_trade=-0.20,  # Rebate
            spread_pct=0.0,
            slippage_pct=0.00005,  # 0.5 bps
        )


@dataclass
class CostReport:
    """Report of costs incurred during backtesting."""

    total_commission: Decimal = Decimal("0")
    total_spread_cost: Decimal = Decimal("0")
    total_slippage: Decimal = Decimal("0")
    total_market_impact: Decimal = Decimal("0")
    num_trades: int = 0
    gross_pnl: Decimal = Decimal("0")
    net_pnl: Decimal = Decimal("0")

    @property
    def total_costs(self) -> Decimal:
        """Total transaction costs."""
        return (
            self.total_commission
            + self.total_spread_cost
            + self.total_slippage
            + self.total_market_impact
        )

    @property
    def cost_per_trade(self) -> Decimal:
        """Average cost per trade."""
        if self.num_trades == 0:
            return Decimal("0")
        return self.total_costs / self.num_trades

    @property
    def cost_as_pct_of_gross(self) -> float:
        """Costs as percentage of gross P&L."""
        if self.gross_pnl == 0:
            return 0.0
        return float(self.total_costs / abs(self.gross_pnl)) * 100

    def summary(self) -> dict:
        """Get cost report as dictionary."""
        return {
            "total_commission": float(self.total_commission),
            "total_spread_cost": float(self.total_spread_cost),
            "total_slippage": float(self.total_slippage),
            "total_market_impact": float(self.total_market_impact),
            "total_costs": float(self.total_costs),
            "num_trades": self.num_trades,
            "cost_per_trade": float(self.cost_per_trade),
            "gross_pnl": float(self.gross_pnl),
            "net_pnl": float(self.net_pnl),
            "cost_as_pct_of_gross": round(self.cost_as_pct_of_gross, 2),
        }
