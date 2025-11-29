"""Backtesting engine for evaluating trading strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

from trader.core.models import (
    OrderSide,
    Position,
    SignalAction,
)

if TYPE_CHECKING:
    from trader.strategies.base import BaseStrategy


@dataclass
class Trade:
    """Record of a completed trade."""

    symbol: str
    entry_time: datetime
    exit_time: datetime
    side: OrderSide
    quantity: int
    entry_price: Decimal
    exit_price: Decimal
    pnl: Decimal
    pnl_pct: float
    reason_entry: str = ""
    reason_exit: str = ""

    @property
    def is_profitable(self) -> bool:
        """Check if trade was profitable."""
        return self.pnl > 0


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    strategy_name: str
    symbol: str
    start_date: datetime
    end_date: datetime
    initial_capital: Decimal
    final_capital: Decimal
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.Series | None = None

    @property
    def total_return(self) -> float:
        """Total return as a percentage."""
        return float((self.final_capital - self.initial_capital) / self.initial_capital)

    @property
    def total_return_pct(self) -> float:
        """Total return as percentage (0-100)."""
        return self.total_return * 100

    @property
    def num_trades(self) -> int:
        """Total number of completed trades."""
        return len(self.trades)

    @property
    def winning_trades(self) -> int:
        """Number of winning trades."""
        return sum(1 for t in self.trades if t.is_profitable)

    @property
    def losing_trades(self) -> int:
        """Number of losing trades."""
        return sum(1 for t in self.trades if not t.is_profitable)

    @property
    def win_rate(self) -> float:
        """Win rate (0-1)."""
        if self.num_trades == 0:
            return 0.0
        return self.winning_trades / self.num_trades

    @property
    def profit_factor(self) -> float:
        """Gross profit / gross loss."""
        gross_profit = sum(float(t.pnl) for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(float(t.pnl) for t in self.trades if t.pnl < 0))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown as percentage (0-1)."""
        if self.equity_curve is None or len(self.equity_curve) == 0:
            return 0.0

        peak = self.equity_curve.expanding().max()
        drawdown = (self.equity_curve - peak) / peak
        return abs(float(drawdown.min()))

    @property
    def sharpe_ratio(self) -> float:
        """Annualized Sharpe ratio (assumes 252 trading days)."""
        if self.equity_curve is None or len(self.equity_curve) < 2:
            return 0.0

        returns = self.equity_curve.pct_change().dropna()
        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        return float(returns.mean() / returns.std() * (252 ** 0.5))

    def summary(self) -> dict:
        """Get summary statistics as dictionary."""
        return {
            "strategy": self.strategy_name,
            "symbol": self.symbol,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "initial_capital": float(self.initial_capital),
            "final_capital": float(self.final_capital),
            "total_return_pct": round(self.total_return_pct, 2),
            "num_trades": self.num_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(self.win_rate * 100, 2),
            "profit_factor": round(self.profit_factor, 2),
            "max_drawdown_pct": round(self.max_drawdown * 100, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
        }

    def print_summary(self) -> None:
        """Print formatted summary to console."""
        summary = self.summary()
        print("\n" + "=" * 50)
        print(f"Backtest Results: {summary['strategy']}")
        print("=" * 50)
        print(f"Symbol:           {summary['symbol']}")
        print(f"Period:           {summary['start_date'][:10]} to {summary['end_date'][:10]}")
        print("-" * 50)
        print(f"Initial Capital:  ${summary['initial_capital']:,.2f}")
        print(f"Final Capital:    ${summary['final_capital']:,.2f}")
        print(f"Total Return:     {summary['total_return_pct']:+.2f}%")
        print("-" * 50)
        print(f"Total Trades:     {summary['num_trades']}")
        print(f"Winning Trades:   {summary['winning_trades']}")
        print(f"Losing Trades:    {summary['losing_trades']}")
        print(f"Win Rate:         {summary['win_rate']:.1f}%")
        print(f"Profit Factor:    {summary['profit_factor']:.2f}")
        print("-" * 50)
        print(f"Max Drawdown:     {summary['max_drawdown_pct']:.2f}%")
        print(f"Sharpe Ratio:     {summary['sharpe_ratio']:.2f}")
        print("=" * 50 + "\n")


class BacktestEngine:
    """
    Simple backtesting engine for evaluating trading strategies.

    This engine runs a strategy against historical data and tracks
    performance metrics. It simulates order execution at the close
    price of each bar.

    Features:
    - Single symbol backtesting
    - Market orders only (executed at close)
    - Commission support
    - Full or fractional position sizing

    Example:
        engine = BacktestEngine(initial_capital=100000)
        result = await engine.run(
            strategy=SMACrossover(fast_period=10, slow_period=50),
            data=historical_data,
            symbol="AAPL",
        )
        result.print_summary()
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission: float = 0.0,
        position_size_pct: float = 1.0,
    ) -> None:
        """Initialize backtest engine.

        Args:
            initial_capital: Starting capital in dollars
            commission: Commission per trade in dollars
            position_size_pct: Fraction of capital to use per trade (0-1)
        """
        self.initial_capital = Decimal(str(initial_capital))
        self.commission = Decimal(str(commission))
        self.position_size_pct = position_size_pct

    async def run(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        symbol: str,
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            strategy: Trading strategy to test
            data: DataFrame with OHLCV columns (open, high, low, close, volume)
                  Index should be datetime
            symbol: Symbol being traded

        Returns:
            BacktestResult with performance metrics
        """
        if len(data) == 0:
            raise ValueError("Data cannot be empty")

        logger.info(
            f"Running backtest: {strategy.name} on {symbol} "
            f"({len(data)} bars)"
        )

        # Calculate indicators once
        data = strategy.calculate_indicators(data)

        # Initialize state
        capital = self.initial_capital
        position: Position | None = None
        trades: list[Trade] = []
        equity_curve: list[tuple[datetime, float]] = []

        # Entry tracking for trade records
        entry_time: datetime | None = None
        entry_price: Decimal | None = None
        entry_reason: str = ""

        # Iterate through each bar
        for i in range(len(data)):
            current_bar = data.iloc[i]
            timestamp = data.index[i]
            close_price = Decimal(str(current_bar["close"]))

            # Update position value
            if position is not None:
                position.update_price(close_price)

            # Calculate current equity
            if position is not None:
                equity = capital + (position.market_value or Decimal(0))
            else:
                equity = capital

            equity_curve.append((timestamp, float(equity)))

            # Skip if not enough data for strategy
            if i < strategy.min_bars_required:
                continue

            # Get data up to current bar (no look-ahead)
            historical_data = data.iloc[: i + 1]

            # Generate signal
            signal = strategy.generate_signal(
                data=historical_data,
                symbol=symbol,
                position=position,
            )

            # Process signal
            if signal.action == SignalAction.BUY and position is None:
                # Calculate position size
                available_capital = capital * Decimal(str(self.position_size_pct))
                quantity = int(available_capital / close_price)

                if quantity > 0:
                    # Open long position
                    cost = close_price * quantity + self.commission
                    capital -= cost

                    position = Position(
                        symbol=symbol,
                        quantity=quantity,
                        avg_entry_price=close_price,
                        current_price=close_price,
                    )
                    position.update_price(close_price)

                    entry_time = timestamp
                    entry_price = close_price
                    entry_reason = signal.reason

                    logger.debug(
                        f"BUY {quantity} {symbol} @ {close_price} "
                        f"(capital: {capital})"
                    )

            elif signal.action == SignalAction.SELL and position is not None:
                # Close position
                proceeds = close_price * position.quantity - self.commission
                capital += proceeds

                # These should always be set when we have a position
                assert entry_price is not None
                assert entry_time is not None

                # Record trade
                pnl = proceeds - (entry_price * position.quantity)
                pnl_pct = float(pnl / (entry_price * position.quantity))

                trade = Trade(
                    symbol=symbol,
                    entry_time=entry_time,
                    exit_time=timestamp,
                    side=OrderSide.BUY,
                    quantity=position.quantity,
                    entry_price=entry_price,
                    exit_price=close_price,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    reason_entry=entry_reason,
                    reason_exit=signal.reason,
                )
                trades.append(trade)

                logger.debug(
                    f"SELL {position.quantity} {symbol} @ {close_price} "
                    f"(P&L: {pnl:.2f}, capital: {capital})"
                )

                position = None
                entry_time = None
                entry_price = None
                entry_reason = ""

        # Close any remaining position at end
        if position is not None:
            final_price = Decimal(str(data.iloc[-1]["close"]))
            proceeds = final_price * position.quantity - self.commission
            capital += proceeds

            # These should always be set when we have a position
            assert entry_price is not None
            assert entry_time is not None

            pnl = proceeds - (entry_price * position.quantity)
            pnl_pct = float(pnl / (entry_price * position.quantity))

            trade = Trade(
                symbol=symbol,
                entry_time=entry_time,
                exit_time=data.index[-1],
                side=OrderSide.BUY,
                quantity=position.quantity,
                entry_price=entry_price,
                exit_price=final_price,
                pnl=pnl,
                pnl_pct=pnl_pct,
                reason_entry=entry_reason,
                reason_exit="End of backtest",
            )
            trades.append(trade)

        # Build equity curve series
        equity_series = pd.Series(
            [e[1] for e in equity_curve],
            index=[e[0] for e in equity_curve],
            name="equity",
        )

        result = BacktestResult(
            strategy_name=strategy.name,
            symbol=symbol,
            start_date=data.index[0],
            end_date=data.index[-1],
            initial_capital=self.initial_capital,
            final_capital=capital,
            trades=trades,
            equity_curve=equity_series,
        )

        logger.info(
            f"Backtest complete: {result.num_trades} trades, "
            f"return: {result.total_return_pct:.2f}%"
        )

        return result
