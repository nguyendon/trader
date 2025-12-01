"""Live trading engine for paper and real trading."""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING, Callable
from zoneinfo import ZoneInfo

from loguru import logger

from trader.core.models import Order, OrderClass, OrderSide, OrderType, Signal, SignalAction

if TYPE_CHECKING:
    from trader.broker.base import BaseBroker
    from trader.data.fetcher import BaseDataFetcher
    from trader.notifications.discord import DiscordNotifier
    from trader.risk.manager import RiskManager
    from trader.strategies.base import BaseStrategy


class EngineState(str, Enum):
    """Trading engine states."""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    PAUSED = "paused"
    ERROR = "error"


class TradingMode(str, Enum):
    """Trading mode for position management."""

    DAY = "day"  # Close all positions at end of day
    SWING = "swing"  # Hold positions overnight


@dataclass
class SafetyLimits:
    """Hard safety limits that cannot be overridden."""

    max_position_value: float = 10000.0  # Max $ per position
    max_portfolio_value: float = 50000.0  # Max total $ in positions
    max_loss_per_day: float = 500.0  # Stop trading if daily loss exceeds
    max_trades_per_day: int = 20  # Max number of trades per day
    max_loss_per_trade: float = 200.0  # Max loss on single trade
    require_confirmation: bool = False  # Require human confirmation

    # Stop loss / take profit settings
    stop_loss_pct: float | None = None  # Auto stop loss as % below entry (e.g., 0.05 = 5%)
    take_profit_pct: float | None = None  # Auto take profit as % above entry (e.g., 0.10 = 10%)
    use_bracket_orders: bool = True  # Use bracket orders when stop/profit set


@dataclass
class EngineConfig:
    """Configuration for the trading engine."""

    symbols: list[str] = field(default_factory=lambda: ["AAPL"])
    trading_mode: TradingMode = TradingMode.SWING
    check_interval_seconds: int = 60  # How often to check for signals
    market_open: time = time(9, 30)
    market_close: time = time(16, 0)
    timezone: str = "America/New_York"
    close_positions_before_close: int = 15  # Minutes before close to flatten
    safety: SafetyLimits = field(default_factory=SafetyLimits)


class LiveTradingEngine:
    """
    Live trading engine for executing strategies in real-time.

    This engine connects to a broker, fetches market data, and
    executes trading signals from strategies. It supports both
    paper and live trading modes.

    Features:
    - Multiple symbol support
    - Risk management integration
    - Hard safety limits (circuit breakers)
    - Market hours awareness
    - Day trading mode (close positions EOD)
    - Optional human confirmation
    - Graceful shutdown

    Example:
        engine = LiveTradingEngine(
            broker=AlpacaBroker(...),
            data_fetcher=AlpacaDataFetcher(...),
            strategy=SMACrossover(),
            risk_manager=RiskManager(),
        )

        await engine.run()
    """

    def __init__(
        self,
        broker: BaseBroker,
        data_fetcher: BaseDataFetcher,
        strategy: BaseStrategy,
        risk_manager: RiskManager,
        config: EngineConfig | None = None,
        on_signal: Callable[[Signal, str, int], bool] | None = None,
        notifier: DiscordNotifier | None = None,
    ) -> None:
        """Initialize the trading engine.

        Args:
            broker: Broker for order execution
            data_fetcher: Data source for market data
            strategy: Trading strategy to execute
            risk_manager: Risk manager for position sizing
            config: Engine configuration
            on_signal: Optional callback for signal confirmation.
                       Called with (signal, symbol, quantity). Return True to execute.
            notifier: Optional Discord notifier for trade alerts
        """
        self.broker = broker
        self.data_fetcher = data_fetcher
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.config = config or EngineConfig()
        self.on_signal = on_signal
        self.notifier = notifier

        self.state = EngineState.STOPPED
        self._stop_event = asyncio.Event()
        self._tz = ZoneInfo(self.config.timezone)

        # Daily tracking
        self._daily_trades = 0
        self._daily_pnl = Decimal(0)
        self._starting_value = Decimal(0)
        self._last_reset_date: datetime | None = None

    async def run(self) -> None:
        """
        Main trading loop.

        Connects to broker, then continuously:
        1. Check if market is open
        2. Check safety limits
        3. Fetch latest data
        4. Generate signals
        5. Execute approved signals (with optional confirmation)
        6. Sleep until next check
        """
        self.state = EngineState.STARTING
        safety = self.config.safety

        logger.info(
            f"Starting live trading engine: {self.strategy.name} "
            f"on {self.config.symbols}"
        )
        logger.info(
            f"Safety limits: max_position=${safety.max_position_value:,.0f}, "
            f"max_daily_loss=${safety.max_loss_per_day:,.0f}, "
            f"max_trades={safety.max_trades_per_day}"
        )

        try:
            await self.broker.connect()
            self.state = EngineState.RUNNING

            # Log account info
            account_value = await self.broker.get_account_value()
            self._starting_value = account_value
            logger.info(
                f"Connected to {self.broker.name} "
                f"({'paper' if self.broker.is_paper else 'LIVE'}). "
                f"Account value: ${account_value:,.2f}"
            )

            # Send startup notification
            if self.notifier:
                await self.notifier.notify_engine_started(
                    symbols=self.config.symbols,
                    strategy=self.strategy.name,
                    account_value=float(account_value),
                    is_paper=self.broker.is_paper,
                )

            # Reset daily counters
            self.risk_manager.reset_daily_counters(float(account_value))
            self._reset_daily_counters()

            while not self._stop_event.is_set():
                # Check for daily reset
                self._check_daily_reset()

                # Check circuit breakers
                breaker_reason = self._check_circuit_breakers()
                if breaker_reason:
                    logger.warning(f"Circuit breaker triggered - {breaker_reason}")
                    self.state = EngineState.PAUSED

                    # Send notification
                    if self.notifier:
                        await self.notifier.notify_circuit_breaker(
                            reason=breaker_reason,
                            daily_trades=self._daily_trades,
                            daily_pnl=float(self._daily_pnl),
                        )

                    # Still run loop but don't trade
                    await asyncio.sleep(self.config.check_interval_seconds)
                    continue

                self.state = EngineState.RUNNING
                await self._trading_iteration()

                # Wait for next check or stop signal
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=self.config.check_interval_seconds,
                    )

        except Exception as e:
            self.state = EngineState.ERROR
            logger.exception(f"Engine error: {e}")

            # Send error notification
            if self.notifier:
                await self.notifier.notify_error(
                    error=str(e),
                    context=f"Engine crashed after {self._daily_trades} trades",
                )

            raise
        finally:
            await self._shutdown()

    def _reset_daily_counters(self) -> None:
        """Reset daily tracking counters."""
        self._daily_trades = 0
        self._daily_pnl = Decimal(0)
        self._last_reset_date = datetime.now(self._tz).date()
        logger.debug("Daily counters reset")

    def _check_daily_reset(self) -> None:
        """Check if we need to reset daily counters (new trading day)."""
        today = datetime.now(self._tz).date()
        if self._last_reset_date != today:
            self._reset_daily_counters()

    def _check_circuit_breakers(self) -> str | None:
        """Check if any circuit breakers are triggered.

        Returns:
            Reason string if triggered, None otherwise
        """
        safety = self.config.safety

        # Check max trades per day
        if self._daily_trades >= safety.max_trades_per_day:
            return f"Max trades reached: {self._daily_trades}/{safety.max_trades_per_day}"

        # Check daily loss limit
        if float(self._daily_pnl) <= -safety.max_loss_per_day:
            return f"Daily loss limit reached: ${self._daily_pnl:,.2f}"

        return None

    async def _check_position_limits(self, symbol: str, quantity: int, price: Decimal) -> tuple[bool, str]:
        """Check if a trade would exceed position limits. Returns (ok, reason)."""
        safety = self.config.safety
        trade_value = float(price * quantity)

        # Check single position limit
        if trade_value > safety.max_position_value:
            return False, f"Trade value ${trade_value:,.0f} exceeds max ${safety.max_position_value:,.0f}"

        # Check total portfolio limit
        positions = await self.broker.get_positions()
        total_value = sum(float(p.market_value or 0) for p in positions)

        if total_value + trade_value > safety.max_portfolio_value:
            return False, f"Would exceed portfolio limit: ${total_value + trade_value:,.0f} > ${safety.max_portfolio_value:,.0f}"

        return True, ""

    async def stop(self) -> None:
        """Signal the engine to stop gracefully."""
        logger.info("Stop signal received")
        self.state = EngineState.STOPPING
        self._stop_event.set()

    async def _trading_iteration(self) -> None:
        """Execute one trading iteration."""
        now = datetime.now(self._tz)

        # Check market hours
        if not self._is_market_open(now):
            logger.debug("Market closed, skipping iteration")
            return

        # Check if we need to close positions (day trading mode)
        if self._should_close_positions(now):
            await self._close_all_positions("End of day")
            return

        # Check each symbol
        for symbol in self.config.symbols:
            try:
                await self._process_symbol(symbol)
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")

    async def _process_symbol(self, symbol: str) -> None:
        """Process a single symbol - fetch data, generate signal, execute."""
        # Fetch recent data
        end = datetime.now()
        start = end - timedelta(days=100)  # Get enough for indicators

        from trader.core.models import TimeFrame

        data = await self.data_fetcher.fetch_bars_df(
            symbol=symbol,
            timeframe=TimeFrame.DAY,
            start=start,
            end=end,
        )

        if len(data) < self.strategy.min_bars_required:
            logger.debug(f"{symbol}: Not enough data ({len(data)} bars)")
            return

        # Calculate indicators
        data = self.strategy.calculate_indicators(data)

        # Get current position
        position = await self.broker.get_position(symbol)

        # Generate signal
        signal = self.strategy.generate_signal(
            data=data,
            symbol=symbol,
            position=position,
        )

        if signal.action == SignalAction.HOLD:
            logger.debug(f"{symbol}: {signal.reason}")
            return

        # Check against risk rules
        portfolio_value = await self.broker.get_account_value()
        current_price = await self.broker.get_latest_price(symbol)
        positions = await self.broker.get_positions()

        risk_result = self.risk_manager.check_signal(
            signal=signal,
            portfolio_value=portfolio_value,
            current_price=current_price,
            open_positions=len(positions),
            existing_position=position,
            daily_pnl=self.risk_manager.daily_pnl,
        )

        if not risk_result.approved:
            logger.info(f"{symbol}: Signal rejected - {risk_result.reason}")
            return

        if risk_result.adjusted_quantity is None:
            logger.warning(f"{symbol}: No quantity from risk manager")
            return

        # Execute the trade
        await self._execute_signal(
            symbol=symbol,
            signal=signal,
            quantity=risk_result.adjusted_quantity,
        )

    async def _execute_signal(
        self,
        symbol: str,
        signal: Signal,
        quantity: int,
    ) -> None:
        """Execute a trading signal with safety checks and optional confirmation."""
        side = OrderSide.BUY if signal.action == SignalAction.BUY else OrderSide.SELL

        # Get current price for safety checks
        current_price = await self.broker.get_latest_price(symbol)

        # Check position limits for buys
        if signal.action == SignalAction.BUY:
            ok, reason = await self._check_position_limits(symbol, quantity, current_price)
            if not ok:
                logger.warning(f"{symbol}: Trade blocked by safety limits - {reason}")
                return

        # Optional human confirmation
        if self.config.safety.require_confirmation or self.on_signal:
            if self.on_signal:
                approved = self.on_signal(signal, symbol, quantity)
                if not approved:
                    logger.info(f"{symbol}: Trade rejected by confirmation callback")
                    return

        # Calculate stop loss and take profit prices
        safety = self.config.safety
        stop_loss_price = None
        take_profit_price = None

        # Use signal's stop/profit if provided, otherwise use safety defaults
        if signal.stop_loss:
            stop_loss_price = signal.stop_loss
        elif safety.stop_loss_pct and signal.action == SignalAction.BUY:
            stop_loss_price = current_price * Decimal(1 - safety.stop_loss_pct)

        if signal.take_profit:
            take_profit_price = signal.take_profit
        elif safety.take_profit_pct and signal.action == SignalAction.BUY:
            take_profit_price = current_price * Decimal(1 + safety.take_profit_pct)

        # Determine order class
        use_bracket = (
            safety.use_bracket_orders
            and signal.action == SignalAction.BUY  # Only for entry orders
            and (stop_loss_price or take_profit_price)
        )

        order = Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.MARKET,
            signal=signal,
            order_class=OrderClass.BRACKET if use_bracket else OrderClass.SIMPLE,
            stop_loss_price=stop_loss_price if use_bracket else None,
            take_profit_price=take_profit_price if use_bracket else None,
        )

        # Build log message
        bracket_info = ""
        if use_bracket:
            parts = []
            if stop_loss_price:
                parts.append(f"SL@${float(stop_loss_price):.2f}")
            if take_profit_price:
                parts.append(f"TP@${float(take_profit_price):.2f}")
            bracket_info = f" [{' '.join(parts)}]"

        logger.info(f"Executing {side.value} {quantity} {symbol}{bracket_info}: {signal.reason}")

        result = await self.broker.submit_order(order)
        self.risk_manager.record_trade()

        # Update daily tracking
        self._daily_trades += 1
        # TODO: Track P&L properly with entry price for accurate daily loss calculation

        logger.info(
            f"Order {result.status.value}: {result.filled_quantity} {symbol} "
            f"@ ${result.filled_avg_price} (trade #{self._daily_trades} today)"
        )

        # Send trade notification
        if self.notifier:
            await self.notifier.notify_order_filled(result)

    async def _close_all_positions(self, reason: str) -> None:
        """Close all open positions."""
        positions = await self.broker.get_positions()

        if not positions:
            return

        logger.info(f"Closing {len(positions)} positions: {reason}")

        for position in positions:
            order = Order(
                symbol=position.symbol,
                side=OrderSide.SELL,
                quantity=position.quantity,
                order_type=OrderType.MARKET,
            )

            try:
                await self.broker.submit_order(order)
                logger.info(f"Closed position: {position.symbol}")
            except Exception as e:
                logger.error(f"Failed to close {position.symbol}: {e}")

    def _is_market_open(self, now: datetime) -> bool:
        """Check if market is currently open."""
        # Simple check - weekday and within market hours
        if now.weekday() >= 5:  # Saturday or Sunday
            return False

        current_time = now.time()
        return self.config.market_open <= current_time < self.config.market_close

    def _should_close_positions(self, now: datetime) -> bool:
        """Check if we should close positions (day trading mode)."""
        if self.config.trading_mode != TradingMode.DAY:
            return False

        close_time = datetime.combine(
            now.date(),
            self.config.market_close,
            tzinfo=self._tz,
        )
        close_buffer = close_time - timedelta(
            minutes=self.config.close_positions_before_close
        )

        return now >= close_buffer

    async def _shutdown(self) -> None:
        """Clean shutdown of the engine."""
        self.state = EngineState.STOPPING
        logger.info("Shutting down trading engine")

        # Close positions if in day trading mode
        if self.config.trading_mode == TradingMode.DAY:
            await self._close_all_positions("Engine shutdown")

        # Send shutdown notification
        if self.notifier:
            await self.notifier.notify_engine_stopped(
                reason="Manual shutdown",
                daily_trades=self._daily_trades,
                daily_pnl=float(self._daily_pnl),
            )
            await self.notifier.close()

        await self.broker.disconnect()
        self.state = EngineState.STOPPED
        logger.info("Trading engine stopped")
