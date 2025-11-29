"""Live trading engine for paper and real trading."""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from enum import Enum
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from loguru import logger

from trader.core.models import Order, OrderSide, OrderType, Signal, SignalAction

if TYPE_CHECKING:
    from trader.broker.base import BaseBroker
    from trader.data.fetcher import BaseDataFetcher
    from trader.risk.manager import RiskManager
    from trader.strategies.base import BaseStrategy


class EngineState(str, Enum):
    """Trading engine states."""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


class TradingMode(str, Enum):
    """Trading mode for position management."""

    DAY = "day"  # Close all positions at end of day
    SWING = "swing"  # Hold positions overnight


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


class LiveTradingEngine:
    """
    Live trading engine for executing strategies in real-time.

    This engine connects to a broker, fetches market data, and
    executes trading signals from strategies. It supports both
    paper and live trading modes.

    Features:
    - Multiple symbol support
    - Risk management integration
    - Market hours awareness
    - Day trading mode (close positions EOD)
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
    ) -> None:
        """Initialize the trading engine.

        Args:
            broker: Broker for order execution
            data_fetcher: Data source for market data
            strategy: Trading strategy to execute
            risk_manager: Risk manager for position sizing
            config: Engine configuration
        """
        self.broker = broker
        self.data_fetcher = data_fetcher
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.config = config or EngineConfig()

        self.state = EngineState.STOPPED
        self._stop_event = asyncio.Event()
        self._tz = ZoneInfo(self.config.timezone)

    async def run(self) -> None:
        """
        Main trading loop.

        Connects to broker, then continuously:
        1. Check if market is open
        2. Fetch latest data
        3. Generate signals
        4. Execute approved signals
        5. Sleep until next check
        """
        self.state = EngineState.STARTING
        logger.info(
            f"Starting live trading engine: {self.strategy.name} "
            f"on {self.config.symbols}"
        )

        try:
            await self.broker.connect()
            self.state = EngineState.RUNNING

            # Log account info
            account_value = await self.broker.get_account_value()
            logger.info(
                f"Connected to {self.broker.name} "
                f"({'paper' if self.broker.is_paper else 'LIVE'}). "
                f"Account value: ${account_value:,.2f}"
            )

            # Reset daily counters
            self.risk_manager.reset_daily_counters(account_value)

            while not self._stop_event.is_set():
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
            raise
        finally:
            await self._shutdown()

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
        """Execute a trading signal."""
        side = OrderSide.BUY if signal.action == SignalAction.BUY else OrderSide.SELL

        order = Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.MARKET,
            signal=signal,
        )

        logger.info(
            f"Executing {side.value} {quantity} {symbol}: {signal.reason}"
        )

        result = await self.broker.submit_order(order)
        self.risk_manager.record_trade()

        logger.info(
            f"Order {result.status.value}: {result.filled_quantity} {symbol} "
            f"@ ${result.filled_avg_price}"
        )

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

        await self.broker.disconnect()
        self.state = EngineState.STOPPED
        logger.info("Trading engine stopped")
