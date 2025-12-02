"""Live trading dashboard with Rich terminal UI."""

from __future__ import annotations

import asyncio
import contextlib
import sys
import typing
from collections import deque
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from loguru import logger
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from trader.broker.base import BaseBroker
    from trader.core.models import Order, Position


# Sparkline characters (block elements for mini chart)
SPARKLINE_CHARS = "▁▂▃▄▅▆▇█"


def _make_sparkline(values: list[float], width: int = 20) -> str:
    """Create a sparkline string from a list of values.

    Args:
        values: List of numeric values
        width: Maximum width of sparkline

    Returns:
        String representation of sparkline
    """
    if not values:
        return ""

    # Take last N values to fit width
    values = list(values)[-width:]

    if len(values) < 2:
        return SPARKLINE_CHARS[4] * len(values)

    min_val = min(values)
    max_val = max(values)
    val_range = max_val - min_val

    if val_range == 0:
        return SPARKLINE_CHARS[4] * len(values)

    # Map values to character indices
    result = []
    for val in values:
        normalized = (val - min_val) / val_range
        char_idx = int(normalized * (len(SPARKLINE_CHARS) - 1))
        result.append(SPARKLINE_CHARS[char_idx])

    return "".join(result)


def _setup_keyboard_listener() -> tuple[
    asyncio.Queue[str], typing.Callable[[], typing.Coroutine], typing.Callable[[], None]
]:
    """Set up non-blocking keyboard input for Unix systems."""
    import termios
    import tty

    queue: asyncio.Queue[str] = asyncio.Queue()
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    def restore_terminal() -> None:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    async def read_keys() -> None:
        """Read keyboard input in background."""
        loop = asyncio.get_event_loop()
        try:
            tty.setcbreak(fd)
            while True:
                # Use run_in_executor for non-blocking read
                char = await loop.run_in_executor(None, sys.stdin.read, 1)
                if char:
                    await queue.put(char.lower())
        except asyncio.CancelledError:
            pass
        finally:
            restore_terminal()

    return queue, read_keys, restore_terminal


class TradingDashboard:
    """
    Real-time trading dashboard using Rich Live display.

    Shows:
    - Account summary (equity, P&L, buying power)
    - Equity sparkline chart
    - Open positions with live P&L
    - Recent trades
    - Open orders
    - Strategy performance metrics
    - Keyboard shortcuts
    """

    # Maximum history points to keep for sparklines
    MAX_HISTORY = 60

    def __init__(
        self,
        broker: BaseBroker,
        refresh_rate: float = 2.0,
    ) -> None:
        """Initialize dashboard.

        Args:
            broker: Broker instance for fetching data
            refresh_rate: How often to refresh in seconds
        """
        self.broker = broker
        self.refresh_rate = refresh_rate
        self.console = Console()
        self._running = False
        self._start_equity: Decimal | None = None
        self._recent_trades: list[dict] = []
        self._status_message: str = ""

        # History tracking for sparklines
        self._equity_history: deque[float] = deque(maxlen=self.MAX_HISTORY)
        self._pnl_history: deque[float] = deque(maxlen=self.MAX_HISTORY)

        # Strategy performance tracking
        self._strategy_stats: dict[str, dict] = {}

        # Session stats
        self._session_start = datetime.now(UTC)
        self._total_trades = 0
        self._winning_trades = 0

    def add_trade(self, trade: dict) -> None:
        """Add a trade to the recent trades list."""
        self._recent_trades.insert(0, trade)
        # Keep only last 10 trades
        self._recent_trades = self._recent_trades[:10]

        # Update session stats
        self._total_trades += 1
        pnl = trade.get("pnl", 0)
        if pnl > 0:
            self._winning_trades += 1

        # Track strategy performance
        strategy = trade.get("strategy", "unknown")
        if strategy not in self._strategy_stats:
            self._strategy_stats[strategy] = {
                "trades": 0,
                "wins": 0,
                "total_pnl": 0.0,
            }
        self._strategy_stats[strategy]["trades"] += 1
        if pnl > 0:
            self._strategy_stats[strategy]["wins"] += 1
        self._strategy_stats[strategy]["total_pnl"] += pnl

    def record_equity(self, equity: Decimal) -> None:
        """Record equity value for sparkline history."""
        self._equity_history.append(float(equity))
        if self._start_equity:
            pnl = float(equity - self._start_equity)
            self._pnl_history.append(pnl)

    async def run(self) -> None:
        """Run the dashboard with live updates."""
        self._running = True
        self._status_message = ""
        key_task = None
        restore_fn = None

        # Get starting equity for daily P&L calculation
        self._start_equity = await self.broker.get_account_value()

        # Set up keyboard listener
        try:
            key_queue, read_keys_coro, restore_fn = _setup_keyboard_listener()
            key_task = asyncio.create_task(read_keys_coro())
        except Exception:
            # Keyboard input not available (e.g., not a terminal)
            key_queue = None

        try:
            with Live(
                self._generate_layout(),
                console=self.console,
                refresh_per_second=1 / self.refresh_rate,
                screen=True,
            ) as live:
                while self._running:
                    try:
                        # Check for keyboard input
                        if key_queue:
                            try:
                                key = key_queue.get_nowait()
                                await self._handle_key(key)
                            except asyncio.QueueEmpty:
                                pass

                        live.update(await self._generate_layout_async())
                        await asyncio.sleep(self.refresh_rate)
                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        # Log error but keep running
                        self._running = False
                        raise e
        finally:
            if key_task:
                key_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await key_task
            if restore_fn:
                restore_fn()

    async def _handle_key(self, key: str) -> None:
        """Handle keyboard input."""
        if key == "q":
            self._running = False
            self._status_message = "Quitting..."
        elif key == "r":
            self._status_message = "Refreshing..."
        elif key == "c":
            # Close all positions
            self._status_message = "Closing all positions..."
            try:
                orders = await self.broker.close_all_positions()
                if orders:
                    self._status_message = f"Closed {len(orders)} positions"
                    for order in orders:
                        self.add_trade(
                            {
                                "symbol": order.symbol,
                                "side": "sell",
                                "quantity": order.quantity,
                                "price": float(order.filled_avg_price or 0),
                                "time": datetime.now(UTC),
                            }
                        )
                else:
                    self._status_message = "No positions to close"
            except Exception as e:
                self._status_message = f"Error: {e}"
                logger.error(f"Failed to close positions: {e}")
        elif key == "x":
            # Cancel all open orders
            self._status_message = "Cancelling all orders..."
            try:
                orders = await self.broker.get_open_orders()
                cancelled = 0
                for order in orders:
                    if order.id:
                        await self.broker.cancel_order(order.id)
                        cancelled += 1
                self._status_message = f"Cancelled {cancelled} orders"
            except Exception as e:
                self._status_message = f"Error: {e}"
                logger.error(f"Failed to cancel orders: {e}")
        elif key == "h":
            # Clear history
            self._equity_history.clear()
            self._pnl_history.clear()
            self._status_message = "History cleared"
        elif key == "s":
            # Reset session stats
            self._total_trades = 0
            self._winning_trades = 0
            self._strategy_stats.clear()
            self._session_start = datetime.now(UTC)
            self._status_message = "Session stats reset"

    def stop(self) -> None:
        """Stop the dashboard."""
        self._running = False

    def _generate_layout(self) -> Layout:
        """Generate the initial layout structure."""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=4),
        )

        layout["body"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=1),
        )

        layout["left"].split_column(
            Layout(name="account", size=12),
            Layout(name="positions"),
        )

        layout["right"].split_column(
            Layout(name="stats", size=8),
            Layout(name="orders", ratio=1),
            Layout(name="trades", ratio=1),
        )

        # Set placeholders
        layout["header"].update(self._make_header())
        layout["account"].update(Panel("Loading...", title="Account"))
        layout["stats"].update(Panel("Loading...", title="Session Stats"))
        layout["positions"].update(Panel("Loading...", title="Positions"))
        layout["orders"].update(Panel("Loading...", title="Open Orders"))
        layout["trades"].update(Panel("Loading...", title="Recent Trades"))
        layout["footer"].update(self._make_footer())

        return layout

    async def _generate_layout_async(self) -> Layout:
        """Generate layout with live data."""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=4),
        )

        layout["body"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=1),
        )

        layout["left"].split_column(
            Layout(name="account", size=12),
            Layout(name="positions"),
        )

        layout["right"].split_column(
            Layout(name="stats", size=8),
            Layout(name="orders", ratio=1),
            Layout(name="trades", ratio=1),
        )

        # Fetch data
        equity = await self.broker.get_account_value()
        buying_power = await self.broker.get_buying_power()
        cash = await self.broker.get_cash()
        positions = await self.broker.get_positions()
        orders = await self.broker.get_open_orders()

        # Record equity for sparkline
        self.record_equity(equity)

        # Update panels
        layout["header"].update(self._make_header())
        layout["account"].update(
            self._make_account_panel(equity, buying_power, cash, positions)
        )
        layout["stats"].update(self._make_stats_panel())
        layout["positions"].update(self._make_positions_panel(positions))
        layout["orders"].update(self._make_orders_panel(orders))
        layout["trades"].update(self._make_trades_panel())
        layout["footer"].update(self._make_footer())

        return layout

    def _make_header(self) -> Panel:
        """Create header panel."""
        mode = "PAPER" if self.broker.is_paper else "LIVE"
        mode_color = "yellow" if self.broker.is_paper else "red"

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        header_text = Text()
        header_text.append("📈 Trading Dashboard ", style="bold blue")
        header_text.append(f"[{mode}]", style=f"bold {mode_color}")
        header_text.append(f"  |  {now}", style="dim")

        return Panel(header_text, style="blue")

    def _make_footer(self) -> Panel:
        """Create footer with keyboard shortcuts."""
        shortcuts = Text()
        shortcuts.append("[Q] ", style="bold yellow")
        shortcuts.append("Quit  ", style="dim")
        shortcuts.append("[R] ", style="bold yellow")
        shortcuts.append("Refresh  ", style="dim")
        shortcuts.append("[C] ", style="bold yellow")
        shortcuts.append("Close All  ", style="dim")
        shortcuts.append("[X] ", style="bold yellow")
        shortcuts.append("Cancel Orders  ", style="dim")
        shortcuts.append("[H] ", style="bold yellow")
        shortcuts.append("Clear History  ", style="dim")
        shortcuts.append("[S] ", style="bold yellow")
        shortcuts.append("Reset Stats", style="dim")

        # Show status message if any
        if self._status_message:
            status = Text()
            status.append("  → ", style="dim")
            status.append(self._status_message, style="bold cyan")
            from rich.console import Group

            return Panel(Group(shortcuts, status), style="dim")

        return Panel(shortcuts, style="dim")

    def _make_account_panel(
        self,
        equity: Decimal,
        buying_power: Decimal,
        cash: Decimal,
        positions: list[Position],
    ) -> Panel:
        """Create account summary panel with sparkline."""
        from rich.console import Group

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Label", style="dim", width=14)
        table.add_column("Value", style="bold")

        # Equity
        table.add_row("Equity", f"${float(equity):,.2f}")

        # Daily P&L
        if self._start_equity:
            daily_pnl = float(equity - self._start_equity)
            daily_pnl_pct = (
                (daily_pnl / float(self._start_equity)) * 100
                if self._start_equity
                else 0
            )
            pnl_color = "green" if daily_pnl >= 0 else "red"
            pnl_str = (
                f"[{pnl_color}]${daily_pnl:+,.2f} ({daily_pnl_pct:+.2f}%)[/{pnl_color}]"
            )
            table.add_row("Daily P&L", pnl_str)

        # Unrealized P&L from positions
        unrealized_pnl = sum(float(p.unrealized_pnl or 0) for p in positions)
        pnl_color = "green" if unrealized_pnl >= 0 else "red"
        table.add_row(
            "Unrealized P&L",
            f"[{pnl_color}]${unrealized_pnl:+,.2f}[/{pnl_color}]",
        )

        table.add_row("Buying Power", f"${float(buying_power):,.2f}")
        table.add_row("Cash", f"${float(cash):,.2f}")
        table.add_row("Positions", str(len(positions)))

        # Add sparkline if we have history
        if len(self._equity_history) > 1:
            sparkline = _make_sparkline(list(self._equity_history), width=30)
            # Determine color based on trend
            if len(self._equity_history) >= 2:
                trend_color = (
                    "green"
                    if self._equity_history[-1] >= self._equity_history[0]
                    else "red"
                )
            else:
                trend_color = "white"

            spark_text = Text()
            spark_text.append("Equity Trend: ", style="dim")
            spark_text.append(sparkline, style=trend_color)
            return Panel(
                Group(table, Text(""), spark_text),
                title="💰 Account",
                border_style="green",
            )

        return Panel(table, title="💰 Account", border_style="green")

    def _make_positions_panel(self, positions: list[Position]) -> Panel:
        """Create positions table panel."""
        if not positions:
            return Panel(
                Text("No open positions", style="dim italic"),
                title="📊 Positions",
                border_style="blue",
            )

        table = Table(box=None, padding=(0, 1))
        table.add_column("Symbol", style="bold")
        table.add_column("Qty", justify="right")
        table.add_column("Entry", justify="right")
        table.add_column("Current", justify="right")
        table.add_column("P&L", justify="right")
        table.add_column("%", justify="right")

        for pos in positions:
            pnl = float(pos.unrealized_pnl or 0)
            pnl_pct = (pos.unrealized_pnl_pct or 0) * 100
            pnl_color = "green" if pnl >= 0 else "red"

            table.add_row(
                pos.symbol,
                str(pos.quantity),
                f"${float(pos.avg_entry_price):,.2f}",
                f"${float(pos.current_price or 0):,.2f}",
                f"[{pnl_color}]${pnl:+,.2f}[/{pnl_color}]",
                f"[{pnl_color}]{pnl_pct:+.1f}%[/{pnl_color}]",
            )

        return Panel(table, title="📊 Positions", border_style="blue")

    def _make_stats_panel(self) -> Panel:
        """Create session statistics panel."""
        from rich.columns import Columns
        from rich.console import Group

        # Session duration
        duration = datetime.now(UTC) - self._session_start
        hours, remainder = divmod(int(duration.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        # Win rate
        win_rate = (
            (self._winning_trades / self._total_trades * 100)
            if self._total_trades > 0
            else 0
        )

        # Build stats text
        stats_text = Text()
        stats_text.append("Session: ", style="dim")
        stats_text.append(duration_str, style="bold")
        stats_text.append("  |  Trades: ", style="dim")
        stats_text.append(str(self._total_trades), style="bold")
        stats_text.append("  |  Win Rate: ", style="dim")
        win_color = "green" if win_rate >= 50 else "yellow" if win_rate >= 30 else "red"
        stats_text.append(f"{win_rate:.0f}%", style=f"bold {win_color}")

        # Strategy breakdown if we have data
        if self._strategy_stats:
            strat_items = []
            for name, stats in sorted(
                self._strategy_stats.items(), key=lambda x: -x[1]["total_pnl"]
            ):
                strat_win_rate = (
                    stats["wins"] / stats["trades"] * 100 if stats["trades"] > 0 else 0
                )
                pnl_color = "green" if stats["total_pnl"] >= 0 else "red"
                strat_text = Text()
                strat_text.append(f"{name}: ", style="bold")
                strat_text.append(
                    f"${stats['total_pnl']:+,.0f}", style=f"{pnl_color}"
                )
                strat_text.append(f" ({stats['trades']}T, {strat_win_rate:.0f}%W)", style="dim")
                strat_items.append(strat_text)

            if strat_items:
                return Panel(
                    Group(stats_text, Text(""), Columns(strat_items[:4], expand=True)),
                    title="📊 Session Stats",
                    border_style="cyan",
                )

        return Panel(stats_text, title="📊 Session Stats", border_style="cyan")

    def _make_orders_panel(self, orders: list[Order]) -> Panel:
        """Create open orders panel."""
        if not orders:
            return Panel(
                Text("No open orders", style="dim italic"),
                title="📋 Open Orders",
                border_style="yellow",
            )

        table = Table(box=None, padding=(0, 1))
        table.add_column("Symbol", style="bold")
        table.add_column("Side")
        table.add_column("Qty", justify="right")
        table.add_column("Type")
        table.add_column("Status")

        for order in orders[:8]:  # Show max 8 orders
            side_color = "green" if order.side.value == "buy" else "red"
            side_str = f"[{side_color}]{order.side.value.upper()}[/{side_color}]"

            table.add_row(
                order.symbol,
                side_str,
                str(order.quantity),
                order.order_type.value,
                order.status.value,
            )

        if len(orders) > 8:
            table.add_row("...", "", "", "", f"+{len(orders) - 8} more")

        return Panel(table, title="📋 Open Orders", border_style="yellow")

    def _make_trades_panel(self) -> Panel:
        """Create recent trades panel."""
        if not self._recent_trades:
            return Panel(
                Text("No recent trades", style="dim italic"),
                title="📜 Recent Trades",
                border_style="magenta",
            )

        table = Table(box=None, padding=(0, 1))
        table.add_column("Time", style="dim")
        table.add_column("Symbol", style="bold")
        table.add_column("Side")
        table.add_column("Qty", justify="right")
        table.add_column("Price", justify="right")

        for trade in self._recent_trades[:8]:
            side_color = "green" if trade.get("side") == "buy" else "red"
            side_str = f"[{side_color}]{trade.get('side', '').upper()}[/{side_color}]"
            time_str = trade.get("time", "")
            if isinstance(time_str, datetime):
                time_str = time_str.strftime("%H:%M:%S")

            table.add_row(
                str(time_str),
                trade.get("symbol", ""),
                side_str,
                str(trade.get("quantity", "")),
                f"${trade.get('price', 0):,.2f}",
            )

        return Panel(table, title="📜 Recent Trades", border_style="magenta")


async def run_dashboard(broker: BaseBroker, refresh_rate: float = 2.0) -> None:
    """Run the trading dashboard.

    Args:
        broker: Connected broker instance
        refresh_rate: Refresh rate in seconds
    """
    dashboard = TradingDashboard(broker, refresh_rate)
    await dashboard.run()
