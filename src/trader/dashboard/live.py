"""Live trading dashboard with Rich terminal UI."""

from __future__ import annotations

import asyncio
import contextlib
import sys
import typing
from datetime import datetime
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


def _setup_keyboard_listener() -> (
    tuple[asyncio.Queue[str], typing.Callable[[], typing.Coroutine], typing.Callable[[], None]]
):
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
    - Open positions with live P&L
    - Recent trades
    - Open orders
    - Keyboard shortcuts
    """

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

    def add_trade(self, trade: dict) -> None:
        """Add a trade to the recent trades list."""
        self._recent_trades.insert(0, trade)
        # Keep only last 10 trades
        self._recent_trades = self._recent_trades[:10]

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
                        self.add_trade({
                            "symbol": order.symbol,
                            "side": "sell",
                            "quantity": order.quantity,
                            "price": float(order.filled_avg_price or 0),
                            "time": datetime.now(),
                        })
                else:
                    self._status_message = "No positions to close"
            except Exception as e:
                self._status_message = f"Error: {e}"
                logger.error(f"Failed to close positions: {e}")

    def stop(self) -> None:
        """Stop the dashboard."""
        self._running = False

    def _generate_layout(self) -> Layout:
        """Generate the initial layout structure."""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )

        layout["body"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=1),
        )

        layout["left"].split_column(
            Layout(name="account", size=10),
            Layout(name="positions"),
        )

        layout["right"].split_column(
            Layout(name="orders", ratio=1),
            Layout(name="trades", ratio=1),
        )

        # Set placeholders
        layout["header"].update(self._make_header())
        layout["account"].update(Panel("Loading...", title="Account"))
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
            Layout(name="footer", size=3),
        )

        layout["body"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=1),
        )

        layout["left"].split_column(
            Layout(name="account", size=10),
            Layout(name="positions"),
        )

        layout["right"].split_column(
            Layout(name="orders", ratio=1),
            Layout(name="trades", ratio=1),
        )

        # Fetch data
        equity = await self.broker.get_account_value()
        buying_power = await self.broker.get_buying_power()
        cash = await self.broker.get_cash()
        positions = await self.broker.get_positions()
        orders = await self.broker.get_open_orders()

        # Update panels
        layout["header"].update(self._make_header())
        layout["account"].update(
            self._make_account_panel(equity, buying_power, cash, positions)
        )
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
        footer_text = Text()
        footer_text.append("  [Q] ", style="bold yellow")
        footer_text.append("Quit", style="dim")
        footer_text.append("  |  [R] ", style="bold yellow")
        footer_text.append("Refresh", style="dim")
        footer_text.append("  |  [C] ", style="bold yellow")
        footer_text.append("Close All", style="dim")

        # Show status message if any
        if self._status_message:
            footer_text.append("  |  ", style="dim")
            footer_text.append(self._status_message, style="bold cyan")

        return Panel(footer_text, style="dim")

    def _make_account_panel(
        self,
        equity: Decimal,
        buying_power: Decimal,
        cash: Decimal,
        positions: list[Position],
    ) -> Panel:
        """Create account summary panel."""
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
            pnl_str = f"[{pnl_color}]${daily_pnl:+,.2f} ({daily_pnl_pct:+.2f}%)[/{pnl_color}]"
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
