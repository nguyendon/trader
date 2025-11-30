"""Discord webhook notifications."""

from __future__ import annotations

import asyncio
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

import httpx
from loguru import logger

if TYPE_CHECKING:
    from trader.core.models import Order, Position, Signal


class DiscordNotifier:
    """Send trading notifications to Discord via webhook."""

    def __init__(self, webhook_url: str, username: str = "Trader Bot") -> None:
        """Initialize Discord notifier.

        Args:
            webhook_url: Discord webhook URL
            username: Bot username to display
        """
        self.webhook_url = webhook_url
        self.username = username
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _send(self, content: str | None = None, embed: dict | None = None) -> bool:
        """Send a message to Discord.

        Args:
            content: Plain text message
            embed: Rich embed object

        Returns:
            True if sent successfully
        """
        client = await self._get_client()

        payload: dict = {"username": self.username}
        if content:
            payload["content"] = content
        if embed:
            payload["embeds"] = [embed]

        try:
            response = await client.post(self.webhook_url, json=payload)
            if response.status_code in (200, 204):
                logger.debug("Discord notification sent")
                return True
            else:
                logger.warning(f"Discord webhook failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Discord notification error: {e}")
            return False

    async def send_message(self, message: str) -> bool:
        """Send a simple text message."""
        return await self._send(content=message)

    async def notify_trade_signal(
        self,
        signal: Signal,
        symbol: str,
        quantity: int,
        price: Decimal,
        approved: bool = True,
    ) -> bool:
        """Notify about a trade signal.

        Args:
            signal: The trading signal
            symbol: Stock symbol
            quantity: Number of shares
            price: Current price
            approved: Whether the trade was approved/executed
        """
        action = signal.action.value.upper()
        value = float(price) * quantity

        if approved:
            color = 0x00FF00 if action == "BUY" else 0xFF6B6B  # Green for buy, red for sell
            title = f"🔔 Trade Executed: {action} {symbol}"
        else:
            color = 0xFFAA00  # Orange for rejected
            title = f"⚠️ Trade Rejected: {action} {symbol}"

        embed = {
            "title": title,
            "color": color,
            "fields": [
                {"name": "Symbol", "value": symbol, "inline": True},
                {"name": "Action", "value": action, "inline": True},
                {"name": "Quantity", "value": str(quantity), "inline": True},
                {"name": "Price", "value": f"${float(price):,.2f}", "inline": True},
                {"name": "Value", "value": f"${value:,.2f}", "inline": True},
                {"name": "Reason", "value": signal.reason or "N/A", "inline": False},
            ],
            "timestamp": datetime.utcnow().isoformat(),
        }

        return await self._send(embed=embed)

    async def notify_order_filled(
        self,
        order: Order,
    ) -> bool:
        """Notify when an order is filled.

        Args:
            order: The filled order
        """
        side = order.side.value.upper()
        color = 0x00FF00 if side == "BUY" else 0xFF6B6B

        filled_price = order.filled_avg_price or Decimal(0)
        value = float(filled_price) * order.filled_quantity

        embed = {
            "title": f"✅ Order Filled: {side} {order.symbol}",
            "color": color,
            "fields": [
                {"name": "Symbol", "value": order.symbol, "inline": True},
                {"name": "Side", "value": side, "inline": True},
                {"name": "Quantity", "value": str(order.filled_quantity), "inline": True},
                {"name": "Fill Price", "value": f"${float(filled_price):,.2f}", "inline": True},
                {"name": "Total Value", "value": f"${value:,.2f}", "inline": True},
                {"name": "Order ID", "value": order.broker_order_id or "N/A", "inline": True},
            ],
            "timestamp": datetime.utcnow().isoformat(),
        }

        return await self._send(embed=embed)

    async def notify_circuit_breaker(
        self,
        reason: str,
        daily_trades: int,
        daily_pnl: float,
    ) -> bool:
        """Notify when circuit breaker is triggered.

        Args:
            reason: Why the circuit breaker triggered
            daily_trades: Number of trades today
            daily_pnl: P&L for the day
        """
        embed = {
            "title": "🛑 Circuit Breaker Triggered",
            "color": 0xFF0000,  # Red
            "description": "Trading has been paused due to safety limits.",
            "fields": [
                {"name": "Reason", "value": reason, "inline": False},
                {"name": "Daily Trades", "value": str(daily_trades), "inline": True},
                {"name": "Daily P&L", "value": f"${daily_pnl:+,.2f}", "inline": True},
            ],
            "timestamp": datetime.utcnow().isoformat(),
        }

        return await self._send(embed=embed)

    async def notify_engine_started(
        self,
        symbols: list[str],
        strategy: str,
        account_value: float,
        is_paper: bool = True,
    ) -> bool:
        """Notify when trading engine starts.

        Args:
            symbols: Symbols being traded
            strategy: Strategy name
            account_value: Current account value
            is_paper: Whether using paper trading
        """
        mode = "📄 Paper" if is_paper else "💰 LIVE"

        embed = {
            "title": "🚀 Trading Engine Started",
            "color": 0x00BFFF,  # Blue
            "fields": [
                {"name": "Mode", "value": mode, "inline": True},
                {"name": "Strategy", "value": strategy, "inline": True},
                {"name": "Account Value", "value": f"${account_value:,.2f}", "inline": True},
                {"name": "Symbols", "value": ", ".join(symbols), "inline": False},
            ],
            "timestamp": datetime.utcnow().isoformat(),
        }

        return await self._send(embed=embed)

    async def notify_engine_stopped(
        self,
        reason: str,
        daily_trades: int,
        daily_pnl: float,
    ) -> bool:
        """Notify when trading engine stops.

        Args:
            reason: Why the engine stopped
            daily_trades: Number of trades executed today
            daily_pnl: P&L for the day
        """
        pnl_color = 0x00FF00 if daily_pnl >= 0 else 0xFF0000

        embed = {
            "title": "🔴 Trading Engine Stopped",
            "color": pnl_color,
            "fields": [
                {"name": "Reason", "value": reason, "inline": False},
                {"name": "Trades Today", "value": str(daily_trades), "inline": True},
                {"name": "Daily P&L", "value": f"${daily_pnl:+,.2f}", "inline": True},
            ],
            "timestamp": datetime.utcnow().isoformat(),
        }

        return await self._send(embed=embed)

    async def notify_daily_summary(
        self,
        date: str,
        starting_equity: float,
        ending_equity: float,
        realized_pnl: float,
        num_trades: int,
        winning_trades: int,
        losing_trades: int,
    ) -> bool:
        """Send end-of-day summary.

        Args:
            date: Trading date
            starting_equity: Starting account value
            ending_equity: Ending account value
            realized_pnl: Realized P&L
            num_trades: Number of trades
            winning_trades: Number of winning trades
            losing_trades: Number of losing trades
        """
        daily_return = ((ending_equity - starting_equity) / starting_equity) * 100
        color = 0x00FF00 if realized_pnl >= 0 else 0xFF0000

        win_rate = (winning_trades / num_trades * 100) if num_trades > 0 else 0

        embed = {
            "title": f"📊 Daily Summary - {date}",
            "color": color,
            "fields": [
                {"name": "Starting Equity", "value": f"${starting_equity:,.2f}", "inline": True},
                {"name": "Ending Equity", "value": f"${ending_equity:,.2f}", "inline": True},
                {"name": "Daily Return", "value": f"{daily_return:+.2f}%", "inline": True},
                {"name": "Realized P&L", "value": f"${realized_pnl:+,.2f}", "inline": True},
                {"name": "Trades", "value": str(num_trades), "inline": True},
                {"name": "Win Rate", "value": f"{win_rate:.0f}% ({winning_trades}W/{losing_trades}L)", "inline": True},
            ],
            "timestamp": datetime.utcnow().isoformat(),
        }

        return await self._send(embed=embed)

    async def notify_position_update(
        self,
        positions: list[Position],
        total_value: float,
        total_pnl: float,
    ) -> bool:
        """Send position update.

        Args:
            positions: List of current positions
            total_value: Total portfolio value
            total_pnl: Total unrealized P&L
        """
        color = 0x00FF00 if total_pnl >= 0 else 0xFF0000

        if not positions:
            description = "No open positions"
        else:
            lines = []
            for pos in positions[:10]:  # Limit to 10
                pnl = float(pos.unrealized_pnl or 0)
                pnl_str = f"+${pnl:,.0f}" if pnl >= 0 else f"-${abs(pnl):,.0f}"
                lines.append(f"**{pos.symbol}**: {pos.quantity} @ ${float(pos.avg_entry_price):,.2f} ({pnl_str})")
            description = "\n".join(lines)

        embed = {
            "title": f"📈 Position Update ({len(positions)} positions)",
            "color": color,
            "description": description,
            "fields": [
                {"name": "Total Value", "value": f"${total_value:,.2f}", "inline": True},
                {"name": "Unrealized P&L", "value": f"${total_pnl:+,.2f}", "inline": True},
            ],
            "timestamp": datetime.utcnow().isoformat(),
        }

        return await self._send(embed=embed)

    async def notify_error(self, error: str, context: str | None = None) -> bool:
        """Send error notification.

        Args:
            error: Error message
            context: Additional context
        """
        embed = {
            "title": "❌ Error",
            "color": 0xFF0000,
            "description": error[:2000],  # Discord limit
            "timestamp": datetime.utcnow().isoformat(),
        }

        if context:
            embed["fields"] = [{"name": "Context", "value": context[:1000], "inline": False}]

        return await self._send(embed=embed)


# Global notifier instance
_notifier: DiscordNotifier | None = None


def get_discord_notifier(webhook_url: str | None = None) -> DiscordNotifier | None:
    """Get or create the global Discord notifier.

    Args:
        webhook_url: Discord webhook URL. Required on first call.

    Returns:
        DiscordNotifier instance or None if no URL configured
    """
    global _notifier

    if webhook_url:
        _notifier = DiscordNotifier(webhook_url)

    return _notifier
