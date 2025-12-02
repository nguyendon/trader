"""Tests for trading dashboard."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from io import StringIO
from unittest.mock import AsyncMock, MagicMock

import pytest
from rich.console import Console

from trader.core.models import Order, OrderSide, OrderStatus, OrderType, Position
from trader.dashboard.live import TradingDashboard, _make_sparkline


def render_panel_to_str(panel) -> str:
    """Render a Rich panel to a string for testing."""
    console = Console(file=StringIO(), force_terminal=True, width=120)
    console.print(panel)
    return console.file.getvalue()


class TestSparkline:
    """Tests for sparkline generation."""

    def test_empty_values(self) -> None:
        """Test sparkline with empty values."""
        assert _make_sparkline([]) == ""

    def test_single_value(self) -> None:
        """Test sparkline with single value."""
        result = _make_sparkline([100.0])
        assert len(result) == 1

    def test_constant_values(self) -> None:
        """Test sparkline with constant values."""
        result = _make_sparkline([100.0, 100.0, 100.0, 100.0])
        # All same height for constant values
        assert len(result) == 4
        assert len(set(result)) == 1  # All chars are the same

    def test_increasing_values(self) -> None:
        """Test sparkline with increasing values."""
        result = _make_sparkline([1, 2, 3, 4, 5])
        assert len(result) == 5
        # First char should be lowest, last should be highest
        assert result[0] == "▁"
        assert result[-1] == "█"

    def test_decreasing_values(self) -> None:
        """Test sparkline with decreasing values."""
        result = _make_sparkline([5, 4, 3, 2, 1])
        assert len(result) == 5
        # First char should be highest, last should be lowest
        assert result[0] == "█"
        assert result[-1] == "▁"

    def test_width_limit(self) -> None:
        """Test that sparkline respects width limit."""
        values = list(range(100))
        result = _make_sparkline(values, width=10)
        assert len(result) == 10

    def test_mixed_values(self) -> None:
        """Test sparkline with mixed values."""
        values = [50, 25, 75, 100, 0, 50]
        result = _make_sparkline(values)
        assert len(result) == 6
        # 0 should be lowest, 100 should be highest
        assert result[4] == "▁"  # 0 is at index 4
        assert result[3] == "█"  # 100 is at index 3


class TestTradingDashboard:
    """Tests for TradingDashboard class."""

    @pytest.fixture
    def mock_broker(self) -> MagicMock:
        """Create a mock broker."""
        broker = MagicMock()
        broker.is_paper = True
        broker.get_account_value = AsyncMock(return_value=Decimal("100000.00"))
        broker.get_buying_power = AsyncMock(return_value=Decimal("50000.00"))
        broker.get_cash = AsyncMock(return_value=Decimal("25000.00"))
        broker.get_positions = AsyncMock(return_value=[])
        broker.get_open_orders = AsyncMock(return_value=[])
        broker.close_all_positions = AsyncMock(return_value=[])
        broker.cancel_order = AsyncMock()
        return broker

    def test_initialization(self, mock_broker: MagicMock) -> None:
        """Test dashboard initialization."""
        dashboard = TradingDashboard(mock_broker, refresh_rate=1.0)

        assert dashboard.broker is mock_broker
        assert dashboard.refresh_rate == 1.0
        assert dashboard._running is False
        assert len(dashboard._recent_trades) == 0
        assert len(dashboard._equity_history) == 0
        assert dashboard._total_trades == 0

    def test_add_trade(self, mock_broker: MagicMock) -> None:
        """Test adding a trade."""
        dashboard = TradingDashboard(mock_broker)

        trade = {
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 100,
            "price": 150.0,
            "pnl": 50.0,
            "strategy": "sma",
        }
        dashboard.add_trade(trade)

        assert len(dashboard._recent_trades) == 1
        assert dashboard._total_trades == 1
        assert dashboard._winning_trades == 1
        assert "sma" in dashboard._strategy_stats
        assert dashboard._strategy_stats["sma"]["trades"] == 1
        assert dashboard._strategy_stats["sma"]["wins"] == 1
        assert dashboard._strategy_stats["sma"]["total_pnl"] == 50.0

    def test_add_losing_trade(self, mock_broker: MagicMock) -> None:
        """Test adding a losing trade."""
        dashboard = TradingDashboard(mock_broker)

        trade = {
            "symbol": "AAPL",
            "side": "sell",
            "quantity": 100,
            "price": 150.0,
            "pnl": -25.0,
            "strategy": "rsi",
        }
        dashboard.add_trade(trade)

        assert dashboard._total_trades == 1
        assert dashboard._winning_trades == 0
        assert dashboard._strategy_stats["rsi"]["wins"] == 0
        assert dashboard._strategy_stats["rsi"]["total_pnl"] == -25.0

    def test_add_multiple_trades(self, mock_broker: MagicMock) -> None:
        """Test adding multiple trades."""
        dashboard = TradingDashboard(mock_broker)

        for i in range(15):
            dashboard.add_trade({"symbol": f"SYM{i}", "pnl": 10.0})

        # Should keep only last 10
        assert len(dashboard._recent_trades) == 10
        assert dashboard._total_trades == 15

    def test_record_equity(self, mock_broker: MagicMock) -> None:
        """Test recording equity for sparkline."""
        dashboard = TradingDashboard(mock_broker)
        dashboard._start_equity = Decimal("100000.00")

        dashboard.record_equity(Decimal("100500.00"))
        dashboard.record_equity(Decimal("101000.00"))
        dashboard.record_equity(Decimal("100750.00"))

        assert len(dashboard._equity_history) == 3
        assert list(dashboard._equity_history) == [100500.0, 101000.0, 100750.0]
        assert len(dashboard._pnl_history) == 3
        assert list(dashboard._pnl_history) == [500.0, 1000.0, 750.0]

    def test_stop(self, mock_broker: MagicMock) -> None:
        """Test stopping the dashboard."""
        dashboard = TradingDashboard(mock_broker)
        dashboard._running = True

        dashboard.stop()

        assert dashboard._running is False

    def test_make_header(self, mock_broker: MagicMock) -> None:
        """Test header panel creation."""
        dashboard = TradingDashboard(mock_broker)
        panel = dashboard._make_header()

        assert panel is not None
        assert "Trading Dashboard" in str(panel.renderable)
        assert "PAPER" in str(panel.renderable)

    def test_make_header_live_mode(self, mock_broker: MagicMock) -> None:
        """Test header shows LIVE for live trading."""
        mock_broker.is_paper = False
        dashboard = TradingDashboard(mock_broker)
        panel = dashboard._make_header()

        assert "LIVE" in str(panel.renderable)

    def test_make_footer(self, mock_broker: MagicMock) -> None:
        """Test footer panel with shortcuts."""
        dashboard = TradingDashboard(mock_broker)
        panel = dashboard._make_footer()

        footer_str = render_panel_to_str(panel)
        assert "[Q]" in footer_str
        assert "[P]" in footer_str  # Positions menu
        assert "[O]" in footer_str  # Orders menu
        assert "[C]" in footer_str
        assert "[X]" in footer_str
        assert "[?]" in footer_str  # Help

    def test_make_footer_with_status(self, mock_broker: MagicMock) -> None:
        """Test footer shows status message."""
        dashboard = TradingDashboard(mock_broker)
        dashboard._status_message = "Test status"
        panel = dashboard._make_footer()

        panel_str = render_panel_to_str(panel)
        assert "Test status" in panel_str

    def test_make_account_panel(self, mock_broker: MagicMock) -> None:
        """Test account panel creation."""
        dashboard = TradingDashboard(mock_broker)
        dashboard._start_equity = Decimal("100000.00")

        panel = dashboard._make_account_panel(
            equity=Decimal("100500.00"),
            buying_power=Decimal("50000.00"),
            cash=Decimal("25000.00"),
            positions=[],
        )

        panel_str = render_panel_to_str(panel)
        assert "100,500.00" in panel_str
        assert "50,000.00" in panel_str
        assert "Account" in panel_str

    def test_make_account_panel_with_sparkline(self, mock_broker: MagicMock) -> None:
        """Test account panel includes sparkline when history exists."""
        dashboard = TradingDashboard(mock_broker)
        dashboard._start_equity = Decimal("100000.00")

        # Add some equity history
        for i in range(10):
            dashboard._equity_history.append(100000.0 + i * 100)

        panel = dashboard._make_account_panel(
            equity=Decimal("100900.00"),
            buying_power=Decimal("50000.00"),
            cash=Decimal("25000.00"),
            positions=[],
        )

        # Panel should contain sparkline characters
        panel_str = render_panel_to_str(panel)
        assert "Equity Trend" in panel_str

    def test_make_positions_panel_empty(self, mock_broker: MagicMock) -> None:
        """Test positions panel when no positions."""
        dashboard = TradingDashboard(mock_broker)
        panel = dashboard._make_positions_panel([])

        assert "No open positions" in str(panel.renderable)

    def test_make_positions_panel_with_positions(self, mock_broker: MagicMock) -> None:
        """Test positions panel with positions."""
        dashboard = TradingDashboard(mock_broker)

        positions = [
            Position(
                symbol="AAPL",
                quantity=100,
                avg_entry_price=Decimal("150.00"),
                current_price=Decimal("155.00"),
                unrealized_pnl=Decimal("500.00"),
                unrealized_pnl_pct=0.0333,
            ),
            Position(
                symbol="MSFT",
                quantity=50,
                avg_entry_price=Decimal("300.00"),
                current_price=Decimal("295.00"),
                unrealized_pnl=Decimal("-250.00"),
                unrealized_pnl_pct=-0.0167,
            ),
        ]

        panel = dashboard._make_positions_panel(positions)
        panel_str = render_panel_to_str(panel)

        assert "AAPL" in panel_str
        assert "MSFT" in panel_str
        assert "500.00" in panel_str

    def test_make_orders_panel_empty(self, mock_broker: MagicMock) -> None:
        """Test orders panel when no orders."""
        dashboard = TradingDashboard(mock_broker)
        panel = dashboard._make_orders_panel([])

        assert "No open orders" in str(panel.renderable)

    def test_make_orders_panel_with_orders(self, mock_broker: MagicMock) -> None:
        """Test orders panel with orders."""
        dashboard = TradingDashboard(mock_broker)

        orders = [
            Order(
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=100,
                order_type=OrderType.MARKET,
                status=OrderStatus.PENDING,
            ),
        ]

        panel = dashboard._make_orders_panel(orders)
        panel_str = render_panel_to_str(panel)

        assert "AAPL" in panel_str
        assert "BUY" in panel_str

    def test_make_trades_panel_empty(self, mock_broker: MagicMock) -> None:
        """Test trades panel when no trades."""
        dashboard = TradingDashboard(mock_broker)
        panel = dashboard._make_trades_panel()

        assert "No recent trades" in str(panel.renderable)

    def test_make_trades_panel_with_trades(self, mock_broker: MagicMock) -> None:
        """Test trades panel with trades."""
        dashboard = TradingDashboard(mock_broker)
        dashboard._recent_trades = [
            {
                "symbol": "AAPL",
                "side": "buy",
                "quantity": 100,
                "price": 150.0,
                "time": datetime.now(UTC),
            }
        ]

        panel = dashboard._make_trades_panel()
        panel_str = render_panel_to_str(panel)

        assert "AAPL" in panel_str
        assert "BUY" in panel_str

    def test_make_stats_panel(self, mock_broker: MagicMock) -> None:
        """Test stats panel creation."""
        dashboard = TradingDashboard(mock_broker)
        panel = dashboard._make_stats_panel()

        panel_str = render_panel_to_str(panel)
        assert "Session" in panel_str
        assert "Trades" in panel_str
        assert "Win Rate" in panel_str

    def test_make_stats_panel_with_strategy_stats(
        self, mock_broker: MagicMock
    ) -> None:
        """Test stats panel with strategy breakdown."""
        dashboard = TradingDashboard(mock_broker)
        dashboard._strategy_stats = {
            "sma": {"trades": 10, "wins": 6, "total_pnl": 500.0},
            "rsi": {"trades": 5, "wins": 2, "total_pnl": -100.0},
        }

        panel = dashboard._make_stats_panel()
        panel_str = render_panel_to_str(panel)

        assert "sma" in panel_str
        assert "rsi" in panel_str

    @pytest.mark.asyncio
    async def test_handle_key_quit(self, mock_broker: MagicMock) -> None:
        """Test quit key handler."""
        dashboard = TradingDashboard(mock_broker)
        dashboard._running = True

        await dashboard._handle_key("q")

        assert dashboard._running is False
        assert "Quitting" in dashboard._status_message

    @pytest.mark.asyncio
    async def test_handle_key_refresh(self, mock_broker: MagicMock) -> None:
        """Test refresh key handler."""
        dashboard = TradingDashboard(mock_broker)

        await dashboard._handle_key("r")

        assert "Refreshing" in dashboard._status_message

    @pytest.mark.asyncio
    async def test_handle_key_close_positions(self, mock_broker: MagicMock) -> None:
        """Test close all positions key handler."""
        mock_broker.close_all_positions.return_value = []
        dashboard = TradingDashboard(mock_broker)

        await dashboard._handle_key("c")

        mock_broker.close_all_positions.assert_called_once()
        assert "No positions to close" in dashboard._status_message

    @pytest.mark.asyncio
    async def test_handle_key_cancel_orders(self, mock_broker: MagicMock) -> None:
        """Test cancel orders key handler."""
        order = MagicMock()
        order.id = "order123"
        mock_broker.get_open_orders.return_value = [order]
        dashboard = TradingDashboard(mock_broker)

        await dashboard._handle_key("x")

        mock_broker.cancel_order.assert_called_once_with("order123")
        assert "Cancelled 1 orders" in dashboard._status_message

    @pytest.mark.asyncio
    async def test_handle_key_clear_history(self, mock_broker: MagicMock) -> None:
        """Test clear history key handler."""
        dashboard = TradingDashboard(mock_broker)
        dashboard._equity_history.append(100.0)
        dashboard._pnl_history.append(10.0)

        await dashboard._handle_key("h")

        assert len(dashboard._equity_history) == 0
        assert len(dashboard._pnl_history) == 0
        assert "History cleared" in dashboard._status_message

    @pytest.mark.asyncio
    async def test_handle_key_reset_stats(self, mock_broker: MagicMock) -> None:
        """Test reset stats key handler."""
        dashboard = TradingDashboard(mock_broker)
        dashboard._total_trades = 10
        dashboard._winning_trades = 5
        dashboard._strategy_stats = {"test": {}}

        await dashboard._handle_key("s")

        assert dashboard._total_trades == 0
        assert dashboard._winning_trades == 0
        assert len(dashboard._strategy_stats) == 0
        assert "Session stats reset" in dashboard._status_message

    @pytest.mark.asyncio
    async def test_handle_key_open_positions_menu(self, mock_broker: MagicMock) -> None:
        """Test opening positions menu."""
        position = MagicMock()
        position.symbol = "AAPL"
        mock_broker.get_positions.return_value = [position]
        dashboard = TradingDashboard(mock_broker)

        await dashboard._handle_key("p")

        assert dashboard._menu_mode == "positions"
        assert len(dashboard._menu_items) == 1
        assert dashboard._selected_index == 0

    @pytest.mark.asyncio
    async def test_handle_key_open_orders_menu(self, mock_broker: MagicMock) -> None:
        """Test opening orders menu."""
        order = MagicMock()
        order.symbol = "AAPL"
        mock_broker.get_open_orders.return_value = [order]
        dashboard = TradingDashboard(mock_broker)

        await dashboard._handle_key("o")

        assert dashboard._menu_mode == "orders"
        assert len(dashboard._menu_items) == 1

    @pytest.mark.asyncio
    async def test_handle_key_help(self, mock_broker: MagicMock) -> None:
        """Test opening help panel."""
        dashboard = TradingDashboard(mock_broker)

        await dashboard._handle_key("?")

        assert dashboard._menu_mode == "help"

    @pytest.mark.asyncio
    async def test_menu_navigation(self, mock_broker: MagicMock) -> None:
        """Test menu navigation with j/k keys."""
        dashboard = TradingDashboard(mock_broker)
        dashboard._menu_mode = "positions"
        dashboard._menu_items = [MagicMock(), MagicMock(), MagicMock()]
        dashboard._selected_index = 0

        # Navigate down
        await dashboard._handle_menu_key("j")
        assert dashboard._selected_index == 1

        await dashboard._handle_menu_key("j")
        assert dashboard._selected_index == 2

        # Wrap around
        await dashboard._handle_menu_key("j")
        assert dashboard._selected_index == 0

        # Navigate up
        await dashboard._handle_menu_key("k")
        assert dashboard._selected_index == 2

    @pytest.mark.asyncio
    async def test_menu_close_on_escape(self, mock_broker: MagicMock) -> None:
        """Test closing menu with escape/q."""
        dashboard = TradingDashboard(mock_broker)
        dashboard._menu_mode = "positions"
        dashboard._menu_items = [MagicMock()]

        await dashboard._handle_menu_key("q")

        assert dashboard._menu_mode is None
        assert len(dashboard._menu_items) == 0

    @pytest.mark.asyncio
    async def test_menu_close_on_esc_key(self, mock_broker: MagicMock) -> None:
        """Test closing menu with ESC key."""
        dashboard = TradingDashboard(mock_broker)
        dashboard._menu_mode = "positions"
        dashboard._menu_items = [MagicMock()]

        await dashboard._handle_menu_key("ESC")

        assert dashboard._menu_mode is None
        assert len(dashboard._menu_items) == 0

    @pytest.mark.asyncio
    async def test_menu_navigation_arrow_keys(self, mock_broker: MagicMock) -> None:
        """Test menu navigation with arrow keys."""
        dashboard = TradingDashboard(mock_broker)
        dashboard._menu_mode = "positions"
        dashboard._menu_items = [MagicMock(), MagicMock(), MagicMock()]
        dashboard._selected_index = 0

        # Navigate down with arrow key
        await dashboard._handle_menu_key("DOWN")
        assert dashboard._selected_index == 1

        # Navigate up with arrow key
        await dashboard._handle_menu_key("UP")
        assert dashboard._selected_index == 0

    @pytest.mark.asyncio
    async def test_menu_number_selection(self, mock_broker: MagicMock) -> None:
        """Test direct number selection in menu."""
        position = MagicMock()
        position.symbol = "AAPL"
        position.current_price = Decimal("150.00")
        mock_broker.close_position.return_value = MagicMock(
            quantity=100, filled_avg_price=Decimal("150.00")
        )

        dashboard = TradingDashboard(mock_broker)
        dashboard._menu_mode = "positions"
        dashboard._menu_items = [position]
        dashboard._selected_index = 0

        await dashboard._handle_menu_key("1")

        mock_broker.close_position.assert_called_once_with("AAPL")
        assert dashboard._menu_mode is None

    def test_make_menu_panel_positions(self, mock_broker: MagicMock) -> None:
        """Test positions menu panel."""
        dashboard = TradingDashboard(mock_broker)
        dashboard._selected_index = 0

        positions = [
            Position(
                symbol="AAPL",
                quantity=100,
                avg_entry_price=Decimal("150.00"),
                current_price=Decimal("155.00"),
                unrealized_pnl=Decimal("500.00"),
                unrealized_pnl_pct=0.0333,
            ),
        ]

        panel = dashboard._make_menu_panel("positions", positions)
        panel_str = render_panel_to_str(panel)

        assert "Select Position to Close" in panel_str
        assert "AAPL" in panel_str
        assert "►1" in panel_str  # Selected indicator

    def test_make_menu_panel_orders(self, mock_broker: MagicMock) -> None:
        """Test orders menu panel."""
        dashboard = TradingDashboard(mock_broker)
        dashboard._selected_index = 0

        orders = [
            Order(
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=100,
                order_type=OrderType.MARKET,
                status=OrderStatus.PENDING,
            ),
        ]

        panel = dashboard._make_menu_panel("orders", orders)
        panel_str = render_panel_to_str(panel)

        assert "Select Order to Cancel" in panel_str
        assert "AAPL" in panel_str

    def test_make_help_panel(self, mock_broker: MagicMock) -> None:
        """Test help panel content."""
        dashboard = TradingDashboard(mock_broker)
        panel = dashboard._make_help_panel()
        panel_str = render_panel_to_str(panel)

        assert "Keyboard Shortcuts" in panel_str
        assert "[Q]" in panel_str
        assert "[P]" in panel_str
        assert "[O]" in panel_str
        assert "Navigate" in panel_str
