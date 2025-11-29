"""Tests for backtesting engine."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pandas as pd
import pytest

from trader.engine.backtest import BacktestEngine, BacktestResult, Trade
from trader.core.models import OrderSide, SignalAction
from trader.strategies.builtin.sma_crossover import SMACrossover


class TestTrade:
    """Tests for Trade dataclass."""

    def test_profitable_trade(self) -> None:
        """Test is_profitable for winning trade."""
        trade = Trade(
            symbol="AAPL",
            entry_time=datetime(2024, 1, 1),
            exit_time=datetime(2024, 1, 10),
            side=OrderSide.BUY,
            quantity=100,
            entry_price=Decimal("100"),
            exit_price=Decimal("110"),
            pnl=Decimal("1000"),
            pnl_pct=0.10,
        )
        assert trade.is_profitable is True

    def test_losing_trade(self) -> None:
        """Test is_profitable for losing trade."""
        trade = Trade(
            symbol="AAPL",
            entry_time=datetime(2024, 1, 1),
            exit_time=datetime(2024, 1, 10),
            side=OrderSide.BUY,
            quantity=100,
            entry_price=Decimal("100"),
            exit_price=Decimal("90"),
            pnl=Decimal("-1000"),
            pnl_pct=-0.10,
        )
        assert trade.is_profitable is False


class TestBacktestResult:
    """Tests for BacktestResult."""

    @pytest.fixture
    def sample_result(self) -> BacktestResult:
        """Create sample backtest result."""
        trades = [
            Trade(
                symbol="TEST",
                entry_time=datetime(2024, 1, 1),
                exit_time=datetime(2024, 1, 5),
                side=OrderSide.BUY,
                quantity=100,
                entry_price=Decimal("100"),
                exit_price=Decimal("110"),
                pnl=Decimal("1000"),
                pnl_pct=0.10,
            ),
            Trade(
                symbol="TEST",
                entry_time=datetime(2024, 1, 10),
                exit_time=datetime(2024, 1, 15),
                side=OrderSide.BUY,
                quantity=100,
                entry_price=Decimal("110"),
                exit_price=Decimal("105"),
                pnl=Decimal("-500"),
                pnl_pct=-0.045,
            ),
            Trade(
                symbol="TEST",
                entry_time=datetime(2024, 1, 20),
                exit_time=datetime(2024, 1, 25),
                side=OrderSide.BUY,
                quantity=100,
                entry_price=Decimal("105"),
                exit_price=Decimal("115"),
                pnl=Decimal("1000"),
                pnl_pct=0.095,
            ),
        ]

        equity = pd.Series(
            [100000, 100500, 101000, 100500, 101500],
            index=pd.date_range("2024-01-01", periods=5),
        )

        return BacktestResult(
            strategy_name="test_strategy",
            symbol="TEST",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            initial_capital=Decimal("100000"),
            final_capital=Decimal("101500"),
            trades=trades,
            equity_curve=equity,
        )

    def test_total_return(self, sample_result: BacktestResult) -> None:
        """Test total return calculation."""
        assert sample_result.total_return == pytest.approx(0.015, rel=0.01)
        assert sample_result.total_return_pct == pytest.approx(1.5, rel=0.01)

    def test_num_trades(self, sample_result: BacktestResult) -> None:
        """Test trade counting."""
        assert sample_result.num_trades == 3
        assert sample_result.winning_trades == 2
        assert sample_result.losing_trades == 1

    def test_win_rate(self, sample_result: BacktestResult) -> None:
        """Test win rate calculation."""
        assert sample_result.win_rate == pytest.approx(0.667, rel=0.01)

    def test_profit_factor(self, sample_result: BacktestResult) -> None:
        """Test profit factor calculation."""
        # Gross profit: 2000, Gross loss: 500
        assert sample_result.profit_factor == pytest.approx(4.0, rel=0.01)

    def test_profit_factor_no_losses(self) -> None:
        """Test profit factor with no losing trades."""
        result = BacktestResult(
            strategy_name="test",
            symbol="TEST",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            initial_capital=Decimal("100000"),
            final_capital=Decimal("110000"),
            trades=[
                Trade(
                    symbol="TEST",
                    entry_time=datetime(2024, 1, 1),
                    exit_time=datetime(2024, 1, 5),
                    side=OrderSide.BUY,
                    quantity=100,
                    entry_price=Decimal("100"),
                    exit_price=Decimal("110"),
                    pnl=Decimal("1000"),
                    pnl_pct=0.10,
                ),
            ],
        )
        assert result.profit_factor == float("inf")

    def test_max_drawdown(self, sample_result: BacktestResult) -> None:
        """Test max drawdown calculation."""
        # Peak: 101000, Trough: 100500
        # Drawdown: (101000 - 100500) / 101000 = 0.00495
        assert sample_result.max_drawdown == pytest.approx(0.00495, rel=0.01)

    def test_summary(self, sample_result: BacktestResult) -> None:
        """Test summary dict generation."""
        summary = sample_result.summary()

        assert summary["strategy"] == "test_strategy"
        assert summary["symbol"] == "TEST"
        assert summary["num_trades"] == 3
        assert "total_return_pct" in summary
        assert "sharpe_ratio" in summary

    def test_empty_result(self) -> None:
        """Test result with no trades."""
        result = BacktestResult(
            strategy_name="test",
            symbol="TEST",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            initial_capital=Decimal("100000"),
            final_capital=Decimal("100000"),
            trades=[],
        )

        assert result.num_trades == 0
        assert result.win_rate == 0.0
        assert result.profit_factor == 0.0


class TestBacktestEngine:
    """Tests for BacktestEngine."""

    @pytest.fixture
    def engine(self) -> BacktestEngine:
        """Create backtest engine."""
        return BacktestEngine(
            initial_capital=100_000.0,
            commission=0.0,
            position_size_pct=1.0,
        )

    @pytest.fixture
    def trending_data(self) -> pd.DataFrame:
        """Create trending data that should generate signals."""
        # Strong uptrend data - should generate buy signal
        prices = list(range(100, 200, 2))  # 50 bars, steady uptrend

        return pd.DataFrame(
            {
                "open": prices,
                "high": [p + 2 for p in prices],
                "low": [p - 1 for p in prices],
                "close": [p + 1 for p in prices],
                "volume": [1000000] * len(prices),
            },
            index=pd.date_range("2024-01-01", periods=len(prices)),
        )

    @pytest.fixture
    def crossover_data(self) -> pd.DataFrame:
        """Create data with clear SMA crossover."""
        # Downtrend followed by uptrend - should trigger buy
        prices = (
            list(range(150, 100, -2))  # 25 bars down
            + list(range(100, 160, 2))  # 30 bars up
        )

        return pd.DataFrame(
            {
                "open": prices,
                "high": [p + 2 for p in prices],
                "low": [p - 1 for p in prices],
                "close": [p + 1 for p in prices],
                "volume": [1000000] * len(prices),
            },
            index=pd.date_range("2024-01-01", periods=len(prices)),
        )

    @pytest.mark.asyncio
    async def test_run_basic(
        self, engine: BacktestEngine, trending_data: pd.DataFrame
    ) -> None:
        """Test basic backtest run."""
        strategy = SMACrossover(fast_period=5, slow_period=10)

        result = await engine.run(
            strategy=strategy,
            data=trending_data,
            symbol="TEST",
        )

        assert result.strategy_name == strategy.name
        assert result.symbol == "TEST"
        assert result.initial_capital == Decimal("100000")
        assert result.equity_curve is not None
        assert len(result.equity_curve) == len(trending_data)

    @pytest.mark.asyncio
    async def test_run_with_crossover(
        self, engine: BacktestEngine, crossover_data: pd.DataFrame
    ) -> None:
        """Test backtest with data that triggers crossover."""
        strategy = SMACrossover(fast_period=5, slow_period=10)

        result = await engine.run(
            strategy=strategy,
            data=crossover_data,
            symbol="TEST",
        )

        # Should have at least one trade from the crossover
        assert result.num_trades >= 1

    @pytest.mark.asyncio
    async def test_run_empty_data_raises(self, engine: BacktestEngine) -> None:
        """Test that empty data raises error."""
        strategy = SMACrossover()
        empty_df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        with pytest.raises(ValueError, match="Data cannot be empty"):
            await engine.run(strategy, empty_df, "TEST")

    @pytest.mark.asyncio
    async def test_commission_applied(self, crossover_data: pd.DataFrame) -> None:
        """Test that commission is deducted."""
        engine_with_commission = BacktestEngine(
            initial_capital=100_000.0,
            commission=10.0,  # $10 per trade
        )
        engine_no_commission = BacktestEngine(
            initial_capital=100_000.0,
            commission=0.0,
        )

        strategy = SMACrossover(fast_period=5, slow_period=10)

        result_with = await engine_with_commission.run(
            strategy, crossover_data, "TEST"
        )
        result_without = await engine_no_commission.run(
            strategy, crossover_data, "TEST"
        )

        # If trades occurred, commission should reduce final capital
        if result_with.num_trades > 0:
            # Each trade has entry and exit, so 2x commission per trade
            expected_commission = result_with.num_trades * 2 * 10
            capital_diff = float(
                result_without.final_capital - result_with.final_capital
            )
            assert capital_diff == pytest.approx(expected_commission, rel=0.1)

    @pytest.mark.asyncio
    async def test_position_size_respected(
        self, crossover_data: pd.DataFrame
    ) -> None:
        """Test that position size percentage is respected."""
        engine_full = BacktestEngine(
            initial_capital=100_000.0,
            position_size_pct=1.0,
        )
        engine_half = BacktestEngine(
            initial_capital=100_000.0,
            position_size_pct=0.5,
        )

        strategy = SMACrossover(fast_period=5, slow_period=10)

        result_full = await engine_full.run(strategy, crossover_data, "TEST")
        result_half = await engine_half.run(strategy, crossover_data, "TEST")

        # Half position should have smaller absolute returns
        if result_full.num_trades > 0 and result_half.num_trades > 0:
            full_change = abs(
                float(result_full.final_capital - result_full.initial_capital)
            )
            half_change = abs(
                float(result_half.final_capital - result_half.initial_capital)
            )
            # Half position should produce roughly half the P&L
            assert half_change < full_change

    @pytest.mark.asyncio
    async def test_trade_records(
        self, engine: BacktestEngine, crossover_data: pd.DataFrame
    ) -> None:
        """Test that trade records are complete."""
        strategy = SMACrossover(fast_period=5, slow_period=10)

        result = await engine.run(strategy, crossover_data, "TEST")

        for trade in result.trades:
            assert trade.symbol == "TEST"
            assert trade.entry_time < trade.exit_time
            assert trade.quantity > 0
            assert trade.entry_price > 0
            assert trade.exit_price > 0
            assert trade.reason_entry != ""
            assert trade.reason_exit != ""
