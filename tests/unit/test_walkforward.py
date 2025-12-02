"""Tests for walk-forward optimization."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from trader.engine.backtest import BacktestResult
from trader.engine.costs import CostModel
from trader.engine.walkforward import (
    WalkForwardOptimizer,
    WalkForwardResult,
    WalkForwardWindow,
)


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    n = 500  # 500 bars for walk-forward testing
    dates = pd.date_range(end="2024-01-01", periods=n, freq="D")

    # Generate trending price data
    base_price = 100.0
    returns = np.random.randn(n) * 0.02
    prices = base_price * np.cumprod(1 + returns)

    return pd.DataFrame(
        {
            "open": prices * (1 + np.random.randn(n) * 0.005),
            "high": prices * (1 + abs(np.random.randn(n) * 0.01)),
            "low": prices * (1 - abs(np.random.randn(n) * 0.01)),
            "close": prices,
            "volume": np.random.randint(100000, 1000000, n),
        },
        index=dates,
    )


class TestWalkForwardWindow:
    """Tests for WalkForwardWindow."""

    def test_initialization(self) -> None:
        """Test window initialization."""
        window = WalkForwardWindow(
            window_id=1,
            train_start=datetime(2023, 1, 1),
            train_end=datetime(2023, 6, 30),
            test_start=datetime(2023, 7, 1),
            test_end=datetime(2023, 9, 30),
            train_bars=180,
            test_bars=90,
        )

        assert window.window_id == 1
        assert window.train_bars == 180
        assert window.test_bars == 90

    def test_train_return(self) -> None:
        """Test train return property."""
        window = WalkForwardWindow(
            window_id=1,
            train_start=datetime(2023, 1, 1),
            train_end=datetime(2023, 6, 30),
            test_start=datetime(2023, 7, 1),
            test_end=datetime(2023, 9, 30),
            train_bars=180,
            test_bars=90,
        )

        # No result yet
        assert window.train_return == 0.0

        # Add train result
        window.train_result = BacktestResult(
            strategy_name="test",
            symbol="AAPL",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 6, 30),
            initial_capital=Decimal("100000"),
            final_capital=Decimal("110000"),
        )

        assert window.train_return == 0.1

    def test_efficiency_ratio(self) -> None:
        """Test efficiency ratio calculation."""
        window = WalkForwardWindow(
            window_id=1,
            train_start=datetime(2023, 1, 1),
            train_end=datetime(2023, 6, 30),
            test_start=datetime(2023, 7, 1),
            test_end=datetime(2023, 9, 30),
            train_bars=180,
            test_bars=90,
        )

        # Train result: +10%
        window.train_result = BacktestResult(
            strategy_name="test",
            symbol="AAPL",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 6, 30),
            initial_capital=Decimal("100000"),
            final_capital=Decimal("110000"),
        )

        # Test result: +5%
        window.test_result = BacktestResult(
            strategy_name="test",
            symbol="AAPL",
            start_date=datetime(2023, 7, 1),
            end_date=datetime(2023, 9, 30),
            initial_capital=Decimal("100000"),
            final_capital=Decimal("105000"),
        )

        # Efficiency = 0.05 / 0.10 = 0.5
        assert window.efficiency_ratio == 0.5

    def test_efficiency_ratio_zero_train(self) -> None:
        """Test efficiency ratio when train return is zero."""
        window = WalkForwardWindow(
            window_id=1,
            train_start=datetime(2023, 1, 1),
            train_end=datetime(2023, 6, 30),
            test_start=datetime(2023, 7, 1),
            test_end=datetime(2023, 9, 30),
            train_bars=180,
            test_bars=90,
        )

        # Train result: 0%
        window.train_result = BacktestResult(
            strategy_name="test",
            symbol="AAPL",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 6, 30),
            initial_capital=Decimal("100000"),
            final_capital=Decimal("100000"),
        )

        assert window.efficiency_ratio == 0.0


class TestWalkForwardResult:
    """Tests for WalkForwardResult."""

    def test_initialization(self) -> None:
        """Test result initialization."""
        result = WalkForwardResult(
            strategy_name="SMA",
            symbol="AAPL",
            total_windows=5,
            train_pct=0.7,
        )

        assert result.strategy_name == "SMA"
        assert result.symbol == "AAPL"
        assert result.total_windows == 5
        assert result.train_pct == 0.7

    def test_avg_returns(self) -> None:
        """Test average return calculations."""
        result = WalkForwardResult(
            strategy_name="SMA",
            symbol="AAPL",
            total_windows=2,
            train_pct=0.7,
        )

        # Add windows with results
        for i, (train_return, test_return) in enumerate([(0.10, 0.05), (0.08, 0.06)]):
            window = WalkForwardWindow(
                window_id=i + 1,
                train_start=datetime(2023, 1, 1),
                train_end=datetime(2023, 6, 30),
                test_start=datetime(2023, 7, 1),
                test_end=datetime(2023, 9, 30),
                train_bars=180,
                test_bars=90,
            )
            window.train_result = BacktestResult(
                strategy_name="test",
                symbol="AAPL",
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 6, 30),
                initial_capital=Decimal("100000"),
                final_capital=Decimal(str(100000 * (1 + train_return))),
            )
            window.test_result = BacktestResult(
                strategy_name="test",
                symbol="AAPL",
                start_date=datetime(2023, 7, 1),
                end_date=datetime(2023, 9, 30),
                initial_capital=Decimal("100000"),
                final_capital=Decimal(str(100000 * (1 + test_return))),
            )
            result.windows.append(window)

        # Avg train = (0.10 + 0.08) / 2 = 0.09
        assert abs(result.avg_train_return - 0.09) < 0.001

        # Avg test = (0.05 + 0.06) / 2 = 0.055
        assert abs(result.avg_test_return - 0.055) < 0.001

    def test_total_test_return(self) -> None:
        """Test compounded total test return."""
        result = WalkForwardResult(
            strategy_name="SMA",
            symbol="AAPL",
            total_windows=2,
            train_pct=0.7,
        )

        # Add windows with 10% and 5% test returns
        for i, test_return in enumerate([0.10, 0.05]):
            window = WalkForwardWindow(
                window_id=i + 1,
                train_start=datetime(2023, 1, 1),
                train_end=datetime(2023, 6, 30),
                test_start=datetime(2023, 7, 1),
                test_end=datetime(2023, 9, 30),
                train_bars=180,
                test_bars=90,
            )
            window.test_result = BacktestResult(
                strategy_name="test",
                symbol="AAPL",
                start_date=datetime(2023, 7, 1),
                end_date=datetime(2023, 9, 30),
                initial_capital=Decimal("100000"),
                final_capital=Decimal(str(100000 * (1 + test_return))),
            )
            result.windows.append(window)

        # Total = (1 + 0.10) * (1 + 0.05) - 1 = 0.155
        assert abs(result.total_test_return - 0.155) < 0.001

    def test_win_rate(self) -> None:
        """Test window win rate."""
        result = WalkForwardResult(
            strategy_name="SMA",
            symbol="AAPL",
            total_windows=4,
            train_pct=0.7,
        )

        # 3 profitable windows, 1 losing
        for i, test_return in enumerate([0.05, 0.03, -0.02, 0.04]):
            window = WalkForwardWindow(
                window_id=i + 1,
                train_start=datetime(2023, 1, 1),
                train_end=datetime(2023, 6, 30),
                test_start=datetime(2023, 7, 1),
                test_end=datetime(2023, 9, 30),
                train_bars=180,
                test_bars=90,
            )
            window.test_result = BacktestResult(
                strategy_name="test",
                symbol="AAPL",
                start_date=datetime(2023, 7, 1),
                end_date=datetime(2023, 9, 30),
                initial_capital=Decimal("100000"),
                final_capital=Decimal(str(100000 * (1 + test_return))),
            )
            result.windows.append(window)

        assert result.win_rate == 0.75  # 3/4

    def test_summary(self) -> None:
        """Test summary generation."""
        result = WalkForwardResult(
            strategy_name="SMA",
            symbol="AAPL",
            total_windows=5,
            train_pct=0.7,
        )

        summary = result.summary()

        assert summary["strategy"] == "SMA"
        assert summary["symbol"] == "AAPL"
        assert summary["total_windows"] == 5
        assert summary["train_pct"] == 0.7


class TestWalkForwardOptimizer:
    """Tests for WalkForwardOptimizer."""

    def test_initialization(self) -> None:
        """Test optimizer initialization."""
        optimizer = WalkForwardOptimizer(
            initial_capital=100000.0,
            train_pct=0.7,
            n_windows=5,
        )

        assert optimizer.initial_capital == 100000.0
        assert optimizer.train_pct == 0.7
        assert optimizer.n_windows == 5

    def test_invalid_train_pct(self) -> None:
        """Test that invalid train_pct raises error."""
        with pytest.raises(ValueError, match="train_pct must be between"):
            WalkForwardOptimizer(train_pct=0.3)

        with pytest.raises(ValueError, match="train_pct must be between"):
            WalkForwardOptimizer(train_pct=0.95)

    def test_invalid_n_windows(self) -> None:
        """Test that invalid n_windows raises error."""
        with pytest.raises(ValueError, match="n_windows must be at least"):
            WalkForwardOptimizer(n_windows=1)

    def test_with_cost_model(self) -> None:
        """Test optimizer with cost model."""
        cost_model = CostModel.retail_investor()
        optimizer = WalkForwardOptimizer(
            initial_capital=100000.0,
            cost_model=cost_model,
        )

        assert optimizer.cost_model == cost_model

    def test_generate_param_combinations(self) -> None:
        """Test parameter combination generation."""
        optimizer = WalkForwardOptimizer()

        param_grid = {
            "fast_period": [5, 10],
            "slow_period": [30, 50],
        }

        combinations = optimizer._generate_param_combinations(param_grid)

        assert len(combinations) == 4
        assert {"fast_period": 5, "slow_period": 30} in combinations
        assert {"fast_period": 5, "slow_period": 50} in combinations
        assert {"fast_period": 10, "slow_period": 30} in combinations
        assert {"fast_period": 10, "slow_period": 50} in combinations

    def test_generate_param_combinations_empty(self) -> None:
        """Test param combinations with empty grid."""
        optimizer = WalkForwardOptimizer()

        combinations = optimizer._generate_param_combinations({})

        assert combinations == [{}]

    @pytest.mark.asyncio
    async def test_run_walkforward(self, sample_data: pd.DataFrame) -> None:
        """Test running walk-forward optimization."""
        from trader.strategies.builtin.sma_crossover import SMACrossover

        optimizer = WalkForwardOptimizer(
            initial_capital=100000.0,
            train_pct=0.7,
            n_windows=3,
            min_train_bars=50,
            min_test_bars=20,
        )

        param_grid = {
            "fast_period": [5, 10],
            "slow_period": [30, 50],
        }

        result = await optimizer.run(
            strategy_class=SMACrossover,
            param_grid=param_grid,
            data=sample_data,
            symbol="TEST",
        )

        assert result.strategy_name == "SMACrossover"
        assert result.symbol == "TEST"
        assert len(result.windows) <= 3
        assert all(w.best_params is not None for w in result.windows)
        assert all(w.train_result is not None for w in result.windows)
        assert all(w.test_result is not None for w in result.windows)

    @pytest.mark.asyncio
    async def test_run_walkforward_with_costs(self, sample_data: pd.DataFrame) -> None:
        """Test walk-forward with transaction costs."""
        from trader.strategies.builtin.sma_crossover import SMACrossover

        cost_model = CostModel.retail_investor()

        optimizer = WalkForwardOptimizer(
            initial_capital=100000.0,
            train_pct=0.7,
            n_windows=2,
            cost_model=cost_model,
            min_train_bars=50,
            min_test_bars=20,
        )

        param_grid = {"fast_period": [10], "slow_period": [50]}

        result = await optimizer.run(
            strategy_class=SMACrossover,
            param_grid=param_grid,
            data=sample_data,
            symbol="TEST",
        )

        # Should complete without error
        assert len(result.windows) > 0

    def test_create_windows(self, sample_data: pd.DataFrame) -> None:
        """Test window creation."""
        optimizer = WalkForwardOptimizer(
            train_pct=0.7,
            n_windows=5,
            min_train_bars=50,
            min_test_bars=10,
        )

        windows = optimizer._create_windows(sample_data)

        # Should create windows
        assert len(windows) > 0

        # Each window should have train and test data
        for train_data, test_data in windows:
            assert len(train_data) >= optimizer.min_train_bars
            assert len(test_data) >= optimizer.min_test_bars

    def test_create_windows_insufficient_data(self) -> None:
        """Test window creation with insufficient data."""
        optimizer = WalkForwardOptimizer(
            train_pct=0.7,
            n_windows=5,
            min_train_bars=100,
            min_test_bars=50,
        )

        # Create small dataset
        small_data = pd.DataFrame(
            {"close": [100] * 50},
            index=pd.date_range("2023-01-01", periods=50),
        )

        with pytest.raises(ValueError, match="Training window too small"):
            optimizer._create_windows(small_data)
