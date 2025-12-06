"""Tests for Monte Carlo simulation."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from trader.core.models import OrderSide
from trader.engine.backtest import BacktestResult, Trade
from trader.engine.montecarlo import (
    MonteCarloResult,
    MonteCarloSimulator,
    run_monte_carlo,
)


@pytest.fixture
def sample_trades() -> list[Trade]:
    """Create sample trades for testing."""
    return [
        Trade(
            symbol="AAPL",
            entry_time=datetime(2023, 1, 1),
            exit_time=datetime(2023, 1, 15),
            side=OrderSide.BUY,
            quantity=100,
            entry_price=Decimal("150.00"),
            exit_price=Decimal("155.00"),
            pnl=Decimal("500.00"),
            pnl_pct=0.0333,
            reason_entry="Signal",
            reason_exit="Signal",
        ),
        Trade(
            symbol="AAPL",
            entry_time=datetime(2023, 2, 1),
            exit_time=datetime(2023, 2, 15),
            side=OrderSide.BUY,
            quantity=100,
            entry_price=Decimal("155.00"),
            exit_price=Decimal("150.00"),
            pnl=Decimal("-500.00"),
            pnl_pct=-0.0323,
            reason_entry="Signal",
            reason_exit="Signal",
        ),
        Trade(
            symbol="AAPL",
            entry_time=datetime(2023, 3, 1),
            exit_time=datetime(2023, 3, 15),
            side=OrderSide.BUY,
            quantity=100,
            entry_price=Decimal("150.00"),
            exit_price=Decimal("160.00"),
            pnl=Decimal("1000.00"),
            pnl_pct=0.0667,
            reason_entry="Signal",
            reason_exit="Signal",
        ),
        Trade(
            symbol="AAPL",
            entry_time=datetime(2023, 4, 1),
            exit_time=datetime(2023, 4, 15),
            side=OrderSide.BUY,
            quantity=100,
            entry_price=Decimal("160.00"),
            exit_price=Decimal("158.00"),
            pnl=Decimal("-200.00"),
            pnl_pct=-0.0125,
            reason_entry="Signal",
            reason_exit="Signal",
        ),
        Trade(
            symbol="AAPL",
            entry_time=datetime(2023, 5, 1),
            exit_time=datetime(2023, 5, 15),
            side=OrderSide.BUY,
            quantity=100,
            entry_price=Decimal("158.00"),
            exit_price=Decimal("165.00"),
            pnl=Decimal("700.00"),
            pnl_pct=0.0443,
            reason_entry="Signal",
            reason_exit="Signal",
        ),
    ]


@pytest.fixture
def sample_backtest_result(sample_trades: list[Trade]) -> BacktestResult:
    """Create sample backtest result."""
    # Create equity curve
    equity_values = [100000, 103333, 100000, 106667, 105333, 110000]
    dates = pd.date_range("2023-01-01", periods=6, freq="ME")
    equity_curve = pd.Series(equity_values, index=dates)

    return BacktestResult(
        strategy_name="TestStrategy",
        symbol="AAPL",
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 5, 15),
        initial_capital=Decimal("100000"),
        final_capital=Decimal("110000"),
        trades=sample_trades,
        equity_curve=equity_curve,
    )


class TestMonteCarloResult:
    """Tests for MonteCarloResult."""

    def test_basic_properties(self, sample_backtest_result: BacktestResult) -> None:
        """Test basic properties."""
        simulated_returns = np.array([0.05, 0.10, -0.02, 0.08, 0.15])
        simulated_drawdowns = np.array([0.05, 0.08, 0.10, 0.06, 0.04])
        simulated_sharpes = np.array([1.0, 1.5, 0.5, 1.2, 2.0])

        result = MonteCarloResult(
            original_result=sample_backtest_result,
            n_simulations=5,
            simulated_returns=simulated_returns,
            simulated_max_drawdowns=simulated_drawdowns,
            simulated_sharpe_ratios=simulated_sharpes,
        )

        assert result.n_simulations == 5
        assert result.original_return == 0.10  # 100k -> 110k = 10%

    def test_mean_return(self, sample_backtest_result: BacktestResult) -> None:
        """Test mean return calculation."""
        simulated_returns = np.array([0.10, 0.20, 0.30])

        result = MonteCarloResult(
            original_result=sample_backtest_result,
            n_simulations=3,
            simulated_returns=simulated_returns,
            simulated_max_drawdowns=np.zeros(3),
            simulated_sharpe_ratios=np.zeros(3),
        )

        assert abs(result.mean_return - 0.20) < 0.0001

    def test_median_return(self, sample_backtest_result: BacktestResult) -> None:
        """Test median return calculation."""
        simulated_returns = np.array([0.10, 0.20, 0.30, 0.40, 0.50])

        result = MonteCarloResult(
            original_result=sample_backtest_result,
            n_simulations=5,
            simulated_returns=simulated_returns,
            simulated_max_drawdowns=np.zeros(5),
            simulated_sharpe_ratios=np.zeros(5),
        )

        assert result.median_return == 0.30

    def test_percentiles(self, sample_backtest_result: BacktestResult) -> None:
        """Test percentile calculations."""
        # 100 values from 0 to 0.99
        simulated_returns = np.linspace(0, 0.99, 100)

        result = MonteCarloResult(
            original_result=sample_backtest_result,
            n_simulations=100,
            simulated_returns=simulated_returns,
            simulated_max_drawdowns=np.zeros(100),
            simulated_sharpe_ratios=np.zeros(100),
        )

        assert abs(result.percentile_5 - 0.05) < 0.02
        assert abs(result.percentile_95 - 0.94) < 0.02

    def test_probability_of_profit(self, sample_backtest_result: BacktestResult) -> None:
        """Test probability of profit calculation."""
        # 7 positive, 3 negative
        simulated_returns = np.array([0.1, 0.2, -0.1, 0.3, -0.2, 0.4, 0.5, -0.3, 0.6, 0.7])

        result = MonteCarloResult(
            original_result=sample_backtest_result,
            n_simulations=10,
            simulated_returns=simulated_returns,
            simulated_max_drawdowns=np.zeros(10),
            simulated_sharpe_ratios=np.zeros(10),
        )

        assert result.probability_of_profit == 0.7

    def test_original_percentile(self, sample_backtest_result: BacktestResult) -> None:
        """Test original result percentile calculation."""
        # Original return is 0.10 (10%)
        # Returns below 0.10: 0.05, 0.08, 0.09 = 3 out of 10
        simulated_returns = np.array([0.05, 0.08, 0.09, 0.11, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25])

        result = MonteCarloResult(
            original_result=sample_backtest_result,
            n_simulations=10,
            simulated_returns=simulated_returns,
            simulated_max_drawdowns=np.zeros(10),
            simulated_sharpe_ratios=np.zeros(10),
        )

        # 3 values <= 0.10 out of 10 = 30%
        assert result.original_percentile == 30.0

    def test_luck_factor_very_lucky(self, sample_backtest_result: BacktestResult) -> None:
        """Test luck assessment for very lucky result."""
        # All simulations worse than original (0.10)
        simulated_returns = np.array([0.01, 0.02, 0.03, 0.04, 0.05])

        result = MonteCarloResult(
            original_result=sample_backtest_result,
            n_simulations=5,
            simulated_returns=simulated_returns,
            simulated_max_drawdowns=np.zeros(5),
            simulated_sharpe_ratios=np.zeros(5),
        )

        assert "Very Lucky" in result.luck_factor

    def test_luck_factor_unlucky(self, sample_backtest_result: BacktestResult) -> None:
        """Test luck assessment for unlucky result."""
        # All simulations better than original (0.10)
        simulated_returns = np.array([0.15, 0.20, 0.25, 0.30, 0.35])

        result = MonteCarloResult(
            original_result=sample_backtest_result,
            n_simulations=5,
            simulated_returns=simulated_returns,
            simulated_max_drawdowns=np.zeros(5),
            simulated_sharpe_ratios=np.zeros(5),
        )

        assert "Unlucky" in result.luck_factor

    def test_summary(self, sample_backtest_result: BacktestResult) -> None:
        """Test summary generation."""
        simulated_returns = np.array([0.05, 0.10, 0.15])
        simulated_drawdowns = np.array([0.05, 0.08, 0.10])
        simulated_sharpes = np.array([1.0, 1.5, 2.0])

        result = MonteCarloResult(
            original_result=sample_backtest_result,
            n_simulations=3,
            simulated_returns=simulated_returns,
            simulated_max_drawdowns=simulated_drawdowns,
            simulated_sharpe_ratios=simulated_sharpes,
        )

        summary = result.summary()

        assert summary["n_simulations"] == 3
        assert "original_return_pct" in summary
        assert "mean_return_pct" in summary
        assert "probability_of_profit_pct" in summary
        assert "luck_assessment" in summary


class TestMonteCarloSimulator:
    """Tests for MonteCarloSimulator."""

    def test_initialization(self) -> None:
        """Test simulator initialization."""
        simulator = MonteCarloSimulator(
            n_simulations=500,
            random_seed=42,
        )

        assert simulator.n_simulations == 500
        assert simulator.random_seed == 42

    def test_shuffle_simulation(self, sample_backtest_result: BacktestResult) -> None:
        """Test shuffle simulation."""
        simulator = MonteCarloSimulator(
            n_simulations=100,
            random_seed=42,
        )

        result = simulator.run_shuffle(sample_backtest_result)

        assert result.n_simulations == 100
        assert len(result.simulated_returns) == 100
        assert len(result.simulated_max_drawdowns) == 100
        assert len(result.simulated_sharpe_ratios) == 100

    def test_bootstrap_simulation(self, sample_backtest_result: BacktestResult) -> None:
        """Test bootstrap simulation."""
        simulator = MonteCarloSimulator(
            n_simulations=100,
            random_seed=42,
        )

        result = simulator.run_bootstrap(sample_backtest_result)

        assert result.n_simulations == 100
        assert len(result.simulated_returns) == 100

    def test_block_bootstrap_simulation(self, sample_backtest_result: BacktestResult) -> None:
        """Test block bootstrap simulation."""
        simulator = MonteCarloSimulator(
            n_simulations=100,
            random_seed=42,
        )

        result = simulator.run_block_bootstrap(sample_backtest_result, block_size=2)

        assert result.n_simulations == 100
        assert len(result.simulated_returns) == 100

    def test_reproducibility_with_seed(self, sample_backtest_result: BacktestResult) -> None:
        """Test that results are reproducible with same seed."""
        simulator1 = MonteCarloSimulator(n_simulations=50, random_seed=42)
        simulator2 = MonteCarloSimulator(n_simulations=50, random_seed=42)

        result1 = simulator1.run_shuffle(sample_backtest_result)
        result2 = simulator2.run_shuffle(sample_backtest_result)

        np.testing.assert_array_almost_equal(
            result1.simulated_returns,
            result2.simulated_returns,
        )

    def test_different_results_without_seed(self, sample_backtest_result: BacktestResult) -> None:
        """Test that results differ without fixed seed."""
        simulator1 = MonteCarloSimulator(n_simulations=50)
        simulator2 = MonteCarloSimulator(n_simulations=50)

        result1 = simulator1.run_shuffle(sample_backtest_result)
        result2 = simulator2.run_shuffle(sample_backtest_result)

        # Very unlikely to be exactly equal without same seed
        assert not np.array_equal(
            result1.simulated_returns,
            result2.simulated_returns,
        )

    def test_store_equity_curves(self, sample_backtest_result: BacktestResult) -> None:
        """Test storing equity curves."""
        simulator = MonteCarloSimulator(
            n_simulations=10,
            random_seed=42,
            store_equity_curves=True,
        )

        result = simulator.run_shuffle(sample_backtest_result)

        assert result.simulated_equity_curves is not None
        assert len(result.simulated_equity_curves) == 10

    def test_insufficient_trades_error(self) -> None:
        """Test error with too few trades."""
        result = BacktestResult(
            strategy_name="Test",
            symbol="AAPL",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_capital=Decimal("100000"),
            final_capital=Decimal("105000"),
            trades=[
                Trade(
                    symbol="AAPL",
                    entry_time=datetime(2023, 1, 1),
                    exit_time=datetime(2023, 1, 15),
                    side=OrderSide.BUY,
                    quantity=100,
                    entry_price=Decimal("150.00"),
                    exit_price=Decimal("155.00"),
                    pnl=Decimal("500.00"),
                    pnl_pct=0.0333,
                )
            ],
        )

        simulator = MonteCarloSimulator(n_simulations=100)

        with pytest.raises(ValueError, match="at least 2 trades"):
            simulator.run_shuffle(result)

    def test_calculate_equity_curve(self) -> None:
        """Test equity curve calculation."""
        simulator = MonteCarloSimulator()

        trade_returns = np.array([0.10, -0.05, 0.08])
        equity = simulator._calculate_equity_curve(100000, trade_returns)

        assert len(equity) == 4
        assert equity[0] == 100000
        assert abs(equity[1] - 110000) < 0.01  # +10%
        assert abs(equity[2] - 104500) < 0.01  # -5%
        assert abs(equity[3] - 112860) < 0.01  # +8%

    def test_calculate_max_drawdown(self) -> None:
        """Test max drawdown calculation."""
        simulator = MonteCarloSimulator()

        # Peak at 110, drops to 100, then to 90
        equity = np.array([100, 110, 105, 100, 95, 90, 95])
        max_dd = simulator._calculate_max_drawdown(equity)

        # Max drawdown is from 110 to 90 = 18.18%
        assert abs(max_dd - 0.1818) < 0.01


class TestRunMonteCarlo:
    """Tests for convenience function."""

    def test_run_shuffle(self, sample_backtest_result: BacktestResult) -> None:
        """Test run_monte_carlo with shuffle method."""
        result = run_monte_carlo(
            sample_backtest_result,
            n_simulations=50,
            method="shuffle",
            random_seed=42,
        )

        assert result.n_simulations == 50

    def test_run_bootstrap(self, sample_backtest_result: BacktestResult) -> None:
        """Test run_monte_carlo with bootstrap method."""
        result = run_monte_carlo(
            sample_backtest_result,
            n_simulations=50,
            method="bootstrap",
            random_seed=42,
        )

        assert result.n_simulations == 50

    def test_run_block_bootstrap(self, sample_backtest_result: BacktestResult) -> None:
        """Test run_monte_carlo with block_bootstrap method."""
        result = run_monte_carlo(
            sample_backtest_result,
            n_simulations=50,
            method="block_bootstrap",
            random_seed=42,
        )

        assert result.n_simulations == 50

    def test_invalid_method(self, sample_backtest_result: BacktestResult) -> None:
        """Test error with invalid method."""
        with pytest.raises(ValueError, match="Unknown method"):
            run_monte_carlo(
                sample_backtest_result,
                method="invalid",
            )
