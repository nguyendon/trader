"""Walk-forward optimization for robust strategy validation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

import pandas as pd
from loguru import logger

from trader.engine.backtest import BacktestEngine, BacktestResult, Trade
from trader.engine.costs import CostModel

if TYPE_CHECKING:
    from trader.strategies.base import BaseStrategy


@dataclass
class WalkForwardWindow:
    """A single walk-forward optimization window."""

    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_bars: int
    test_bars: int

    # Results
    best_params: dict[str, Any] | None = None
    train_result: BacktestResult | None = None
    test_result: BacktestResult | None = None

    @property
    def train_return(self) -> float:
        """Training period return."""
        if self.train_result is None:
            return 0.0
        return self.train_result.total_return

    @property
    def test_return(self) -> float:
        """Test (out-of-sample) period return."""
        if self.test_result is None:
            return 0.0
        return self.test_result.total_return

    @property
    def efficiency_ratio(self) -> float:
        """
        Ratio of test to train performance.

        Values close to 1.0 indicate robust parameters.
        Values << 1.0 indicate overfitting to training data.
        """
        if self.train_return == 0:
            return 0.0
        return self.test_return / self.train_return


@dataclass
class WalkForwardResult:
    """Results from walk-forward optimization."""

    strategy_name: str
    symbol: str
    total_windows: int
    train_pct: float  # Percentage of each window used for training
    windows: list[WalkForwardWindow] = field(default_factory=list)

    # Aggregated out-of-sample results
    combined_equity_curve: pd.Series | None = None
    combined_trades: list[Trade] = field(default_factory=list)

    @property
    def avg_train_return(self) -> float:
        """Average return on training data."""
        if not self.windows:
            return 0.0
        return sum(w.train_return for w in self.windows) / len(self.windows)

    @property
    def avg_test_return(self) -> float:
        """Average return on test (out-of-sample) data."""
        if not self.windows:
            return 0.0
        return sum(w.test_return for w in self.windows) / len(self.windows)

    @property
    def total_test_return(self) -> float:
        """Compounded return from all test periods."""
        if not self.windows:
            return 0.0
        cumulative = 1.0
        for w in self.windows:
            cumulative *= (1 + w.test_return)
        return cumulative - 1

    @property
    def avg_efficiency_ratio(self) -> float:
        """
        Average ratio of test to train performance.

        Indicates how well in-sample results predict out-of-sample.
        - 1.0 = Perfect efficiency (test equals train)
        - >1.0 = Test outperforms train (rare, good)
        - <1.0 = Test underperforms train (common, overfitting)

        Values >0.5 are generally acceptable.
        """
        if not self.windows:
            return 0.0
        ratios = [w.efficiency_ratio for w in self.windows if w.train_return != 0]
        if not ratios:
            return 0.0
        return sum(ratios) / len(ratios)

    @property
    def win_rate(self) -> float:
        """Percentage of test windows that were profitable."""
        if not self.windows:
            return 0.0
        winners = sum(1 for w in self.windows if w.test_return > 0)
        return winners / len(self.windows)

    @property
    def num_trades(self) -> int:
        """Total number of trades across all test windows."""
        return len(self.combined_trades)

    @property
    def sharpe_ratio(self) -> float:
        """Sharpe ratio from combined out-of-sample equity curve."""
        if self.combined_equity_curve is None or len(self.combined_equity_curve) < 2:
            return 0.0
        returns = self.combined_equity_curve.pct_change().dropna()
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        return float(returns.mean() / returns.std() * (252**0.5))

    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown from combined out-of-sample equity curve."""
        if self.combined_equity_curve is None or len(self.combined_equity_curve) == 0:
            return 0.0
        peak = self.combined_equity_curve.expanding().max()
        drawdown = (self.combined_equity_curve - peak) / peak
        return abs(float(drawdown.min()))

    def summary(self) -> dict:
        """Get summary statistics."""
        return {
            "strategy": self.strategy_name,
            "symbol": self.symbol,
            "total_windows": self.total_windows,
            "train_pct": self.train_pct,
            "avg_train_return_pct": round(self.avg_train_return * 100, 2),
            "avg_test_return_pct": round(self.avg_test_return * 100, 2),
            "total_test_return_pct": round(self.total_test_return * 100, 2),
            "efficiency_ratio": round(self.avg_efficiency_ratio, 3),
            "window_win_rate_pct": round(self.win_rate * 100, 1),
            "num_trades": self.num_trades,
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "max_drawdown_pct": round(self.max_drawdown * 100, 2),
        }

    def print_summary(self) -> None:
        """Print formatted summary to console."""
        summary = self.summary()
        print("\n" + "=" * 60)
        print(f"Walk-Forward Results: {summary['strategy']}")
        print("=" * 60)
        print(f"Symbol:              {summary['symbol']}")
        print(f"Windows:             {summary['total_windows']}")
        print(f"Train/Test Split:    {int(summary['train_pct']*100)}% / {int((1-summary['train_pct'])*100)}%")
        print("-" * 60)
        print("IN-SAMPLE (Training)")
        print(f"  Avg Return:        {summary['avg_train_return_pct']:+.2f}%")
        print("-" * 60)
        print("OUT-OF-SAMPLE (Test) - What Actually Matters")
        print(f"  Avg Return:        {summary['avg_test_return_pct']:+.2f}%")
        print(f"  Total Return:      {summary['total_test_return_pct']:+.2f}%")
        print(f"  Window Win Rate:   {summary['window_win_rate_pct']:.1f}%")
        print(f"  Total Trades:      {summary['num_trades']}")
        print("-" * 60)
        print("ROBUSTNESS METRICS")
        print(f"  Efficiency Ratio:  {summary['efficiency_ratio']:.3f}")
        efficiency = summary['efficiency_ratio']
        if efficiency >= 0.7:
            grade = "Excellent - strategy appears robust"
        elif efficiency >= 0.5:
            grade = "Good - acceptable level of overfitting"
        elif efficiency >= 0.3:
            grade = "Warning - significant overfitting detected"
        else:
            grade = "Poor - severe overfitting, strategy likely won't work"
        print(f"  Assessment:        {grade}")
        print(f"  Sharpe Ratio:      {summary['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown:      {summary['max_drawdown_pct']:.2f}%")
        print("=" * 60 + "\n")

    def print_windows(self) -> None:
        """Print detailed window-by-window results."""
        print("\nWindow-by-Window Results:")
        print("-" * 80)
        print(f"{'Window':<8} {'Train Period':<25} {'Train %':>10} {'Test %':>10} {'Efficiency':>12}")
        print("-" * 80)
        for w in self.windows:
            train_period = f"{w.train_start.strftime('%Y-%m-%d')} to {w.train_end.strftime('%Y-%m-%d')}"
            print(
                f"{w.window_id:<8} {train_period:<25} "
                f"{w.train_return*100:>+9.2f}% {w.test_return*100:>+9.2f}% "
                f"{w.efficiency_ratio:>11.3f}"
            )
        print("-" * 80)


class WalkForwardOptimizer:
    """
    Walk-forward optimization engine.

    Walk-forward analysis is the gold standard for strategy validation.
    It splits data into multiple train/test windows and:

    1. Optimizes strategy parameters on training data
    2. Tests those parameters on out-of-sample data
    3. Rolls forward and repeats

    This prevents overfitting by ensuring you always test on unseen data.

    Example:
        optimizer = WalkForwardOptimizer(
            initial_capital=100000,
            train_pct=0.7,  # 70% train, 30% test
            n_windows=5,    # 5 walk-forward windows
        )

        result = await optimizer.run(
            strategy_class=SMACrossover,
            param_grid={
                "fast_period": [5, 10, 15, 20],
                "slow_period": [30, 50, 100],
            },
            data=historical_data,
            symbol="AAPL",
        )

        result.print_summary()
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        train_pct: float = 0.7,
        n_windows: int = 5,
        cost_model: CostModel | None = None,
        optimization_metric: str = "sharpe_ratio",
        min_train_bars: int = 100,
        min_test_bars: int = 20,
    ) -> None:
        """
        Initialize walk-forward optimizer.

        Args:
            initial_capital: Starting capital for each window
            train_pct: Fraction of each window for training (0.5-0.9)
            n_windows: Number of walk-forward windows
            cost_model: Transaction cost model (None = zero costs)
            optimization_metric: Metric to optimize ("sharpe_ratio", "total_return", "profit_factor")
            min_train_bars: Minimum bars required for training
            min_test_bars: Minimum bars required for testing
        """
        if not 0.5 <= train_pct <= 0.9:
            raise ValueError("train_pct must be between 0.5 and 0.9")
        if n_windows < 2:
            raise ValueError("n_windows must be at least 2")

        self.initial_capital = initial_capital
        self.train_pct = train_pct
        self.n_windows = n_windows
        self.cost_model = cost_model or CostModel.zero_cost()
        self.optimization_metric = optimization_metric
        self.min_train_bars = min_train_bars
        self.min_test_bars = min_test_bars

    def _create_windows(
        self, data: pd.DataFrame
    ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create train/test windows for walk-forward analysis.

        Uses anchored walk-forward: training window expands over time.
        """
        n_bars = len(data)
        window_size = n_bars // self.n_windows
        train_size = int(window_size * self.train_pct)
        test_size = window_size - train_size

        if train_size < self.min_train_bars:
            raise ValueError(
                f"Training window too small ({train_size} bars). "
                f"Need at least {self.min_train_bars}. "
                f"Reduce n_windows or provide more data."
            )

        if test_size < self.min_test_bars:
            raise ValueError(
                f"Test window too small ({test_size} bars). "
                f"Need at least {self.min_test_bars}. "
                f"Increase train_pct or provide more data."
            )

        windows = []
        for i in range(self.n_windows):
            # Training: from start to end of this window's train period
            train_end = (i + 1) * window_size - test_size
            train_data = data.iloc[:train_end]

            # Test: the remaining portion of this window
            test_start = train_end
            test_end = min((i + 1) * window_size, n_bars)
            test_data = data.iloc[test_start:test_end]

            if len(train_data) >= self.min_train_bars and len(test_data) >= self.min_test_bars:
                windows.append((train_data, test_data))

        return windows

    def _get_metric_value(self, result: BacktestResult) -> float:
        """Extract the optimization metric from backtest result."""
        if self.optimization_metric == "sharpe_ratio":
            return result.sharpe_ratio
        elif self.optimization_metric == "total_return":
            return result.total_return
        elif self.optimization_metric == "profit_factor":
            return result.profit_factor
        elif self.optimization_metric == "win_rate":
            return result.win_rate
        else:
            raise ValueError(f"Unknown optimization metric: {self.optimization_metric}")

    async def run(
        self,
        strategy_class: type[BaseStrategy],
        param_grid: dict[str, list[Any]],
        data: pd.DataFrame,
        symbol: str,
        strategy_factory: Callable[..., BaseStrategy] | None = None,
    ) -> WalkForwardResult:
        """
        Run walk-forward optimization.

        Args:
            strategy_class: Strategy class to optimize
            param_grid: Parameter grid to search, e.g., {"fast_period": [5, 10], "slow_period": [30, 50]}
            data: Full historical data
            symbol: Symbol being traded
            strategy_factory: Optional custom factory to create strategy instances

        Returns:
            WalkForwardResult with combined out-of-sample performance
        """
        logger.info(
            f"Starting walk-forward optimization: {strategy_class.__name__} on {symbol}"
        )
        logger.info(
            f"Windows: {self.n_windows}, Train: {self.train_pct*100:.0f}%, "
            f"Test: {(1-self.train_pct)*100:.0f}%"
        )

        # Create windows
        windows = self._create_windows(data)
        if len(windows) < self.n_windows:
            logger.warning(
                f"Could only create {len(windows)} windows (requested {self.n_windows})"
            )

        # Generate all parameter combinations
        param_combinations = self._generate_param_combinations(param_grid)
        logger.info(f"Parameter combinations to test: {len(param_combinations)}")

        # Results storage
        wf_windows: list[WalkForwardWindow] = []
        all_test_trades: list[Trade] = []
        all_test_equity: list[pd.Series] = []

        # Process each window
        for window_id, (train_data, test_data) in enumerate(windows):
            logger.info(f"Processing window {window_id + 1}/{len(windows)}")

            window = WalkForwardWindow(
                window_id=window_id + 1,
                train_start=train_data.index[0],
                train_end=train_data.index[-1],
                test_start=test_data.index[0],
                test_end=test_data.index[-1],
                train_bars=len(train_data),
                test_bars=len(test_data),
            )

            # Find best parameters on training data
            best_params, best_train_result = await self._optimize_on_train(
                strategy_class=strategy_class,
                param_combinations=param_combinations,
                train_data=train_data,
                symbol=symbol,
                strategy_factory=strategy_factory,
            )

            window.best_params = best_params
            window.train_result = best_train_result

            # Test best parameters on out-of-sample data
            test_result = await self._test_params(
                strategy_class=strategy_class,
                params=best_params,
                data=test_data,
                symbol=symbol,
                strategy_factory=strategy_factory,
            )

            window.test_result = test_result
            wf_windows.append(window)

            # Collect test results
            all_test_trades.extend(test_result.trades)
            if test_result.equity_curve is not None:
                all_test_equity.append(test_result.equity_curve)

            logger.info(
                f"Window {window_id + 1}: Train={window.train_return*100:+.2f}%, "
                f"Test={window.test_return*100:+.2f}%, "
                f"Efficiency={window.efficiency_ratio:.3f}"
            )

        # Combine equity curves
        combined_equity = None
        if all_test_equity:
            combined_equity = pd.concat(all_test_equity)
            # Normalize to starting capital
            combined_equity = combined_equity / combined_equity.iloc[0] * self.initial_capital

        result = WalkForwardResult(
            strategy_name=strategy_class.__name__,
            symbol=symbol,
            total_windows=len(windows),
            train_pct=self.train_pct,
            windows=wf_windows,
            combined_equity_curve=combined_equity,
            combined_trades=all_test_trades,
        )

        logger.info(
            f"Walk-forward complete: Efficiency={result.avg_efficiency_ratio:.3f}, "
            f"OOS Return={result.total_test_return*100:+.2f}%"
        )

        return result

    async def _optimize_on_train(
        self,
        strategy_class: type[BaseStrategy],
        param_combinations: list[dict[str, Any]],
        train_data: pd.DataFrame,
        symbol: str,
        strategy_factory: Callable[..., BaseStrategy] | None = None,
    ) -> tuple[dict[str, Any], BacktestResult]:
        """Find best parameters on training data."""
        best_params: dict[str, Any] = {}
        best_metric = float("-inf")
        best_result: BacktestResult | None = None

        engine = BacktestEngine(
            initial_capital=self.initial_capital,
            cost_model=self.cost_model,
        )

        for params in param_combinations:
            try:
                if strategy_factory:
                    strategy = strategy_factory(**params)
                else:
                    strategy = strategy_class(**params)

                result = await engine.run(strategy, train_data.copy(), symbol)
                metric_value = self._get_metric_value(result)

                if metric_value > best_metric:
                    best_metric = metric_value
                    best_params = params
                    best_result = result

            except Exception as e:
                logger.debug(f"Params {params} failed: {e}")
                continue

        if best_result is None:
            raise ValueError("No valid parameter combination found")

        return best_params, best_result

    async def _test_params(
        self,
        strategy_class: type[BaseStrategy],
        params: dict[str, Any],
        data: pd.DataFrame,
        symbol: str,
        strategy_factory: Callable[..., BaseStrategy] | None = None,
    ) -> BacktestResult:
        """Test specific parameters on data."""
        engine = BacktestEngine(
            initial_capital=self.initial_capital,
            commission=float(self.cost_model.commission_per_trade),
        )

        if strategy_factory:
            strategy = strategy_factory(**params)
        else:
            strategy = strategy_class(**params)

        return await engine.run(strategy, data.copy(), symbol)

    def _generate_param_combinations(
        self, param_grid: dict[str, list[Any]]
    ) -> list[dict[str, Any]]:
        """Generate all combinations of parameters."""
        if not param_grid:
            return [{}]

        keys = list(param_grid.keys())
        values = list(param_grid.values())

        combinations: list[dict[str, Any]] = []
        self._generate_combinations_recursive(keys, values, 0, {}, combinations)
        return combinations

    def _generate_combinations_recursive(
        self,
        keys: list[str],
        values: list[list[Any]],
        index: int,
        current: dict[str, Any],
        results: list[dict[str, Any]],
    ) -> None:
        """Recursively generate parameter combinations."""
        if index == len(keys):
            results.append(current.copy())
            return

        for value in values[index]:
            current[keys[index]] = value
            self._generate_combinations_recursive(keys, values, index + 1, current, results)


async def run_walk_forward(
    strategy_class: type[BaseStrategy],
    param_grid: dict[str, list[Any]],
    data: pd.DataFrame,
    symbol: str,
    n_windows: int = 5,
    train_pct: float = 0.7,
    initial_capital: float = 100_000.0,
    cost_model: CostModel | None = None,
) -> WalkForwardResult:
    """
    Convenience function to run walk-forward optimization.

    Args:
        strategy_class: Strategy class to optimize
        param_grid: Parameter grid to search
        data: Historical OHLCV data
        symbol: Symbol being traded
        n_windows: Number of walk-forward windows
        train_pct: Fraction of each window for training
        initial_capital: Starting capital
        cost_model: Transaction cost model

    Returns:
        WalkForwardResult with out-of-sample performance metrics
    """
    optimizer = WalkForwardOptimizer(
        initial_capital=initial_capital,
        train_pct=train_pct,
        n_windows=n_windows,
        cost_model=cost_model,
    )

    return await optimizer.run(
        strategy_class=strategy_class,
        param_grid=param_grid,
        data=data,
        symbol=symbol,
    )
