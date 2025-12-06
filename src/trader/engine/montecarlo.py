"""Monte Carlo simulation for strategy validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from trader.engine.backtest import BacktestResult


@dataclass
class MonteCarloResult:
    """
    Results from Monte Carlo simulation.

    Monte Carlo simulation randomizes trade order to show the range
    of possible outcomes, not just the single historical path.
    This helps answer: "Was my backtest result skill or luck?"
    """

    original_result: BacktestResult
    n_simulations: int
    simulated_returns: np.ndarray  # Array of final returns from each simulation
    simulated_max_drawdowns: np.ndarray
    simulated_sharpe_ratios: np.ndarray

    # Equity curves from simulations (optional, memory intensive)
    simulated_equity_curves: list[pd.Series] | None = None

    @property
    def original_return(self) -> float:
        """Original backtest return."""
        return self.original_result.total_return

    @property
    def mean_return(self) -> float:
        """Mean return across all simulations."""
        return float(np.mean(self.simulated_returns))

    @property
    def median_return(self) -> float:
        """Median return across all simulations."""
        return float(np.median(self.simulated_returns))

    @property
    def std_return(self) -> float:
        """Standard deviation of returns."""
        return float(np.std(self.simulated_returns))

    @property
    def min_return(self) -> float:
        """Worst case return."""
        return float(np.min(self.simulated_returns))

    @property
    def max_return(self) -> float:
        """Best case return."""
        return float(np.max(self.simulated_returns))

    @property
    def percentile_5(self) -> float:
        """5th percentile return (95% of outcomes are better)."""
        return float(np.percentile(self.simulated_returns, 5))

    @property
    def percentile_25(self) -> float:
        """25th percentile return."""
        return float(np.percentile(self.simulated_returns, 25))

    @property
    def percentile_75(self) -> float:
        """75th percentile return."""
        return float(np.percentile(self.simulated_returns, 75))

    @property
    def percentile_95(self) -> float:
        """95th percentile return (only 5% of outcomes are better)."""
        return float(np.percentile(self.simulated_returns, 95))

    @property
    def probability_of_profit(self) -> float:
        """Probability of positive return (0-1)."""
        return float(np.mean(self.simulated_returns > 0))

    @property
    def probability_of_loss(self) -> float:
        """Probability of negative return (0-1)."""
        return float(np.mean(self.simulated_returns < 0))

    @property
    def mean_max_drawdown(self) -> float:
        """Average maximum drawdown across simulations."""
        return float(np.mean(self.simulated_max_drawdowns))

    @property
    def worst_max_drawdown(self) -> float:
        """Worst maximum drawdown across simulations."""
        return float(np.max(self.simulated_max_drawdowns))

    @property
    def percentile_95_drawdown(self) -> float:
        """95th percentile drawdown (5% of simulations are worse)."""
        return float(np.percentile(self.simulated_max_drawdowns, 95))

    @property
    def mean_sharpe(self) -> float:
        """Average Sharpe ratio across simulations."""
        valid_sharpes = self.simulated_sharpe_ratios[
            np.isfinite(self.simulated_sharpe_ratios)
        ]
        if len(valid_sharpes) == 0:
            return 0.0
        return float(np.mean(valid_sharpes))

    @property
    def original_percentile(self) -> float:
        """
        Percentile rank of original result among simulations.

        If original is at 90th percentile, it means the actual
        sequence of trades was luckier than 90% of random orderings.
        High values (>80%) suggest luck played a role.
        """
        return float(
            np.mean(self.simulated_returns <= self.original_return) * 100
        )

    @property
    def luck_factor(self) -> str:
        """
        Qualitative assessment of luck in original result.

        Based on where original falls in the distribution.
        """
        pct = self.original_percentile
        if pct >= 90:
            return "Very Lucky - Original in top 10%"
        elif pct >= 75:
            return "Somewhat Lucky - Original in top 25%"
        elif pct >= 50:
            return "Average - Original near median"
        elif pct >= 25:
            return "Somewhat Unlucky - Original in bottom 25%"
        else:
            return "Very Unlucky - Original in bottom 10%"

    def summary(self) -> dict:
        """Get summary statistics."""
        return {
            "n_simulations": self.n_simulations,
            "original_return_pct": round(self.original_return * 100, 2),
            "mean_return_pct": round(self.mean_return * 100, 2),
            "median_return_pct": round(self.median_return * 100, 2),
            "std_return_pct": round(self.std_return * 100, 2),
            "min_return_pct": round(self.min_return * 100, 2),
            "max_return_pct": round(self.max_return * 100, 2),
            "percentile_5_pct": round(self.percentile_5 * 100, 2),
            "percentile_95_pct": round(self.percentile_95 * 100, 2),
            "probability_of_profit_pct": round(self.probability_of_profit * 100, 1),
            "mean_max_drawdown_pct": round(self.mean_max_drawdown * 100, 2),
            "worst_max_drawdown_pct": round(self.worst_max_drawdown * 100, 2),
            "percentile_95_drawdown_pct": round(self.percentile_95_drawdown * 100, 2),
            "mean_sharpe": round(self.mean_sharpe, 2),
            "original_percentile": round(self.original_percentile, 1),
            "luck_assessment": self.luck_factor,
        }

    def print_summary(self) -> None:
        """Print formatted summary to console."""
        summary = self.summary()
        print("\n" + "=" * 65)
        print("Monte Carlo Simulation Results")
        print("=" * 65)
        print(f"Simulations:         {summary['n_simulations']:,}")
        print(f"Original Trades:     {len(self.original_result.trades)}")
        print("-" * 65)
        print("RETURN DISTRIBUTION")
        print(f"  Original Return:   {summary['original_return_pct']:+.2f}%")
        print(f"  Mean Return:       {summary['mean_return_pct']:+.2f}%")
        print(f"  Median Return:     {summary['median_return_pct']:+.2f}%")
        print(f"  Std Deviation:     {summary['std_return_pct']:.2f}%")
        print(f"  Range:             {summary['min_return_pct']:+.2f}% to {summary['max_return_pct']:+.2f}%")
        print("-" * 65)
        print("CONFIDENCE INTERVALS")
        print(f"  5th Percentile:    {summary['percentile_5_pct']:+.2f}%  (95% of outcomes better)")
        print(f"  95th Percentile:   {summary['percentile_95_pct']:+.2f}%  (5% of outcomes better)")
        print(f"  Profit Probability: {summary['probability_of_profit_pct']:.1f}%")
        print("-" * 65)
        print("RISK METRICS")
        print(f"  Mean Max Drawdown:  {summary['mean_max_drawdown_pct']:.2f}%")
        print(f"  95th Pctl Drawdown: {summary['percentile_95_drawdown_pct']:.2f}%")
        print(f"  Worst Drawdown:     {summary['worst_max_drawdown_pct']:.2f}%")
        print(f"  Mean Sharpe Ratio:  {summary['mean_sharpe']:.2f}")
        print("-" * 65)
        print("LUCK ANALYSIS")
        print(f"  Original Percentile: {summary['original_percentile']:.1f}%")
        print(f"  Assessment:          {summary['luck_assessment']}")
        print("=" * 65)

        # Interpretation
        if summary['probability_of_profit_pct'] < 50:
            print("\n⚠️  WARNING: Less than 50% probability of profit!")
            print("   Strategy may not have a real edge.")
        elif summary['original_percentile'] > 85:
            print("\n⚠️  CAUTION: Original result was unusually lucky.")
            print("   Expect worse performance going forward.")
        elif summary['percentile_5_pct'] < -10:
            print("\n⚠️  NOTE: 5% chance of losing more than 10%.")
            print("   Consider position sizing and risk limits.")

        print()


class MonteCarloSimulator:
    """
    Monte Carlo simulator for backtest validation.

    Runs multiple simulations by randomizing trade order to show
    the distribution of possible outcomes. This reveals whether
    a good backtest was due to skill (consistent across orderings)
    or luck (highly dependent on specific trade sequence).

    Methods:
    - shuffle_trades: Randomize trade order, recalculate equity curve
    - bootstrap_trades: Resample trades with replacement
    - randomize_returns: Shuffle daily/trade returns

    Example:
        simulator = MonteCarloSimulator(n_simulations=1000)
        result = simulator.run_shuffle(backtest_result)
        result.print_summary()
    """

    def __init__(
        self,
        n_simulations: int = 1000,
        random_seed: int | None = None,
        store_equity_curves: bool = False,
    ) -> None:
        """
        Initialize Monte Carlo simulator.

        Args:
            n_simulations: Number of simulations to run
            random_seed: Random seed for reproducibility
            store_equity_curves: Whether to store all equity curves (memory intensive)
        """
        self.n_simulations = n_simulations
        self.random_seed = random_seed
        self.store_equity_curves = store_equity_curves

        if random_seed is not None:
            np.random.seed(random_seed)

    def run_shuffle(
        self,
        result: BacktestResult,
        initial_capital: float | None = None,
    ) -> MonteCarloResult:
        """
        Run Monte Carlo by shuffling trade order.

        This tests whether the sequence of trades mattered.
        If results vary widely, the original outcome may have been luck.

        Args:
            result: Original backtest result with trades
            initial_capital: Starting capital (uses original if not specified)

        Returns:
            MonteCarloResult with simulation statistics
        """
        if len(result.trades) < 2:
            raise ValueError("Need at least 2 trades for Monte Carlo simulation")

        capital = float(result.initial_capital) if initial_capital is None else initial_capital

        logger.info(
            f"Running Monte Carlo shuffle simulation: "
            f"{self.n_simulations} simulations, {len(result.trades)} trades"
        )

        # Extract trade P&L percentages
        trade_returns = np.array([t.pnl_pct for t in result.trades])

        simulated_returns = np.zeros(self.n_simulations)
        simulated_max_drawdowns = np.zeros(self.n_simulations)
        simulated_sharpe_ratios = np.zeros(self.n_simulations)
        equity_curves: list[pd.Series] | None = [] if self.store_equity_curves else None

        for i in range(self.n_simulations):
            # Shuffle trade returns
            shuffled_returns = np.random.permutation(trade_returns)

            # Calculate equity curve
            equity = self._calculate_equity_curve(capital, shuffled_returns)

            # Calculate metrics
            simulated_returns[i] = (equity[-1] - capital) / capital
            simulated_max_drawdowns[i] = self._calculate_max_drawdown(equity)
            simulated_sharpe_ratios[i] = self._calculate_sharpe(equity)

            if self.store_equity_curves and equity_curves is not None:
                equity_curves.append(pd.Series(equity))

        return MonteCarloResult(
            original_result=result,
            n_simulations=self.n_simulations,
            simulated_returns=simulated_returns,
            simulated_max_drawdowns=simulated_max_drawdowns,
            simulated_sharpe_ratios=simulated_sharpe_ratios,
            simulated_equity_curves=equity_curves,
        )

    def run_bootstrap(
        self,
        result: BacktestResult,
        initial_capital: float | None = None,
        sample_size: int | None = None,
    ) -> MonteCarloResult:
        """
        Run Monte Carlo by bootstrapping (resampling with replacement).

        This creates new trade sequences by randomly sampling from
        the original trades, allowing repeats. Useful for estimating
        confidence intervals.

        Args:
            result: Original backtest result with trades
            initial_capital: Starting capital
            sample_size: Number of trades per simulation (uses original count if not specified)

        Returns:
            MonteCarloResult with simulation statistics
        """
        if len(result.trades) < 2:
            raise ValueError("Need at least 2 trades for Monte Carlo simulation")

        capital = float(result.initial_capital) if initial_capital is None else initial_capital
        n_trades = sample_size if sample_size else len(result.trades)

        logger.info(
            f"Running Monte Carlo bootstrap simulation: "
            f"{self.n_simulations} simulations, {n_trades} trades each"
        )

        trade_returns = np.array([t.pnl_pct for t in result.trades])

        simulated_returns = np.zeros(self.n_simulations)
        simulated_max_drawdowns = np.zeros(self.n_simulations)
        simulated_sharpe_ratios = np.zeros(self.n_simulations)
        equity_curves: list[pd.Series] | None = [] if self.store_equity_curves else None

        for i in range(self.n_simulations):
            # Bootstrap: sample with replacement
            bootstrap_indices = np.random.randint(0, len(trade_returns), n_trades)
            bootstrap_returns = trade_returns[bootstrap_indices]

            # Calculate equity curve
            equity = self._calculate_equity_curve(capital, bootstrap_returns)

            # Calculate metrics
            simulated_returns[i] = (equity[-1] - capital) / capital
            simulated_max_drawdowns[i] = self._calculate_max_drawdown(equity)
            simulated_sharpe_ratios[i] = self._calculate_sharpe(equity)

            if self.store_equity_curves and equity_curves is not None:
                equity_curves.append(pd.Series(equity))

        return MonteCarloResult(
            original_result=result,
            n_simulations=self.n_simulations,
            simulated_returns=simulated_returns,
            simulated_max_drawdowns=simulated_max_drawdowns,
            simulated_sharpe_ratios=simulated_sharpe_ratios,
            simulated_equity_curves=equity_curves,
        )

    def run_block_bootstrap(
        self,
        result: BacktestResult,
        block_size: int = 5,
        initial_capital: float | None = None,
    ) -> MonteCarloResult:
        """
        Run Monte Carlo with block bootstrap.

        Preserves some autocorrelation by resampling blocks of
        consecutive trades rather than individual trades.

        Args:
            result: Original backtest result with trades
            block_size: Size of blocks to resample
            initial_capital: Starting capital

        Returns:
            MonteCarloResult with simulation statistics
        """
        if len(result.trades) < block_size:
            raise ValueError(f"Need at least {block_size} trades for block bootstrap")

        capital = float(result.initial_capital) if initial_capital is None else initial_capital
        n_trades = len(result.trades)
        n_blocks = (n_trades + block_size - 1) // block_size

        logger.info(
            f"Running Monte Carlo block bootstrap: "
            f"{self.n_simulations} simulations, block_size={block_size}"
        )

        trade_returns = np.array([t.pnl_pct for t in result.trades])

        simulated_returns = np.zeros(self.n_simulations)
        simulated_max_drawdowns = np.zeros(self.n_simulations)
        simulated_sharpe_ratios = np.zeros(self.n_simulations)
        equity_curves: list[pd.Series] | None = [] if self.store_equity_curves else None

        for i in range(self.n_simulations):
            # Sample random block starting positions
            block_starts = np.random.randint(
                0, max(1, n_trades - block_size + 1), n_blocks
            )

            # Collect trades from blocks
            block_returns_list: list[float] = []
            for start in block_starts:
                end = min(start + block_size, n_trades)
                block_returns_list.extend(trade_returns[start:end])

            # Trim to original length
            block_returns_arr = np.array(block_returns_list[:n_trades])

            # Calculate equity curve
            equity = self._calculate_equity_curve(capital, block_returns_arr)

            # Calculate metrics
            simulated_returns[i] = (equity[-1] - capital) / capital
            simulated_max_drawdowns[i] = self._calculate_max_drawdown(equity)
            simulated_sharpe_ratios[i] = self._calculate_sharpe(equity)

            if self.store_equity_curves and equity_curves is not None:
                equity_curves.append(pd.Series(equity))

        return MonteCarloResult(
            original_result=result,
            n_simulations=self.n_simulations,
            simulated_returns=simulated_returns,
            simulated_max_drawdowns=simulated_max_drawdowns,
            simulated_sharpe_ratios=simulated_sharpe_ratios,
            simulated_equity_curves=equity_curves,
        )

    def _calculate_equity_curve(
        self,
        initial_capital: float,
        trade_returns: np.ndarray,
    ) -> np.ndarray:
        """Calculate equity curve from trade returns."""
        equity = np.zeros(len(trade_returns) + 1)
        equity[0] = initial_capital

        for i, ret in enumerate(trade_returns):
            equity[i + 1] = equity[i] * (1 + ret)

        return equity

    def _calculate_max_drawdown(self, equity: np.ndarray) -> float:
        """Calculate maximum drawdown from equity curve."""
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        return abs(float(np.min(drawdown)))

    def _calculate_sharpe(self, equity: np.ndarray) -> float:
        """Calculate Sharpe ratio from equity curve."""
        returns = np.diff(equity) / equity[:-1]
        if len(returns) < 2 or np.std(returns) == 0:
            return 0.0
        return float(np.mean(returns) / np.std(returns) * np.sqrt(252))


def run_monte_carlo(
    result: BacktestResult,
    n_simulations: int = 1000,
    method: str = "shuffle",
    random_seed: int | None = None,
) -> MonteCarloResult:
    """
    Convenience function to run Monte Carlo simulation.

    Args:
        result: Backtest result to analyze
        n_simulations: Number of simulations
        method: "shuffle", "bootstrap", or "block_bootstrap"
        random_seed: Random seed for reproducibility

    Returns:
        MonteCarloResult with simulation statistics
    """
    simulator = MonteCarloSimulator(
        n_simulations=n_simulations,
        random_seed=random_seed,
    )

    if method == "shuffle":
        return simulator.run_shuffle(result)
    elif method == "bootstrap":
        return simulator.run_bootstrap(result)
    elif method == "block_bootstrap":
        return simulator.run_block_bootstrap(result)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'shuffle', 'bootstrap', or 'block_bootstrap'")
