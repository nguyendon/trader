"""Risk analytics calculations for portfolio analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from trader.core.models import Position


@dataclass
class RiskMetrics:
    """Portfolio risk metrics."""

    timestamp: datetime
    var_95: float  # Value at Risk at 95% confidence
    var_99: float  # Value at Risk at 99% confidence
    volatility: float  # Annualized volatility
    sharpe_ratio: float  # Sharpe ratio (assuming 0 risk-free rate)
    sortino_ratio: float  # Sortino ratio (downside deviation)
    max_drawdown: float  # Maximum drawdown percentage
    current_drawdown: float  # Current drawdown from peak
    beta: float | None = None  # Beta vs benchmark
    alpha: float | None = None  # Alpha vs benchmark


@dataclass
class CorrelationResult:
    """Correlation matrix result."""

    symbols: list[str]
    matrix: pd.DataFrame
    timestamp: datetime = field(default_factory=lambda: datetime.now())


@dataclass
class PositionRisk:
    """Risk metrics for a single position."""

    symbol: str
    weight: float  # Portfolio weight
    var_contribution: float  # Contribution to portfolio VaR
    beta: float | None
    volatility: float
    current_pnl_pct: float


class PortfolioAnalytics:
    """
    Portfolio risk analytics calculator.

    Provides methods for calculating various risk metrics including:
    - Value at Risk (VaR) using historical and parametric methods
    - Correlation matrix between assets
    - Beta vs benchmark
    - Maximum drawdown
    - Sharpe and Sortino ratios
    """

    # Standard trading days per year for annualization
    TRADING_DAYS_PER_YEAR = 252
    # Risk-free rate (can be updated)
    RISK_FREE_RATE = 0.0

    def __init__(self, risk_free_rate: float = 0.0) -> None:
        """Initialize analytics with optional risk-free rate.

        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
        """
        self.risk_free_rate = risk_free_rate

    def calculate_returns(
        self,
        prices: pd.Series,
        method: str = "log",
    ) -> pd.Series:
        """Calculate returns from price series.

        Args:
            prices: Price series
            method: "log" for log returns, "simple" for simple returns

        Returns:
            Returns series
        """
        if method == "log":
            returns: pd.Series = np.log(prices / prices.shift(1))
        else:
            returns = prices.pct_change()

        return returns.dropna()  # type: ignore[return-value]

    def calculate_var(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95,
        method: str = "historical",
        holding_period: int = 1,
    ) -> float:
        """Calculate Value at Risk (VaR).

        Args:
            returns: Series of returns (daily)
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            method: "historical" or "parametric"
            holding_period: Number of days (for scaling)

        Returns:
            VaR as a positive percentage (potential loss)
        """
        if len(returns) < 2:
            return 0.0

        if method == "historical":
            # Historical VaR: percentile of historical returns
            var = np.percentile(returns, (1 - confidence_level) * 100)
        else:
            # Parametric VaR: assumes normal distribution
            from scipy import stats

            mean = returns.mean()
            std = returns.std()
            z_score = stats.norm.ppf(1 - confidence_level)
            var = mean + z_score * std

        # Scale by holding period (square root of time rule)
        var = var * np.sqrt(holding_period)

        # Return as positive number (loss)
        return abs(float(var))

    def calculate_cvar(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95,
    ) -> float:
        """Calculate Conditional VaR (Expected Shortfall).

        CVaR is the expected loss given that the loss exceeds VaR.

        Args:
            returns: Series of returns
            confidence_level: Confidence level

        Returns:
            CVaR as a positive percentage
        """
        if len(returns) < 2:
            return 0.0

        var = self.calculate_var(returns, confidence_level, method="historical")
        # CVaR is mean of returns below VaR threshold
        threshold = -var
        tail_returns = returns[returns <= threshold]

        if len(tail_returns) == 0:
            return var

        return abs(float(tail_returns.mean()))

    def calculate_volatility(
        self,
        returns: pd.Series,
        annualize: bool = True,
    ) -> float:
        """Calculate volatility (standard deviation of returns).

        Args:
            returns: Series of returns
            annualize: Whether to annualize (multiply by sqrt(252))

        Returns:
            Volatility as decimal (e.g., 0.20 for 20%)
        """
        if len(returns) < 2:
            return 0.0

        vol = returns.std()

        if annualize:
            vol = vol * np.sqrt(self.TRADING_DAYS_PER_YEAR)

        return float(vol)

    def calculate_sharpe_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float | None = None,
    ) -> float:
        """Calculate Sharpe ratio.

        Args:
            returns: Series of daily returns
            risk_free_rate: Annual risk-free rate (uses instance default if None)

        Returns:
            Annualized Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0

        rf = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        daily_rf = rf / self.TRADING_DAYS_PER_YEAR

        excess_returns = returns - daily_rf
        mean_excess = excess_returns.mean()
        std = returns.std()

        if std == 0:
            return 0.0

        # Annualize
        sharpe = (mean_excess / std) * np.sqrt(self.TRADING_DAYS_PER_YEAR)

        return float(sharpe)

    def calculate_sortino_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float | None = None,
        target_return: float = 0.0,
    ) -> float:
        """Calculate Sortino ratio (using downside deviation).

        Args:
            returns: Series of daily returns
            risk_free_rate: Annual risk-free rate
            target_return: Target return for downside calculation

        Returns:
            Annualized Sortino ratio
        """
        if len(returns) < 2:
            return 0.0

        rf = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        daily_rf = rf / self.TRADING_DAYS_PER_YEAR

        excess_returns = returns - daily_rf
        mean_excess = excess_returns.mean()

        # Downside deviation: std of returns below target
        downside_returns = returns[returns < target_return]
        if len(downside_returns) < 2:
            return 0.0 if mean_excess <= 0 else float("inf")

        downside_std = downside_returns.std()

        if downside_std == 0:
            return 0.0 if mean_excess <= 0 else float("inf")

        # Annualize
        sortino = (mean_excess / downside_std) * np.sqrt(self.TRADING_DAYS_PER_YEAR)

        return float(sortino)

    def calculate_max_drawdown(
        self,
        equity_curve: pd.Series,
    ) -> tuple[float, datetime | None, datetime | None]:
        """Calculate maximum drawdown from equity curve.

        Args:
            equity_curve: Series of portfolio values over time

        Returns:
            Tuple of (max_drawdown_pct, peak_date, trough_date)
        """
        if len(equity_curve) < 2:
            return 0.0, None, None

        # Calculate running maximum
        running_max = equity_curve.expanding().max()

        # Calculate drawdown at each point
        drawdowns = (equity_curve - running_max) / running_max

        # Find max drawdown
        max_dd = drawdowns.min()
        max_dd_idx = drawdowns.idxmin()

        # Find peak (most recent high before max drawdown)
        peak_idx = equity_curve.loc[:max_dd_idx].idxmax()  # type: ignore[misc]

        return (
            abs(float(max_dd)),  # type: ignore[arg-type]
            peak_idx if isinstance(peak_idx, datetime) else None,
            max_dd_idx if isinstance(max_dd_idx, datetime) else None,
        )

    def calculate_current_drawdown(
        self,
        equity_curve: pd.Series,
    ) -> float:
        """Calculate current drawdown from peak.

        Args:
            equity_curve: Series of portfolio values

        Returns:
            Current drawdown as positive percentage
        """
        if len(equity_curve) < 1:
            return 0.0

        peak = equity_curve.max()
        current = equity_curve.iloc[-1]

        if peak == 0:
            return 0.0

        drawdown = (peak - current) / peak
        return abs(float(drawdown))

    def calculate_beta(
        self,
        asset_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> float:
        """Calculate beta of asset relative to benchmark.

        Beta = Cov(asset, benchmark) / Var(benchmark)

        Args:
            asset_returns: Returns of the asset
            benchmark_returns: Returns of the benchmark (e.g., SPY)

        Returns:
            Beta coefficient
        """
        # Align the series
        aligned = pd.concat([asset_returns, benchmark_returns], axis=1).dropna()
        if len(aligned) < 2:
            return 1.0

        asset_ret = aligned.iloc[:, 0]
        bench_ret = aligned.iloc[:, 1]

        covariance = asset_ret.cov(bench_ret)
        variance = bench_ret.var()

        if variance == 0:
            return 1.0

        return float(covariance / variance)  # type: ignore[arg-type]

    def calculate_alpha(
        self,
        asset_returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_free_rate: float | None = None,
    ) -> float:
        """Calculate Jensen's alpha.

        Alpha = Asset Return - [Rf + Beta * (Benchmark Return - Rf)]

        Args:
            asset_returns: Returns of the asset
            benchmark_returns: Returns of the benchmark
            risk_free_rate: Annual risk-free rate

        Returns:
            Annualized alpha
        """
        rf = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        daily_rf = rf / self.TRADING_DAYS_PER_YEAR

        beta = self.calculate_beta(asset_returns, benchmark_returns)

        asset_mean = asset_returns.mean()
        bench_mean = benchmark_returns.mean()

        # Daily alpha
        daily_alpha = asset_mean - (daily_rf + beta * (bench_mean - daily_rf))

        # Annualize
        return float(daily_alpha * self.TRADING_DAYS_PER_YEAR)

    def calculate_correlation_matrix(
        self,
        price_data: dict[str, pd.Series],
    ) -> CorrelationResult:
        """Calculate correlation matrix between multiple assets.

        Args:
            price_data: Dict of symbol -> price series

        Returns:
            CorrelationResult with matrix and metadata
        """
        if not price_data:
            return CorrelationResult(symbols=[], matrix=pd.DataFrame())

        # Convert prices to returns
        returns_dict = {}
        for symbol, prices in price_data.items():
            returns_dict[symbol] = self.calculate_returns(prices, method="simple")

        # Create DataFrame and calculate correlation
        returns_df = pd.DataFrame(returns_dict)
        corr_matrix = returns_df.corr()

        return CorrelationResult(
            symbols=list(price_data.keys()),
            matrix=corr_matrix,
        )

    def calculate_portfolio_var(
        self,
        weights: dict[str, float],
        returns_data: dict[str, pd.Series],
        confidence_level: float = 0.95,
    ) -> float:
        """Calculate portfolio VaR considering correlations.

        Uses variance-covariance method.

        Args:
            weights: Dict of symbol -> portfolio weight
            returns_data: Dict of symbol -> returns series
            confidence_level: Confidence level

        Returns:
            Portfolio VaR as positive percentage
        """
        from scipy import stats

        symbols = list(weights.keys())
        n = len(symbols)

        if n == 0:
            return 0.0

        # Build returns DataFrame
        returns_df = pd.DataFrame({s: returns_data.get(s, pd.Series()) for s in symbols})
        returns_df = returns_df.dropna()

        if len(returns_df) < 2:
            return 0.0

        # Calculate covariance matrix
        cov_matrix = returns_df.cov()

        # Weight vector
        w = np.array([weights[s] for s in symbols])

        # Portfolio variance
        port_var = np.dot(w.T, np.dot(cov_matrix, w))
        port_std = np.sqrt(port_var)

        # VaR using normal distribution
        z_score = stats.norm.ppf(1 - confidence_level)
        var = abs(z_score * port_std)

        return float(var)

    def calculate_risk_metrics(
        self,
        equity_curve: pd.Series,
        benchmark_returns: pd.Series | None = None,
    ) -> RiskMetrics:
        """Calculate comprehensive risk metrics from equity curve.

        Args:
            equity_curve: Series of portfolio values over time
            benchmark_returns: Optional benchmark returns for beta/alpha

        Returns:
            RiskMetrics dataclass with all calculated metrics
        """
        returns = self.calculate_returns(equity_curve, method="simple")

        var_95 = self.calculate_var(returns, 0.95)
        var_99 = self.calculate_var(returns, 0.99)
        volatility = self.calculate_volatility(returns)
        sharpe = self.calculate_sharpe_ratio(returns)
        sortino = self.calculate_sortino_ratio(returns)
        max_dd, _, _ = self.calculate_max_drawdown(equity_curve)
        current_dd = self.calculate_current_drawdown(equity_curve)

        beta = None
        alpha = None
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            beta = self.calculate_beta(returns, benchmark_returns)
            alpha = self.calculate_alpha(returns, benchmark_returns)

        return RiskMetrics(
            timestamp=datetime.now(),
            var_95=var_95,
            var_99=var_99,
            volatility=volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            current_drawdown=current_dd,
            beta=beta,
            alpha=alpha,
        )

    def analyze_positions(
        self,
        positions: list[Position],
        returns_data: dict[str, pd.Series],
        benchmark_returns: pd.Series | None = None,
    ) -> list[PositionRisk]:
        """Analyze risk for each position.

        Args:
            positions: List of current positions
            returns_data: Dict of symbol -> returns series
            benchmark_returns: Optional benchmark for beta calculation

        Returns:
            List of PositionRisk for each position
        """
        results: list[PositionRisk] = []

        # Calculate total portfolio value
        # market_value already includes quantity, so don't multiply again
        total_value = sum(
            float(p.market_value) if p.market_value else float(p.current_price or 0) * p.quantity
            for p in positions
        )

        if total_value == 0:
            return results

        for pos in positions:
            pos_value = float(pos.market_value) if pos.market_value else float(pos.current_price or 0) * pos.quantity
            weight = pos_value / total_value if total_value > 0 else 0

            returns = returns_data.get(pos.symbol, pd.Series())
            vol = self.calculate_volatility(returns) if len(returns) > 1 else 0.0

            # Calculate VaR contribution (simplified)
            var_contrib = weight * self.calculate_var(returns) if len(returns) > 1 else 0.0

            beta = None
            if benchmark_returns is not None and len(returns) > 1:
                beta = self.calculate_beta(returns, benchmark_returns)

            current_pnl_pct = float(pos.unrealized_pnl_pct or 0)

            results.append(
                PositionRisk(
                    symbol=pos.symbol,
                    weight=weight,
                    var_contribution=var_contrib,
                    beta=beta,
                    volatility=vol,
                    current_pnl_pct=current_pnl_pct,
                )
            )

        return results


def calculate_sector_exposure(
    positions: list[Position],
    sector_map: dict[str, str],
) -> dict[str, float]:
    """Calculate portfolio exposure by sector.

    Args:
        positions: List of positions
        sector_map: Dict of symbol -> sector name

    Returns:
        Dict of sector -> weight
    """
    total_value = sum(
        float(p.market_value or (p.current_price or 0) * p.quantity)
        for p in positions
    )

    if total_value == 0:
        return {}

    sector_exposure: dict[str, float] = {}

    for pos in positions:
        pos_value = float(pos.market_value or (pos.current_price or 0) * pos.quantity)
        sector = sector_map.get(pos.symbol, "Unknown")
        sector_exposure[sector] = sector_exposure.get(sector, 0) + pos_value / total_value

    return sector_exposure


def calculate_concentration_risk(
    positions: list[Position],
    threshold: float = 0.20,
) -> list[tuple[str, float]]:
    """Identify concentrated positions above threshold.

    Args:
        positions: List of positions
        threshold: Weight threshold to flag (e.g., 0.20 = 20%)

    Returns:
        List of (symbol, weight) tuples for concentrated positions
    """
    total_value = sum(
        float(p.market_value or (p.current_price or 0) * p.quantity)
        for p in positions
    )

    if total_value == 0:
        return []

    concentrated = []
    for pos in positions:
        pos_value = float(pos.market_value or (pos.current_price or 0) * pos.quantity)
        weight = pos_value / total_value
        if weight >= threshold:
            concentrated.append((pos.symbol, weight))

    return sorted(concentrated, key=lambda x: -x[1])
