"""Tests for risk analytics module."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from trader.core.models import Position
from trader.risk.analytics import (
    CorrelationResult,
    PortfolioAnalytics,
    PositionRisk,
    RiskMetrics,
    calculate_concentration_risk,
    calculate_sector_exposure,
)


@pytest.fixture
def sample_prices() -> pd.Series:
    """Create sample price series with known characteristics."""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=252, freq="D")
    # Random walk with drift
    returns = np.random.randn(252) * 0.02 + 0.0003  # ~2% daily vol, slight positive drift
    prices = 100 * np.cumprod(1 + returns)
    return pd.Series(prices, index=dates)


@pytest.fixture
def sample_returns() -> pd.Series:
    """Create sample returns series."""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=252, freq="D")
    returns = np.random.randn(252) * 0.02  # ~2% daily volatility
    return pd.Series(returns, index=dates)


@pytest.fixture
def equity_curve() -> pd.Series:
    """Create sample equity curve with drawdown."""
    dates = pd.date_range(end=datetime.now(), periods=100, freq="D")
    # Start at 100k, rise to 120k, drop to 95k, recover to 110k
    values = [100000]
    for i in range(99):
        if i < 30:
            values.append(values[-1] * 1.006)  # Rising
        elif i < 50:
            values.append(values[-1] * 0.988)  # Falling
        else:
            values.append(values[-1] * 1.003)  # Recovery
    return pd.Series(values, index=dates)


@pytest.fixture
def analytics() -> PortfolioAnalytics:
    """Create analytics instance."""
    return PortfolioAnalytics()


class TestPortfolioAnalytics:
    """Tests for PortfolioAnalytics class."""

    def test_initialization(self, analytics: PortfolioAnalytics) -> None:
        """Test analytics initialization."""
        assert analytics.risk_free_rate == 0.0
        assert analytics.TRADING_DAYS_PER_YEAR == 252

    def test_initialization_with_risk_free_rate(self) -> None:
        """Test analytics with custom risk-free rate."""
        analytics = PortfolioAnalytics(risk_free_rate=0.05)
        assert analytics.risk_free_rate == 0.05

    def test_calculate_returns_simple(
        self, analytics: PortfolioAnalytics, sample_prices: pd.Series
    ) -> None:
        """Test simple returns calculation."""
        returns = analytics.calculate_returns(sample_prices, method="simple")

        # Should have one less element (first return is NaN and dropped)
        assert len(returns) == len(sample_prices) - 1
        # Should be close to expected daily returns
        assert returns.mean() < 0.1  # Reasonable daily return

    def test_calculate_returns_log(
        self, analytics: PortfolioAnalytics, sample_prices: pd.Series
    ) -> None:
        """Test log returns calculation."""
        returns = analytics.calculate_returns(sample_prices, method="log")

        assert len(returns) == len(sample_prices) - 1
        # Log returns should be finite and reasonable
        assert returns.isna().sum() == 0
        assert abs(returns.mean()) < 0.1  # Daily return should be small

    def test_calculate_var_historical(
        self, analytics: PortfolioAnalytics, sample_returns: pd.Series
    ) -> None:
        """Test historical VaR calculation."""
        var_95 = analytics.calculate_var(sample_returns, 0.95, method="historical")

        # VaR should be positive (represents potential loss)
        assert var_95 > 0
        # 95% VaR should be less than 99% VaR
        var_99 = analytics.calculate_var(sample_returns, 0.99, method="historical")
        assert var_99 > var_95

    def test_calculate_var_parametric(
        self, analytics: PortfolioAnalytics, sample_returns: pd.Series
    ) -> None:
        """Test parametric VaR calculation."""
        pytest.importorskip("scipy")

        var_95 = analytics.calculate_var(sample_returns, 0.95, method="parametric")

        assert var_95 > 0
        # Should be similar to historical for normal-ish data
        var_hist = analytics.calculate_var(sample_returns, 0.95, method="historical")
        assert abs(var_95 - var_hist) < 0.05  # Within 5%

    def test_calculate_var_empty_returns(self, analytics: PortfolioAnalytics) -> None:
        """Test VaR with empty returns."""
        empty_returns = pd.Series([], dtype=float)
        var = analytics.calculate_var(empty_returns)
        assert var == 0.0

    def test_calculate_cvar(
        self, analytics: PortfolioAnalytics, sample_returns: pd.Series
    ) -> None:
        """Test CVaR (Expected Shortfall) calculation."""
        cvar = analytics.calculate_cvar(sample_returns, 0.95)
        var = analytics.calculate_var(sample_returns, 0.95)

        # CVaR should be >= VaR (it's the average of losses beyond VaR)
        assert cvar >= var

    def test_calculate_volatility(
        self, analytics: PortfolioAnalytics, sample_returns: pd.Series
    ) -> None:
        """Test volatility calculation."""
        vol_annual = analytics.calculate_volatility(sample_returns, annualize=True)
        vol_daily = analytics.calculate_volatility(sample_returns, annualize=False)

        # Annualized should be sqrt(252) times daily
        expected_annual = vol_daily * np.sqrt(252)
        np.testing.assert_allclose(vol_annual, expected_annual, rtol=0.01)

    def test_calculate_volatility_empty(self, analytics: PortfolioAnalytics) -> None:
        """Test volatility with empty returns."""
        empty_returns = pd.Series([], dtype=float)
        vol = analytics.calculate_volatility(empty_returns)
        assert vol == 0.0

    def test_calculate_sharpe_ratio(
        self, analytics: PortfolioAnalytics, sample_returns: pd.Series
    ) -> None:
        """Test Sharpe ratio calculation."""
        sharpe = analytics.calculate_sharpe_ratio(sample_returns)

        # Should be a reasonable value for random returns
        assert -5 < sharpe < 5

    def test_calculate_sharpe_ratio_with_risk_free(
        self, analytics: PortfolioAnalytics, sample_returns: pd.Series
    ) -> None:
        """Test Sharpe ratio with risk-free rate."""
        sharpe_zero = analytics.calculate_sharpe_ratio(sample_returns, risk_free_rate=0.0)
        sharpe_with_rf = analytics.calculate_sharpe_ratio(sample_returns, risk_free_rate=0.05)

        # Higher risk-free rate should reduce Sharpe ratio
        assert sharpe_with_rf < sharpe_zero

    def test_calculate_sortino_ratio(
        self, analytics: PortfolioAnalytics, sample_returns: pd.Series
    ) -> None:
        """Test Sortino ratio calculation."""
        sortino = analytics.calculate_sortino_ratio(sample_returns)

        # Sortino uses downside deviation, should be similar magnitude to Sharpe
        sharpe = analytics.calculate_sharpe_ratio(sample_returns)
        # Both should be in similar range
        assert -10 < sortino < 10

    def test_calculate_max_drawdown(
        self, analytics: PortfolioAnalytics, equity_curve: pd.Series
    ) -> None:
        """Test max drawdown calculation."""
        max_dd, peak_date, trough_date = analytics.calculate_max_drawdown(equity_curve)

        # Max drawdown should be positive
        assert max_dd > 0
        # Should be less than 100%
        assert max_dd < 1.0
        # Peak should be before trough
        if peak_date and trough_date:
            assert peak_date <= trough_date

    def test_calculate_current_drawdown(
        self, analytics: PortfolioAnalytics, equity_curve: pd.Series
    ) -> None:
        """Test current drawdown calculation."""
        current_dd = analytics.calculate_current_drawdown(equity_curve)

        # Should be >= 0
        assert current_dd >= 0
        # Should be <= max drawdown
        max_dd, _, _ = analytics.calculate_max_drawdown(equity_curve)
        assert current_dd <= max_dd + 0.01  # Small tolerance

    def test_calculate_beta(self, analytics: PortfolioAnalytics) -> None:
        """Test beta calculation."""
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=100, freq="D")

        # Create market returns
        market_returns = pd.Series(np.random.randn(100) * 0.01, index=dates)

        # Create asset with beta ~1.5 (moves 1.5x the market)
        asset_returns = pd.Series(1.5 * market_returns + np.random.randn(100) * 0.005, index=dates)

        beta = analytics.calculate_beta(asset_returns, market_returns)

        # Should be close to 1.5
        assert 1.0 < beta < 2.0

    def test_calculate_alpha(self, analytics: PortfolioAnalytics) -> None:
        """Test alpha calculation."""
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=100, freq="D")

        market_returns = pd.Series(np.random.randn(100) * 0.01, index=dates)
        # Asset with positive alpha (outperforms market)
        asset_returns = pd.Series(
            market_returns + 0.001, index=dates  # 0.1% daily outperformance
        )

        alpha = analytics.calculate_alpha(asset_returns, market_returns)

        # Annualized alpha should be positive
        assert alpha > 0

    def test_calculate_correlation_matrix(self, analytics: PortfolioAnalytics) -> None:
        """Test correlation matrix calculation."""
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=100, freq="D")

        price_data = {
            "AAPL": pd.Series(100 * np.cumprod(1 + np.random.randn(100) * 0.02), index=dates),
            "MSFT": pd.Series(200 * np.cumprod(1 + np.random.randn(100) * 0.02), index=dates),
            "GOOGL": pd.Series(150 * np.cumprod(1 + np.random.randn(100) * 0.02), index=dates),
        }

        result = analytics.calculate_correlation_matrix(price_data)

        assert isinstance(result, CorrelationResult)
        assert len(result.symbols) == 3
        # Diagonal should be 1.0
        for sym in result.symbols:
            assert result.matrix.loc[sym, sym] == pytest.approx(1.0)
        # Correlation should be between -1 and 1
        assert (result.matrix >= -1).all().all()
        assert (result.matrix <= 1).all().all()

    def test_calculate_portfolio_var(self, analytics: PortfolioAnalytics) -> None:
        """Test portfolio VaR calculation."""
        pytest.importorskip("scipy")

        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=100, freq="D")

        returns_data = {
            "AAPL": pd.Series(np.random.randn(100) * 0.02, index=dates),
            "MSFT": pd.Series(np.random.randn(100) * 0.02, index=dates),
        }

        weights = {"AAPL": 0.6, "MSFT": 0.4}

        portfolio_var = analytics.calculate_portfolio_var(weights, returns_data, 0.95)

        # Should be positive
        assert portfolio_var > 0
        # Should be less than max individual VaR (diversification benefit)
        individual_vars = [
            analytics.calculate_var(returns_data[sym], 0.95) for sym in returns_data
        ]
        assert portfolio_var < max(individual_vars) * 1.5  # Some tolerance

    def test_calculate_risk_metrics(
        self, analytics: PortfolioAnalytics, equity_curve: pd.Series
    ) -> None:
        """Test comprehensive risk metrics calculation."""
        metrics = analytics.calculate_risk_metrics(equity_curve)

        assert isinstance(metrics, RiskMetrics)
        assert metrics.var_95 > 0
        assert metrics.var_99 >= metrics.var_95
        assert metrics.volatility > 0
        assert metrics.max_drawdown > 0
        assert metrics.current_drawdown >= 0

    def test_calculate_risk_metrics_with_benchmark(
        self, analytics: PortfolioAnalytics, equity_curve: pd.Series, sample_returns: pd.Series
    ) -> None:
        """Test risk metrics with benchmark."""
        metrics = analytics.calculate_risk_metrics(equity_curve, sample_returns)

        assert metrics.beta is not None
        assert metrics.alpha is not None


class TestAnalyzePositions:
    """Tests for position analysis functions."""

    @pytest.fixture
    def sample_positions(self) -> list[Position]:
        """Create sample positions."""
        # Total value: 16000 + 16000 + 2100 = 34100
        return [
            Position(
                symbol="AAPL",
                quantity=100,
                avg_entry_price=Decimal("150.00"),
                current_price=Decimal("160.00"),
                market_value=Decimal("16000.00"),  # 100 * 160
                unrealized_pnl=Decimal("1000.00"),
                unrealized_pnl_pct=0.0667,
            ),
            Position(
                symbol="MSFT",
                quantity=50,
                avg_entry_price=Decimal("300.00"),
                current_price=Decimal("320.00"),
                market_value=Decimal("16000.00"),  # 50 * 320
                unrealized_pnl=Decimal("1000.00"),
                unrealized_pnl_pct=0.0667,
            ),
            Position(
                symbol="GOOGL",
                quantity=20,
                avg_entry_price=Decimal("100.00"),
                current_price=Decimal("105.00"),
                market_value=Decimal("2100.00"),  # 20 * 105
                unrealized_pnl=Decimal("100.00"),
                unrealized_pnl_pct=0.05,
            ),
        ]

    def test_analyze_positions(
        self,
        analytics: PortfolioAnalytics,
        sample_positions: list[Position],
    ) -> None:
        """Test position analysis."""
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=100, freq="D")

        returns_data = {
            "AAPL": pd.Series(np.random.randn(100) * 0.02, index=dates),
            "MSFT": pd.Series(np.random.randn(100) * 0.02, index=dates),
            "GOOGL": pd.Series(np.random.randn(100) * 0.02, index=dates),
        }

        results = analytics.analyze_positions(sample_positions, returns_data)

        assert len(results) == 3
        assert all(isinstance(r, PositionRisk) for r in results)

        # Weights should sum to ~1
        total_weight = sum(r.weight for r in results)
        assert total_weight == pytest.approx(1.0, rel=0.01)

    def test_calculate_sector_exposure(self, sample_positions: list[Position]) -> None:
        """Test sector exposure calculation."""
        sector_map = {
            "AAPL": "Technology",
            "MSFT": "Technology",
            "GOOGL": "Communication",
        }

        exposure = calculate_sector_exposure(sample_positions, sector_map)

        assert "Technology" in exposure
        assert "Communication" in exposure
        # Weights should sum to 1
        assert sum(exposure.values()) == pytest.approx(1.0, rel=0.01)
        # Tech should be larger (AAPL + MSFT)
        assert exposure["Technology"] > exposure["Communication"]

    def test_calculate_concentration_risk(self, sample_positions: list[Position]) -> None:
        """Test concentration risk identification."""
        # With 20% threshold
        concentrated = calculate_concentration_risk(sample_positions, threshold=0.20)

        # AAPL and MSFT should be flagged (each ~47%)
        symbols = [c[0] for c in concentrated]
        assert "AAPL" in symbols
        assert "MSFT" in symbols

    def test_calculate_concentration_risk_high_threshold(
        self, sample_positions: list[Position]
    ) -> None:
        """Test concentration with high threshold."""
        # With 50% threshold, nothing should be flagged
        concentrated = calculate_concentration_risk(sample_positions, threshold=0.50)
        assert len(concentrated) == 0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_price_data(self, analytics: PortfolioAnalytics) -> None:
        """Test with empty data."""
        empty = pd.Series([], dtype=float)

        assert analytics.calculate_var(empty) == 0.0
        assert analytics.calculate_volatility(empty) == 0.0
        assert analytics.calculate_sharpe_ratio(empty) == 0.0

    def test_single_price(self, analytics: PortfolioAnalytics) -> None:
        """Test with single price point."""
        single = pd.Series([100.0])

        max_dd, _, _ = analytics.calculate_max_drawdown(single)
        assert max_dd == 0.0

    def test_constant_prices(self, analytics: PortfolioAnalytics) -> None:
        """Test with constant prices (zero volatility)."""
        dates = pd.date_range(end=datetime.now(), periods=100, freq="D")
        constant = pd.Series([100.0] * 100, index=dates)

        vol = analytics.calculate_volatility(analytics.calculate_returns(constant, "simple"))
        assert vol == 0.0

    def test_correlation_with_two_symbols(self, analytics: PortfolioAnalytics) -> None:
        """Test correlation with minimum symbols."""
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=50, freq="D")

        price_data = {
            "A": pd.Series(100 * np.cumprod(1 + np.random.randn(50) * 0.02), index=dates),
            "B": pd.Series(100 * np.cumprod(1 + np.random.randn(50) * 0.02), index=dates),
        }

        result = analytics.calculate_correlation_matrix(price_data)
        assert len(result.symbols) == 2

    def test_empty_positions(self, analytics: PortfolioAnalytics) -> None:
        """Test with empty positions list."""
        results = analytics.analyze_positions([], {})
        assert len(results) == 0

    def test_sector_exposure_empty(self) -> None:
        """Test sector exposure with empty positions."""
        exposure = calculate_sector_exposure([], {})
        assert exposure == {}

    def test_concentration_risk_empty(self) -> None:
        """Test concentration risk with empty positions."""
        concentrated = calculate_concentration_risk([])
        assert concentrated == []
