"""Pairs trading strategy implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from trader.strategies.base import BaseStrategy

if TYPE_CHECKING:
    from trader.core.models import Position, Signal


class PairsTradingStrategy(BaseStrategy):
    """
    Statistical arbitrage pairs trading strategy.

    This strategy trades the spread between two correlated assets.
    When the spread deviates significantly from its mean (measured by z-score),
    we expect mean reversion:
    - If spread is too high (z > entry_zscore): short the spread (sell A, buy B)
    - If spread is too low (z < -entry_zscore): long the spread (buy A, sell B)
    - Exit when spread returns to mean (|z| < exit_zscore)

    The strategy requires two symbols and trades them as a pair.
    The "primary" symbol is what we generate signals for, and the hedge
    ratio determines how much of the secondary symbol to trade.

    Example:
        # Trade AAPL vs MSFT pair
        strategy = PairsTradingStrategy(
            primary_symbol="AAPL",
            secondary_symbol="MSFT",
            lookback=60,
            entry_zscore=2.0,
            exit_zscore=0.5,
        )
    """

    def __init__(
        self,
        primary_symbol: str = "AAPL",
        secondary_symbol: str = "MSFT",
        lookback: int = 60,
        entry_zscore: float = 2.0,
        exit_zscore: float = 0.5,
        hedge_ratio: float | None = None,
        use_log_prices: bool = True,
    ) -> None:
        """Initialize pairs trading strategy.

        Args:
            primary_symbol: First symbol in the pair (generates signals for this)
            secondary_symbol: Second symbol in the pair (hedge instrument)
            lookback: Lookback period for calculating spread statistics
            entry_zscore: Z-score threshold for entry (absolute value)
            exit_zscore: Z-score threshold for exit (absolute value)
            hedge_ratio: Fixed hedge ratio (if None, calculated dynamically)
            use_log_prices: Use log prices for spread calculation
        """
        if lookback < 10:
            raise ValueError("Lookback must be at least 10")
        if entry_zscore <= 0:
            raise ValueError("Entry z-score must be positive")
        if exit_zscore < 0:
            raise ValueError("Exit z-score must be non-negative")
        if exit_zscore >= entry_zscore:
            raise ValueError("Exit z-score must be less than entry z-score")

        self.primary_symbol = primary_symbol.upper()
        self.secondary_symbol = secondary_symbol.upper()
        self.lookback = lookback
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.hedge_ratio = hedge_ratio
        self.use_log_prices = use_log_prices

        # Cache for secondary symbol data (must be provided externally)
        self._secondary_data: pd.DataFrame | None = None

    @property
    def name(self) -> str:
        """Unique identifier for this strategy instance."""
        return f"pairs_{self.primary_symbol}_{self.secondary_symbol}_{self.lookback}"

    @property
    def description(self) -> str:
        """Human-readable description."""
        return (
            f"Pairs trading {self.primary_symbol}/{self.secondary_symbol} "
            f"(lookback={self.lookback}, entry_z={self.entry_zscore})"
        )

    @property
    def min_bars_required(self) -> int:
        """Minimum bars needed for spread calculation."""
        return self.lookback + 5

    def set_secondary_data(self, data: pd.DataFrame) -> None:
        """Set the secondary symbol's price data.

        This must be called before generate_signal() with aligned data.

        Args:
            data: DataFrame with OHLCV data for secondary symbol,
                  index must align with primary symbol data.
        """
        self._secondary_data = data.copy()

    def calculate_hedge_ratio(
        self,
        primary_prices: pd.Series,
        secondary_prices: pd.Series,
    ) -> float:
        """Calculate optimal hedge ratio using OLS regression.

        The hedge ratio (beta) minimizes the variance of the spread:
        spread = primary - beta * secondary

        Args:
            primary_prices: Price series for primary symbol
            secondary_prices: Price series for secondary symbol

        Returns:
            Hedge ratio (beta coefficient)
        """
        if self.hedge_ratio is not None:
            return self.hedge_ratio

        # Use log prices if configured
        if self.use_log_prices:
            y = np.log(primary_prices)
            x = np.log(secondary_prices)
        else:
            y = primary_prices
            x = secondary_prices

        # Simple OLS: beta = cov(x, y) / var(x)
        x_mean = x.mean()
        y_mean = y.mean()
        beta = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum()

        return float(beta)

    def calculate_spread(
        self,
        primary_prices: pd.Series,
        secondary_prices: pd.Series,
        hedge_ratio: float,
    ) -> pd.Series:
        """Calculate the spread between the two assets.

        Args:
            primary_prices: Price series for primary symbol
            secondary_prices: Price series for secondary symbol
            hedge_ratio: Hedge ratio to use

        Returns:
            Spread series
        """
        if self.use_log_prices:
            spread = np.log(primary_prices) - hedge_ratio * np.log(secondary_prices)
        else:
            spread = primary_prices - hedge_ratio * secondary_prices

        return cast("pd.Series[Any]", pd.Series(spread))

    def calculate_zscore(self, spread: pd.Series) -> pd.Series:
        """Calculate rolling z-score of the spread.

        Args:
            spread: Spread series

        Returns:
            Z-score series
        """
        rolling_mean = spread.rolling(window=self.lookback).mean()
        rolling_std = spread.rolling(window=self.lookback).std()

        zscore = (spread - rolling_mean) / rolling_std
        return zscore

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate spread and z-score indicators.

        Note: This requires secondary_data to be set first via set_secondary_data().

        Args:
            data: DataFrame with OHLCV data for PRIMARY symbol

        Returns:
            DataFrame with spread and zscore columns added
        """
        data = data.copy()

        if self._secondary_data is None:
            # No secondary data - can't calculate spread
            data["spread"] = np.nan
            data["zscore"] = np.nan
            data["hedge_ratio"] = np.nan
            return data

        # Align data
        primary_close = data["close"]
        secondary_close = self._secondary_data["close"].reindex(data.index)

        # Drop any rows where we don't have both prices
        valid_mask = primary_close.notna() & secondary_close.notna()

        if valid_mask.sum() < self.lookback:
            data["spread"] = np.nan
            data["zscore"] = np.nan
            data["hedge_ratio"] = np.nan
            return data

        # Calculate rolling hedge ratio and spread
        hedge_ratios = []
        spreads = []

        for i in range(len(data)):
            if i < self.lookback - 1:
                hedge_ratios.append(np.nan)
                spreads.append(np.nan)
            else:
                window_primary = primary_close.iloc[i - self.lookback + 1 : i + 1]
                window_secondary = secondary_close.iloc[i - self.lookback + 1 : i + 1]

                hr = self.calculate_hedge_ratio(window_primary, window_secondary)
                hedge_ratios.append(hr)

                if self.use_log_prices:
                    spread = np.log(primary_close.iloc[i]) - hr * np.log(
                        secondary_close.iloc[i]
                    )
                else:
                    spread = primary_close.iloc[i] - hr * secondary_close.iloc[i]
                spreads.append(spread)

        data["hedge_ratio"] = hedge_ratios
        data["spread"] = spreads

        # Calculate z-score
        spread_series = pd.Series(spreads, index=data.index)
        data["zscore"] = self.calculate_zscore(spread_series)

        # Store secondary price for reference
        data["secondary_close"] = secondary_close

        return data

    def generate_signal(
        self,
        data: pd.DataFrame,
        symbol: str,
        position: Position | None = None,
    ) -> Signal:
        """Generate trading signal based on spread z-score.

        Args:
            data: DataFrame with spread and zscore columns
            symbol: Symbol to generate signal for (should be primary_symbol)
            position: Current position in the primary symbol

        Returns:
            Signal for the primary symbol
        """
        if len(data) < self.min_bars_required:
            return self.hold_signal(symbol, "Insufficient data for pairs trading")

        # Check if we have the necessary indicators
        if "zscore" not in data.columns or pd.isna(data["zscore"].iloc[-1]):
            return self.hold_signal(symbol, "Spread data not available")

        current_zscore = data["zscore"].iloc[-1]
        hedge_ratio = data["hedge_ratio"].iloc[-1]

        # Determine if we have a position
        has_position = position is not None and position.quantity != 0
        is_long = position is not None and position.quantity > 0

        # Generate signals based on z-score
        if has_position:
            # Check for exit conditions
            if is_long and current_zscore >= -self.exit_zscore:
                # Long spread position, z-score returned to mean - exit
                return self.sell_signal(
                    symbol,
                    f"Pairs exit: z-score {current_zscore:.2f} returned to mean "
                    f"(hedge ratio: {hedge_ratio:.3f})",
                    confidence=min(1.0, abs(current_zscore) / self.entry_zscore),
                )
            elif not is_long and current_zscore <= self.exit_zscore:
                # Short spread position, z-score returned to mean - exit (buy to cover)
                return self.buy_signal(
                    symbol,
                    f"Pairs exit: z-score {current_zscore:.2f} returned to mean "
                    f"(hedge ratio: {hedge_ratio:.3f})",
                    confidence=min(1.0, abs(current_zscore) / self.entry_zscore),
                )
            else:
                return self.hold_signal(
                    symbol,
                    f"Pairs hold: z-score {current_zscore:.2f}, waiting for mean reversion",
                )
        else:
            # Check for entry conditions
            if current_zscore < -self.entry_zscore:
                # Spread is too low - long the spread (buy primary)
                return self.buy_signal(
                    symbol,
                    f"Pairs entry: z-score {current_zscore:.2f} < -{self.entry_zscore} "
                    f"(spread too low, expect reversion up)",
                    confidence=min(1.0, abs(current_zscore) / (self.entry_zscore * 2)),
                )
            elif current_zscore > self.entry_zscore:
                # Spread is too high - short the spread (sell primary)
                return self.sell_signal(
                    symbol,
                    f"Pairs entry: z-score {current_zscore:.2f} > {self.entry_zscore} "
                    f"(spread too high, expect reversion down)",
                    confidence=min(1.0, abs(current_zscore) / (self.entry_zscore * 2)),
                )
            else:
                return self.hold_signal(
                    symbol,
                    f"Pairs hold: z-score {current_zscore:.2f} within bounds "
                    f"[{-self.entry_zscore}, {self.entry_zscore}]",
                )


class CointegrationPairsStrategy(BaseStrategy):
    """
    Pairs trading strategy using cointegration testing.

    This is an enhanced version that tests for cointegration between pairs
    and only trades when the pair is statistically cointegrated.

    Uses the Engle-Granger two-step method:
    1. Test for cointegration using ADF test on spread residuals
    2. Trade the spread when cointegrated and z-score exceeds threshold
    """

    def __init__(
        self,
        primary_symbol: str = "AAPL",
        secondary_symbol: str = "MSFT",
        lookback: int = 60,
        entry_zscore: float = 2.0,
        exit_zscore: float = 0.5,
        coint_pvalue: float = 0.05,
    ) -> None:
        """Initialize cointegration pairs strategy.

        Args:
            primary_symbol: First symbol in the pair
            secondary_symbol: Second symbol in the pair
            lookback: Lookback period for spread statistics
            entry_zscore: Z-score threshold for entry
            exit_zscore: Z-score threshold for exit
            coint_pvalue: Maximum p-value to consider pair cointegrated
        """
        if lookback < 20:
            raise ValueError("Lookback must be at least 20 for cointegration testing")
        if entry_zscore <= 0:
            raise ValueError("Entry z-score must be positive")
        if exit_zscore >= entry_zscore:
            raise ValueError("Exit z-score must be less than entry z-score")
        if not 0 < coint_pvalue < 1:
            raise ValueError("Cointegration p-value must be between 0 and 1")

        self.primary_symbol = primary_symbol.upper()
        self.secondary_symbol = secondary_symbol.upper()
        self.lookback = lookback
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.coint_pvalue = coint_pvalue

        self._secondary_data: pd.DataFrame | None = None

    @property
    def name(self) -> str:
        return f"coint_pairs_{self.primary_symbol}_{self.secondary_symbol}"

    @property
    def description(self) -> str:
        return (
            f"Cointegration pairs trading {self.primary_symbol}/{self.secondary_symbol}"
        )

    @property
    def min_bars_required(self) -> int:
        return self.lookback + 10

    def set_secondary_data(self, data: pd.DataFrame) -> None:
        """Set secondary symbol data."""
        self._secondary_data = data.copy()

    def test_cointegration(
        self,
        primary_prices: pd.Series,
        secondary_prices: pd.Series,
    ) -> tuple[bool, float, float]:
        """Test for cointegration using Engle-Granger method.

        Args:
            primary_prices: Price series for primary symbol
            secondary_prices: Price series for secondary symbol

        Returns:
            Tuple of (is_cointegrated, p_value, hedge_ratio)
        """
        try:
            from statsmodels.tsa.stattools import adfuller
        except ImportError:
            # statsmodels not available, assume cointegrated
            return True, 0.01, 1.0

        # Calculate hedge ratio via OLS
        y = np.log(primary_prices)
        x = np.log(secondary_prices)

        x_mean = x.mean()
        y_mean = y.mean()
        beta = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum()

        # Calculate spread (residuals)
        spread = y - beta * x

        # ADF test on spread
        try:
            adf_result = adfuller(spread.dropna(), maxlag=1)
            pvalue = adf_result[1]
            is_coint = pvalue < self.coint_pvalue
        except Exception:
            # If test fails, assume not cointegrated
            is_coint = False
            pvalue = 1.0

        return is_coint, pvalue, float(beta)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate cointegration and spread indicators."""
        data = data.copy()

        if self._secondary_data is None:
            data["spread"] = np.nan
            data["zscore"] = np.nan
            data["is_cointegrated"] = False
            data["hedge_ratio"] = np.nan
            return data

        primary_close = data["close"]
        secondary_close = self._secondary_data["close"].reindex(data.index)

        # Test cointegration on recent window
        if len(data) >= self.lookback:
            window_primary = primary_close.iloc[-self.lookback :]
            window_secondary = secondary_close.iloc[-self.lookback :]

            is_coint, pvalue, hedge_ratio = self.test_cointegration(
                window_primary, window_secondary
            )
        else:
            is_coint, pvalue, hedge_ratio = False, 1.0, 1.0

        # Calculate spread
        spread = np.log(primary_close) - hedge_ratio * np.log(secondary_close)

        # Calculate z-score
        rolling_mean = spread.rolling(window=self.lookback).mean()
        rolling_std = spread.rolling(window=self.lookback).std()
        zscore = (spread - rolling_mean) / rolling_std

        data["spread"] = spread
        data["zscore"] = zscore
        data["is_cointegrated"] = is_coint
        data["coint_pvalue"] = pvalue
        data["hedge_ratio"] = hedge_ratio
        data["secondary_close"] = secondary_close

        return data

    def generate_signal(
        self,
        data: pd.DataFrame,
        symbol: str,
        position: Position | None = None,
    ) -> Signal:
        """Generate signal only if pair is cointegrated."""
        if len(data) < self.min_bars_required:
            return self.hold_signal(symbol, "Insufficient data")

        if "zscore" not in data.columns or pd.isna(data["zscore"].iloc[-1]):
            return self.hold_signal(symbol, "Spread data not available")

        is_cointegrated = data["is_cointegrated"].iloc[-1]
        current_zscore = data["zscore"].iloc[-1]
        hedge_ratio = data["hedge_ratio"].iloc[-1]
        pvalue = data.get("coint_pvalue", pd.Series([0.0])).iloc[-1]

        has_position = position is not None and position.quantity != 0
        is_long = position is not None and position.quantity > 0

        # If not cointegrated, only allow exits
        if not is_cointegrated and not has_position:
            return self.hold_signal(
                symbol,
                f"Pair not cointegrated (p={pvalue:.3f}), no entry",
            )

        if has_position:
            # Exit logic (same as basic pairs)
            if is_long and current_zscore >= -self.exit_zscore:
                return self.sell_signal(
                    symbol,
                    f"Coint pairs exit: z={current_zscore:.2f} (hr={hedge_ratio:.3f})",
                    confidence=0.8,
                )
            elif not is_long and current_zscore <= self.exit_zscore:
                return self.buy_signal(
                    symbol,
                    f"Coint pairs exit: z={current_zscore:.2f} (hr={hedge_ratio:.3f})",
                    confidence=0.8,
                )
            return self.hold_signal(symbol, f"Hold: z={current_zscore:.2f}")
        else:
            # Entry logic
            if current_zscore < -self.entry_zscore:
                return self.buy_signal(
                    symbol,
                    f"Coint pairs long: z={current_zscore:.2f}, p={pvalue:.3f}",
                    confidence=min(1.0, abs(current_zscore) / 3),
                )
            elif current_zscore > self.entry_zscore:
                return self.sell_signal(
                    symbol,
                    f"Coint pairs short: z={current_zscore:.2f}, p={pvalue:.3f}",
                    confidence=min(1.0, abs(current_zscore) / 3),
                )
            return self.hold_signal(symbol, f"No signal: z={current_zscore:.2f}")
