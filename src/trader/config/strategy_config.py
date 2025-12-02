"""YAML-based strategy configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from pydantic import BaseModel, Field

from trader.config.settings import DEFAULT_DATA_DIR

# Default config path
DEFAULT_CONFIG_PATH = DEFAULT_DATA_DIR / "config.yaml"


class StrategyConfig(BaseModel):
    """Configuration for a single strategy."""

    name: str
    enabled: bool = True
    params: dict[str, Any] = Field(default_factory=dict)
    symbols: list[str] = Field(default_factory=list)
    description: str | None = None


class RiskConfig(BaseModel):
    """Risk management configuration."""

    max_position_size_pct: float = Field(default=0.1, ge=0, le=1)
    max_daily_loss_pct: float = Field(default=0.02, ge=0, le=1)
    stop_loss_pct: float = Field(default=0.05, ge=0, le=1)
    take_profit_pct: float | None = Field(default=None, ge=0, le=1)
    max_open_positions: int = Field(default=10, ge=1)


class BacktestConfig(BaseModel):
    """Backtest configuration."""

    initial_capital: float = Field(default=100000.0, gt=0)
    commission: float = Field(default=0.0, ge=0)
    days: int = Field(default=365, gt=0)


class WatchlistConfig(BaseModel):
    """Watchlist configuration."""

    name: str
    symbols: list[str]
    description: str | None = None


class TradingConfig(BaseModel):
    """Root configuration object."""

    strategies: list[StrategyConfig] = Field(default_factory=list)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    watchlists: list[WatchlistConfig] = Field(default_factory=list)
    default_symbols: list[str] = Field(
        default_factory=lambda: ["AAPL", "MSFT", "GOOGL"]
    )

    def get_strategy(self, name: str) -> StrategyConfig | None:
        """Get a strategy config by name."""
        for s in self.strategies:
            if s.name.lower() == name.lower():
                return s
        return None

    def get_enabled_strategies(self) -> list[StrategyConfig]:
        """Get all enabled strategies."""
        return [s for s in self.strategies if s.enabled]

    def get_watchlist(self, name: str) -> WatchlistConfig | None:
        """Get a watchlist by name."""
        for w in self.watchlists:
            if w.name.lower() == name.lower():
                return w
        return None


def load_config(config_path: Path | str | None = None) -> TradingConfig:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file. Defaults to ~/.trader/config.yaml

    Returns:
        TradingConfig object
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    config_path = Path(config_path)

    if not config_path.exists():
        logger.debug(f"Config file not found at {config_path}, using defaults")
        return TradingConfig()

    try:
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}

        config = TradingConfig(**data)
        logger.info(f"Loaded config from {config_path}")
        return config

    except yaml.YAMLError as e:
        logger.error(f"Failed to parse config file: {e}")
        return TradingConfig()
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return TradingConfig()


def save_config(config: TradingConfig, config_path: Path | str | None = None) -> None:
    """Save configuration to YAML file.

    Args:
        config: TradingConfig object to save
        config_path: Path to save to. Defaults to ~/.trader/config.yaml
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False, sort_keys=False)

    logger.info(f"Saved config to {config_path}")


def create_default_config(config_path: Path | str | None = None) -> TradingConfig:
    """Create a default config file with example strategies.

    Args:
        config_path: Path to save to. Defaults to ~/.trader/config.yaml

    Returns:
        The created TradingConfig
    """
    config = TradingConfig(
        strategies=[
            StrategyConfig(
                name="sma",
                enabled=True,
                params={"fast_period": 10, "slow_period": 50},
                symbols=["AAPL", "MSFT", "GOOGL"],
                description="SMA crossover strategy",
            ),
            StrategyConfig(
                name="rsi",
                enabled=True,
                params={"period": 14, "oversold": 30, "overbought": 70},
                symbols=["NVDA", "AMD"],
                description="RSI mean reversion",
            ),
            StrategyConfig(
                name="macd",
                enabled=False,
                params={"fast_period": 12, "slow_period": 26, "signal_period": 9},
                symbols=[],
                description="MACD crossover (disabled)",
            ),
            StrategyConfig(
                name="momentum",
                enabled=True,
                params={"lookback_days": 126, "skip_days": 5, "hold_days": 5},
                symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
                description="Momentum ranking strategy",
            ),
        ],
        risk=RiskConfig(
            max_position_size_pct=0.2,
            max_daily_loss_pct=0.02,
            stop_loss_pct=0.05,
            max_open_positions=5,
        ),
        backtest=BacktestConfig(
            initial_capital=100000.0,
            commission=0.0,
            days=365,
        ),
        watchlists=[
            WatchlistConfig(
                name="tech",
                symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
                description="Big tech stocks",
            ),
            WatchlistConfig(
                name="finance",
                symbols=["JPM", "BAC", "WFC", "GS", "MS"],
                description="Financial sector",
            ),
        ],
        default_symbols=["AAPL", "MSFT", "GOOGL"],
    )

    save_config(config, config_path)
    return config
