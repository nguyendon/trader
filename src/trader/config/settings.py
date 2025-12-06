"""Application settings using Pydantic."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Default paths
DEFAULT_DATA_DIR = Path.home() / ".trader"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Alpaca API settings
    alpaca_api_key: SecretStr = Field(default=SecretStr(""))
    alpaca_secret_key: SecretStr = Field(default=SecretStr(""))
    alpaca_paper: bool = Field(default=True)
    alpaca_base_url: str | None = Field(default=None)

    # Sentiment API settings (Alpha Vantage - free tier available)
    alphavantage_api_key: SecretStr = Field(default=SecretStr(""))

    # Reddit API settings (for web sentiment scraping)
    reddit_client_id: str = Field(default="")
    reddit_client_secret: SecretStr = Field(default=SecretStr(""))

    # Trading settings
    default_symbols: list[str] = Field(default=["AAPL", "MSFT", "GOOGL"])
    trading_mode: Literal["backtest", "paper", "live"] = Field(default="paper")

    # Risk settings
    max_position_size_pct: float = Field(default=0.1, ge=0, le=1)
    max_daily_loss_pct: float = Field(default=0.02, ge=0, le=1)
    stop_loss_pct: float = Field(default=0.05, ge=0, le=1)
    max_open_positions: int = Field(default=10, ge=1)

    # Data/storage settings
    data_dir: Path = Field(default=DEFAULT_DATA_DIR)
    db_path: Path | None = Field(default=None)  # Defaults to data_dir/trader.db

    # Logging settings
    log_level: str = Field(default="INFO")
    log_to_file: bool = Field(default=True)
    log_dir: Path | None = Field(default=None)  # Defaults to data_dir/logs
    log_retention_days: int = Field(default=30)

    # Notification settings
    discord_webhook_url: str | None = Field(default=None)
    notify_on_trade: bool = Field(default=True)
    notify_on_error: bool = Field(default=True)
    notify_daily_summary: bool = Field(default=True)

    @property
    def has_alpaca_credentials(self) -> bool:
        """Check if Alpaca credentials are configured."""
        return bool(
            self.alpaca_api_key.get_secret_value()
            and self.alpaca_secret_key.get_secret_value()
        )

    @property
    def has_alphavantage_credentials(self) -> bool:
        """Check if Alpha Vantage API key is configured."""
        return bool(self.alphavantage_api_key.get_secret_value())

    @property
    def has_reddit_credentials(self) -> bool:
        """Check if Reddit API credentials are configured."""
        return bool(
            self.reddit_client_id and self.reddit_client_secret.get_secret_value()
        )

    @property
    def alpaca_url(self) -> str:
        """Get the Alpaca API base URL."""
        if self.alpaca_base_url:
            return self.alpaca_base_url
        if self.alpaca_paper:
            return "https://paper-api.alpaca.markets"
        return "https://api.alpaca.markets"

    @property
    def database_path(self) -> Path:
        """Get the database file path."""
        if self.db_path:
            return self.db_path
        return self.data_dir / "trader.db"

    @property
    def logs_path(self) -> Path:
        """Get the logs directory path."""
        if self.log_dir:
            return self.log_dir
        return self.data_dir / "logs"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def setup_logging(settings: Settings | None = None) -> None:
    """Configure loguru for file and console logging.

    Args:
        settings: Optional settings instance. Uses default if not provided.
    """
    import sys

    from loguru import logger

    if settings is None:
        settings = get_settings()

    # Remove default handler
    logger.remove()

    # Add console handler
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )

    # Add file handler if enabled
    if settings.log_to_file:
        log_path = settings.logs_path
        log_path.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_path / "trader_{time:YYYY-MM-DD}.log",
            level=settings.log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="00:00",  # Rotate at midnight
            retention=f"{settings.log_retention_days} days",
            compression="gz",
        )

        logger.info(f"Logging to {log_path}")
