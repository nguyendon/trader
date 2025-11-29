"""Application settings using Pydantic."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # Trading settings
    default_symbols: list[str] = Field(default=["AAPL", "MSFT", "GOOGL"])
    trading_mode: Literal["backtest", "paper", "live"] = Field(default="paper")

    # Risk settings
    max_position_size_pct: float = Field(default=0.1, ge=0, le=1)
    max_daily_loss_pct: float = Field(default=0.02, ge=0, le=1)
    stop_loss_pct: float = Field(default=0.05, ge=0, le=1)
    max_open_positions: int = Field(default=10, ge=1)

    # Data settings
    data_cache_dir: str = Field(default="data/cache")
    database_url: str = Field(default="sqlite:///data/trader.db")

    # Logging
    log_level: str = Field(default="INFO")

    @property
    def has_alpaca_credentials(self) -> bool:
        """Check if Alpaca credentials are configured."""
        return bool(
            self.alpaca_api_key.get_secret_value()
            and self.alpaca_secret_key.get_secret_value()
        )

    @property
    def alpaca_url(self) -> str:
        """Get the Alpaca API base URL."""
        if self.alpaca_base_url:
            return self.alpaca_base_url
        if self.alpaca_paper:
            return "https://paper-api.alpaca.markets"
        return "https://api.alpaca.markets"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
