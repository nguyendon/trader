"""Tests for configuration settings."""

from __future__ import annotations

import os
from unittest.mock import patch

from trader.config.settings import Settings


class TestSettings:
    """Tests for Settings model."""

    def test_default_settings(self) -> None:
        """Test default settings values."""
        # Clear env vars and disable .env file reading
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)

        assert settings.alpaca_paper is True
        assert settings.trading_mode == "paper"
        assert settings.max_position_size_pct == 0.1
        assert settings.max_daily_loss_pct == 0.02
        assert settings.stop_loss_pct == 0.05
        assert settings.max_open_positions == 10
        assert settings.log_level == "INFO"

    def test_has_alpaca_credentials_false_by_default(self) -> None:
        """Test that has_alpaca_credentials is False without credentials."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)

        assert settings.has_alpaca_credentials is False

    def test_has_alpaca_credentials_true_with_credentials(self) -> None:
        """Test that has_alpaca_credentials is True with credentials."""
        env = {
            "ALPACA_API_KEY": "test_key",
            "ALPACA_SECRET_KEY": "test_secret",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = Settings(_env_file=None)

        assert settings.has_alpaca_credentials is True

    def test_alpaca_url_paper_trading(self) -> None:
        """Test Alpaca URL for paper trading."""
        with patch.dict(os.environ, {"ALPACA_PAPER": "true"}, clear=True):
            settings = Settings(_env_file=None)

        assert settings.alpaca_url == "https://paper-api.alpaca.markets"

    def test_alpaca_url_live_trading(self) -> None:
        """Test Alpaca URL for live trading."""
        with patch.dict(os.environ, {"ALPACA_PAPER": "false"}, clear=True):
            settings = Settings(_env_file=None)

        assert settings.alpaca_url == "https://api.alpaca.markets"

    def test_alpaca_url_custom_override(self) -> None:
        """Test custom Alpaca URL override."""
        env = {"ALPACA_BASE_URL": "https://custom.api.com"}
        with patch.dict(os.environ, env, clear=True):
            settings = Settings(_env_file=None)

        assert settings.alpaca_url == "https://custom.api.com"

    def test_risk_settings_from_env(self) -> None:
        """Test loading risk settings from environment."""
        env = {
            "MAX_POSITION_SIZE_PCT": "0.05",
            "MAX_DAILY_LOSS_PCT": "0.01",
            "STOP_LOSS_PCT": "0.03",
            "MAX_OPEN_POSITIONS": "5",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = Settings(_env_file=None)

        assert settings.max_position_size_pct == 0.05
        assert settings.max_daily_loss_pct == 0.01
        assert settings.stop_loss_pct == 0.03
        assert settings.max_open_positions == 5
