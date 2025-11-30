"""Storage module for persisting trades and backtest results."""

from trader.storage.database import TradeStore, get_trade_store

__all__ = ["TradeStore", "get_trade_store"]
