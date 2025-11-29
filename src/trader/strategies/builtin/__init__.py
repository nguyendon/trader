"""Built-in trading strategies."""

from trader.strategies.builtin.macd import MACDStrategy
from trader.strategies.builtin.momentum import MomentumStrategy
from trader.strategies.builtin.rsi import RSIStrategy
from trader.strategies.builtin.sma_crossover import SMACrossover

__all__ = ["SMACrossover", "RSIStrategy", "MACDStrategy", "MomentumStrategy"]
