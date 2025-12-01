"""Trading strategies."""

from trader.strategies.base import BaseStrategy, StrategyConfig
from trader.strategies.multi import (
    AggregationMethod,
    MultiStrategyProcessor,
    StrategyAllocation,
    StrategyGroup,
    create_strategy_group,
)

__all__ = [
    "BaseStrategy",
    "StrategyConfig",
    "AggregationMethod",
    "MultiStrategyProcessor",
    "StrategyAllocation",
    "StrategyGroup",
    "create_strategy_group",
]
