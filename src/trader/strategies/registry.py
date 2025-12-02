"""Strategy registry for dynamic strategy loading."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from trader.strategies.base import BaseStrategy

# Registry of strategy name -> (class, default_kwargs)
_STRATEGIES: dict[str, tuple[type[BaseStrategy], dict[str, Any]]] = {}


def register_strategy(
    name: str,
    strategy_class: type[BaseStrategy],
    default_kwargs: dict[str, Any] | None = None,
) -> None:
    """Register a strategy class with a name.

    Args:
        name: Short name for CLI usage (e.g., "sma", "rsi")
        strategy_class: The strategy class
        default_kwargs: Default constructor arguments
    """
    _STRATEGIES[name.lower()] = (strategy_class, default_kwargs or {})


def get_strategy(name: str, **kwargs: Any) -> BaseStrategy:
    """Get a strategy instance by name.

    Args:
        name: Strategy name (e.g., "sma", "rsi")
        **kwargs: Override default constructor arguments

    Returns:
        Configured strategy instance

    Raises:
        ValueError: If strategy name is not found
    """
    name_lower = name.lower()
    if name_lower not in _STRATEGIES:
        available = ", ".join(sorted(_STRATEGIES.keys()))
        raise ValueError(f"Unknown strategy: {name}. Available: {available}")

    strategy_class, default_kwargs = _STRATEGIES[name_lower]
    merged_kwargs = {**default_kwargs, **kwargs}
    return strategy_class(**merged_kwargs)


def list_strategies() -> list[dict[str, str]]:
    """List all registered strategies.

    Returns:
        List of dicts with name and description
    """
    result = []
    for name, (strategy_class, defaults) in sorted(_STRATEGIES.items()):
        # Create instance to get description
        try:
            instance = strategy_class(**defaults)
            description = instance.description
        except Exception:
            description = strategy_class.__doc__ or "No description"
        result.append({"name": name, "description": description})
    return result


def _register_builtin_strategies() -> None:
    """Register all built-in strategies."""
    from trader.strategies.builtin.sma_crossover import SMACrossover

    register_strategy("sma", SMACrossover, {"fast_period": 10, "slow_period": 50})

    # Import and register other strategies as they're added
    try:
        from trader.strategies.builtin.rsi import RSIStrategy

        register_strategy(
            "rsi", RSIStrategy, {"period": 14, "oversold": 30, "overbought": 70}
        )
    except ImportError:
        pass

    try:
        from trader.strategies.builtin.macd import MACDStrategy

        register_strategy("macd", MACDStrategy)
    except ImportError:
        pass

    try:
        from trader.strategies.builtin.momentum import MomentumStrategy

        register_strategy(
            "momentum", MomentumStrategy, {"lookback_days": 126, "hold_days": 5}
        )
    except ImportError:
        pass


# Auto-register on import
_register_builtin_strategies()
