"""Multi-strategy support for running multiple strategies simultaneously."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import pandas as pd
from loguru import logger

from trader.core.models import Signal, SignalAction
from trader.strategies.base import BaseStrategy
from trader.strategies.registry import get_strategy

if TYPE_CHECKING:
    from trader.core.models import Position


class AggregationMethod(str, Enum):
    """Methods for aggregating signals from multiple strategies."""

    MAJORITY = "majority"  # Most common action wins
    WEIGHTED = "weighted"  # Weight by confidence
    UNANIMOUS = "unanimous"  # All must agree for action
    ANY = "any"  # Any buy/sell signal triggers action
    FIRST = "first"  # First strategy's signal wins (priority order)


@dataclass
class StrategyAllocation:
    """Configuration for a single strategy in a multi-strategy setup.

    Attributes:
        name: Strategy name (from registry)
        weight: Portfolio weight for this strategy (0.0-1.0)
        symbols: Symbols this strategy trades (empty = all symbols)
        parameters: Strategy-specific parameters
        enabled: Whether this strategy is active
    """

    name: str
    weight: float = 1.0
    symbols: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def __post_init__(self) -> None:
        """Validate allocation."""
        if self.weight < 0:
            raise ValueError(f"Weight must be non-negative, got {self.weight}")


@dataclass
class StrategyGroup:
    """A group of strategies that work together.

    Attributes:
        allocations: List of strategy allocations
        aggregation: How to combine signals from multiple strategies
        min_confidence: Minimum confidence threshold for action
    """

    allocations: list[StrategyAllocation]
    aggregation: AggregationMethod = AggregationMethod.WEIGHTED
    min_confidence: float = 0.5

    def __post_init__(self) -> None:
        """Validate group configuration."""
        if not self.allocations:
            raise ValueError("StrategyGroup must have at least one allocation")

        # Normalize weights
        total_weight = sum(a.weight for a in self.allocations if a.enabled)
        if total_weight > 0:
            for alloc in self.allocations:
                if alloc.enabled:
                    alloc.weight = alloc.weight / total_weight


class MultiStrategyProcessor:
    """
    Processes multiple strategies and aggregates their signals.

    This class manages multiple strategy instances, generates signals from each,
    and combines them according to the specified aggregation method.
    """

    def __init__(self, group: StrategyGroup) -> None:
        """Initialize with a strategy group configuration.

        Args:
            group: Strategy group with allocations and aggregation settings
        """
        self.group = group
        self._strategies: dict[str, BaseStrategy] = {}
        self._initialize_strategies()

    def _initialize_strategies(self) -> None:
        """Create strategy instances from allocations."""
        for alloc in self.group.allocations:
            if not alloc.enabled:
                continue

            try:
                strategy = get_strategy(alloc.name, **alloc.parameters)
                # Use unique key combining name and parameters
                key = self._strategy_key(alloc)
                self._strategies[key] = strategy
                logger.info(
                    f"Initialized strategy '{alloc.name}' "
                    f"(weight={alloc.weight:.2f}, symbols={alloc.symbols or 'all'})"
                )
            except ValueError as e:
                logger.error(f"Failed to initialize strategy '{alloc.name}': {e}")
                raise

    def _strategy_key(self, alloc: StrategyAllocation) -> str:
        """Generate unique key for a strategy allocation."""
        params_str = "_".join(f"{k}={v}" for k, v in sorted(alloc.parameters.items()))
        return f"{alloc.name}_{params_str}" if params_str else alloc.name

    def get_strategies_for_symbol(self, symbol: str) -> list[tuple[StrategyAllocation, BaseStrategy]]:
        """Get all strategies that trade a specific symbol.

        Args:
            symbol: The symbol to check

        Returns:
            List of (allocation, strategy) tuples for this symbol
        """
        result = []
        for alloc in self.group.allocations:
            if not alloc.enabled:
                continue
            # Empty symbols list means trade all symbols
            if not alloc.symbols or symbol in alloc.symbols:
                key = self._strategy_key(alloc)
                if key in self._strategies:
                    result.append((alloc, self._strategies[key]))
        return result

    def generate_signals(
        self,
        data: pd.DataFrame,
        symbol: str,
        position: Position | None = None,
    ) -> list[tuple[StrategyAllocation, Signal]]:
        """Generate signals from all applicable strategies.

        Args:
            data: DataFrame with OHLCV data
            symbol: Symbol to generate signals for
            position: Current position, if any

        Returns:
            List of (allocation, signal) tuples from each strategy
        """
        signals = []
        strategies = self.get_strategies_for_symbol(symbol)

        for alloc, strategy in strategies:
            try:
                # Check if strategy has enough data
                if not strategy.should_generate_signal(data):
                    logger.debug(
                        f"Strategy '{alloc.name}' needs more data "
                        f"(has {len(data)}, needs {strategy.min_bars_required})"
                    )
                    continue

                # Calculate indicators (each strategy may add different ones)
                data_with_indicators = strategy.calculate_indicators(data)

                # Generate signal
                signal = strategy.generate_signal(data_with_indicators, symbol, position)

                # Add strategy info to metadata
                signal.metadata["strategy"] = alloc.name
                signal.metadata["weight"] = alloc.weight

                signals.append((alloc, signal))
                logger.debug(
                    f"Strategy '{alloc.name}' -> {signal.action.value} "
                    f"(confidence={signal.confidence:.2f})"
                )

            except Exception as e:
                logger.error(f"Error generating signal from '{alloc.name}': {e}")
                continue

        return signals

    def aggregate_signals(
        self,
        signals: list[tuple[StrategyAllocation, Signal]],
        symbol: str,
    ) -> Signal:
        """Aggregate multiple signals into a single decision.

        Args:
            signals: List of (allocation, signal) tuples
            symbol: Symbol being traded

        Returns:
            Aggregated signal representing the combined decision
        """
        if not signals:
            return Signal(
                action=SignalAction.HOLD,
                symbol=symbol,
                reason="No signals from any strategy",
                confidence=0.0,
            )

        method = self.group.aggregation

        if method == AggregationMethod.FIRST:
            return self._aggregate_first(signals, symbol)
        elif method == AggregationMethod.MAJORITY:
            return self._aggregate_majority(signals, symbol)
        elif method == AggregationMethod.WEIGHTED:
            return self._aggregate_weighted(signals, symbol)
        elif method == AggregationMethod.UNANIMOUS:
            return self._aggregate_unanimous(signals, symbol)
        elif method == AggregationMethod.ANY:
            return self._aggregate_any(signals, symbol)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def _aggregate_first(
        self,
        signals: list[tuple[StrategyAllocation, Signal]],
        symbol: str,
    ) -> Signal:
        """Use first strategy's signal (priority order)."""
        _, signal = signals[0]
        signal.metadata["aggregation"] = "first"
        signal.metadata["num_strategies"] = len(signals)
        return signal

    def _aggregate_majority(
        self,
        signals: list[tuple[StrategyAllocation, Signal]],
        symbol: str,
    ) -> Signal:
        """Use the most common action."""
        action_counts: dict[SignalAction, int] = {
            SignalAction.BUY: 0,
            SignalAction.SELL: 0,
            SignalAction.HOLD: 0,
        }

        for _, signal in signals:
            action_counts[signal.action] += 1

        # Find majority action
        majority_action = max(action_counts, key=lambda a: action_counts[a])
        majority_count = action_counts[majority_action]
        total = len(signals)

        # Calculate confidence based on agreement
        confidence = majority_count / total

        # Collect reasons from agreeing strategies
        agreeing_strategies = [
            alloc.name for alloc, sig in signals if sig.action == majority_action
        ]

        return Signal(
            action=majority_action,
            symbol=symbol,
            confidence=confidence,
            reason=f"Majority ({majority_count}/{total}): {', '.join(agreeing_strategies)}",
            metadata={
                "aggregation": "majority",
                "num_strategies": total,
                "action_counts": {k.value: v for k, v in action_counts.items()},
            },
        )

    def _aggregate_weighted(
        self,
        signals: list[tuple[StrategyAllocation, Signal]],
        symbol: str,
    ) -> Signal:
        """Weight signals by allocation weight and confidence."""
        action_scores: dict[SignalAction, float] = {
            SignalAction.BUY: 0.0,
            SignalAction.SELL: 0.0,
            SignalAction.HOLD: 0.0,
        }

        # Calculate weighted scores for each action
        # Score = sum(weight * confidence) for each strategy voting for that action
        total_allocation_weight = sum(alloc.weight for alloc, _ in signals)
        for alloc, signal in signals:
            # Weighted contribution = allocation weight * signal confidence
            contribution = alloc.weight * signal.confidence
            action_scores[signal.action] += contribution

        # Normalize by total allocation weight (so scores represent weighted average confidence)
        if total_allocation_weight > 0:
            for action in action_scores:
                action_scores[action] /= total_allocation_weight

        # Find best action
        best_action = max(action_scores, key=lambda a: action_scores[a])
        confidence = action_scores[best_action]

        # Check minimum confidence threshold
        # This checks whether the weighted average confidence meets the threshold
        if confidence < self.group.min_confidence and best_action != SignalAction.HOLD:
            return Signal(
                action=SignalAction.HOLD,
                symbol=symbol,
                confidence=confidence,
                reason=f"Confidence {confidence:.2f} below threshold {self.group.min_confidence}",
                metadata={
                    "aggregation": "weighted",
                    "original_action": best_action.value,
                    "scores": {k.value: v for k, v in action_scores.items()},
                },
            )

        # Get contributing strategies
        contributors = [
            f"{alloc.name}({sig.confidence:.0%})"
            for alloc, sig in signals
            if sig.action == best_action
        ]

        # Aggregate stop loss and take profit from contributing signals
        stop_losses = [
            sig.stop_loss for _, sig in signals
            if sig.action == best_action and sig.stop_loss
        ]
        take_profits = [
            sig.take_profit for _, sig in signals
            if sig.action == best_action and sig.take_profit
        ]

        return Signal(
            action=best_action,
            symbol=symbol,
            confidence=confidence,
            reason=f"Weighted ({confidence:.0%}): {', '.join(contributors)}",
            stop_loss=max(stop_losses) if stop_losses else None,  # Most conservative
            take_profit=min(take_profits) if take_profits else None,  # Most conservative
            metadata={
                "aggregation": "weighted",
                "num_strategies": len(signals),
                "scores": {k.value: v for k, v in action_scores.items()},
            },
        )

    def _aggregate_unanimous(
        self,
        signals: list[tuple[StrategyAllocation, Signal]],
        symbol: str,
    ) -> Signal:
        """All strategies must agree for buy/sell action."""
        actions = {sig.action for _, sig in signals}

        # If all agree on BUY or SELL
        if len(actions) == 1 and SignalAction.HOLD not in actions:
            action = signals[0][1].action
            avg_confidence = sum(sig.confidence for _, sig in signals) / len(signals)
            strategy_names = [alloc.name for alloc, _ in signals]

            return Signal(
                action=action,
                symbol=symbol,
                confidence=avg_confidence,
                reason=f"Unanimous {action.value}: {', '.join(strategy_names)}",
                metadata={
                    "aggregation": "unanimous",
                    "num_strategies": len(signals),
                },
            )

        # No unanimous agreement - hold
        action_list = [f"{alloc.name}={sig.action.value}" for alloc, sig in signals]
        return Signal(
            action=SignalAction.HOLD,
            symbol=symbol,
            confidence=0.0,
            reason=f"No consensus: {', '.join(action_list)}",
            metadata={
                "aggregation": "unanimous",
                "actions": [sig.action.value for _, sig in signals],
            },
        )

    def _aggregate_any(
        self,
        signals: list[tuple[StrategyAllocation, Signal]],
        symbol: str,
    ) -> Signal:
        """Any buy or sell signal triggers action (prioritize sell for safety)."""
        sells = [(a, s) for a, s in signals if s.action == SignalAction.SELL]
        buys = [(a, s) for a, s in signals if s.action == SignalAction.BUY]

        if sells:
            # Prioritize sell signals (risk management)
            best_sell = max(sells, key=lambda x: x[1].confidence)
            alloc, signal = best_sell
            signal.reason = f"Any (sell priority): {alloc.name}"
            signal.metadata["aggregation"] = "any"
            signal.metadata["num_strategies"] = len(signals)
            return signal

        if buys:
            best_buy = max(buys, key=lambda x: x[1].confidence)
            alloc, signal = best_buy
            signal.reason = f"Any: {alloc.name}"
            signal.metadata["aggregation"] = "any"
            signal.metadata["num_strategies"] = len(signals)
            return signal

        # All hold
        return Signal(
            action=SignalAction.HOLD,
            symbol=symbol,
            reason="All strategies holding",
            metadata={"aggregation": "any", "num_strategies": len(signals)},
        )

    def process_symbol(
        self,
        data: pd.DataFrame,
        symbol: str,
        position: Position | None = None,
    ) -> Signal:
        """Generate and aggregate signals for a symbol.

        This is the main entry point for processing a symbol with multiple strategies.

        Args:
            data: DataFrame with OHLCV data
            symbol: Symbol to process
            position: Current position, if any

        Returns:
            Aggregated signal for this symbol
        """
        signals = self.generate_signals(data, symbol, position)
        return self.aggregate_signals(signals, symbol)

    @property
    def strategy_names(self) -> list[str]:
        """Get names of all active strategies."""
        return [
            alloc.name
            for alloc in self.group.allocations
            if alloc.enabled
        ]

    @property
    def min_bars_required(self) -> int:
        """Get maximum min_bars_required across all strategies."""
        return max(
            (s.min_bars_required for s in self._strategies.values()),
            default=1,
        )


def create_strategy_group(
    strategies: list[str | dict[str, Any]],
    aggregation: str = "weighted",
    min_confidence: float = 0.5,
) -> StrategyGroup:
    """Create a strategy group from a simple configuration.

    Args:
        strategies: List of strategy names or dicts with config
            - String: strategy name with default params
            - Dict: {"name": str, "weight": float, "symbols": list, "params": dict}
        aggregation: Aggregation method name
        min_confidence: Minimum confidence threshold

    Returns:
        Configured StrategyGroup

    Example:
        group = create_strategy_group(
            strategies=[
                "sma",  # Default config
                {"name": "rsi", "weight": 0.5, "params": {"period": 21}},
                {"name": "momentum", "symbols": ["AAPL", "MSFT"]},
            ],
            aggregation="weighted",
        )
    """
    allocations = []

    for item in strategies:
        if isinstance(item, str):
            allocations.append(StrategyAllocation(name=item))
        elif isinstance(item, dict):
            allocations.append(
                StrategyAllocation(
                    name=item["name"],
                    weight=item.get("weight", 1.0),
                    symbols=item.get("symbols", []),
                    parameters=item.get("params", item.get("parameters", {})),
                    enabled=item.get("enabled", True),
                )
            )
        else:
            raise ValueError(f"Invalid strategy config: {item}")

    return StrategyGroup(
        allocations=allocations,
        aggregation=AggregationMethod(aggregation),
        min_confidence=min_confidence,
    )
