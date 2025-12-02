"""Tests for multi-strategy support."""

from __future__ import annotations

import pandas as pd
import pytest

from trader.core.models import Position, Signal, SignalAction
from trader.strategies.base import BaseStrategy
from trader.strategies.multi import (
    AggregationMethod,
    MultiStrategyProcessor,
    StrategyAllocation,
    StrategyGroup,
    create_strategy_group,
)
from trader.strategies.registry import register_strategy


class MockBuyStrategy(BaseStrategy):
    """Always generates BUY signals."""

    def __init__(self, confidence: float = 1.0) -> None:
        self._confidence = confidence

    @property
    def name(self) -> str:
        return f"mock_buy_{self._confidence}"

    @property
    def min_bars_required(self) -> int:
        return 1

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.copy()

    def generate_signal(
        self,
        data: pd.DataFrame,
        symbol: str,
        position: Position | None = None,
    ) -> Signal:
        return self.buy_signal(symbol, "Always buy", confidence=self._confidence)


class MockSellStrategy(BaseStrategy):
    """Always generates SELL signals."""

    def __init__(self, confidence: float = 1.0) -> None:
        self._confidence = confidence

    @property
    def name(self) -> str:
        return f"mock_sell_{self._confidence}"

    @property
    def min_bars_required(self) -> int:
        return 1

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.copy()

    def generate_signal(
        self,
        data: pd.DataFrame,
        symbol: str,
        position: Position | None = None,
    ) -> Signal:
        return self.sell_signal(symbol, "Always sell", confidence=self._confidence)


class MockHoldStrategy(BaseStrategy):
    """Always generates HOLD signals."""

    @property
    def name(self) -> str:
        return "mock_hold"

    @property
    def min_bars_required(self) -> int:
        return 1

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.copy()

    def generate_signal(
        self,
        data: pd.DataFrame,
        symbol: str,
        position: Position | None = None,
    ) -> Signal:
        return self.hold_signal(symbol, "Always hold")


# Register mock strategies for testing
register_strategy("mock_buy", MockBuyStrategy)
register_strategy("mock_sell", MockSellStrategy)
register_strategy("mock_hold", MockHoldStrategy)


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create sample OHLCV data."""
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [1000, 1100, 1200],
        }
    )


class TestStrategyAllocation:
    """Tests for StrategyAllocation."""

    def test_create_allocation(self) -> None:
        """Test basic allocation creation."""
        alloc = StrategyAllocation(name="sma")
        assert alloc.name == "sma"
        assert alloc.weight == 1.0
        assert alloc.symbols == []
        assert alloc.parameters == {}
        assert alloc.enabled is True

    def test_allocation_with_params(self) -> None:
        """Test allocation with custom parameters."""
        alloc = StrategyAllocation(
            name="rsi",
            weight=0.5,
            symbols=["AAPL", "MSFT"],
            parameters={"period": 21},
            enabled=True,
        )
        assert alloc.weight == 0.5
        assert alloc.symbols == ["AAPL", "MSFT"]
        assert alloc.parameters == {"period": 21}

    def test_invalid_weight(self) -> None:
        """Test that invalid weights raise errors."""
        with pytest.raises(ValueError, match="Weight must be non-negative"):
            StrategyAllocation(name="sma", weight=-0.1)


class TestStrategyGroup:
    """Tests for StrategyGroup."""

    def test_create_group(self) -> None:
        """Test basic group creation."""
        allocs = [
            StrategyAllocation(name="mock_buy"),
            StrategyAllocation(name="mock_sell"),
        ]
        group = StrategyGroup(allocations=allocs)
        assert len(group.allocations) == 2
        assert group.aggregation == AggregationMethod.WEIGHTED

    def test_weight_normalization(self) -> None:
        """Test that weights are normalized to sum to 1."""
        allocs = [
            StrategyAllocation(name="mock_buy", weight=3.0),
            StrategyAllocation(name="mock_sell", weight=1.0),
        ]
        group = StrategyGroup(allocations=allocs)

        # Weights should be normalized to sum to 1
        total_weight = sum(a.weight for a in group.allocations)
        assert abs(total_weight - 1.0) < 0.01

        # Check relative weights are preserved (3:1 ratio)
        assert group.allocations[0].weight == pytest.approx(0.75, rel=0.01)
        assert group.allocations[1].weight == pytest.approx(0.25, rel=0.01)

    def test_empty_allocations_raises(self) -> None:
        """Test that empty allocations raise error."""
        with pytest.raises(ValueError, match="at least one allocation"):
            StrategyGroup(allocations=[])


class TestMultiStrategyProcessor:
    """Tests for MultiStrategyProcessor."""

    def test_initialize_strategies(self) -> None:
        """Test that strategies are initialized correctly."""
        group = create_strategy_group(["mock_buy", "mock_hold"])
        processor = MultiStrategyProcessor(group)

        assert len(processor._strategies) == 2
        assert "mock_buy" in processor._strategies
        assert "mock_hold" in processor._strategies

    def test_get_strategies_for_symbol(self) -> None:
        """Test getting strategies for a specific symbol."""
        allocs = [
            StrategyAllocation(name="mock_buy", symbols=["AAPL"]),
            StrategyAllocation(name="mock_sell", symbols=["MSFT"]),
            StrategyAllocation(name="mock_hold"),  # All symbols
        ]
        group = StrategyGroup(allocations=allocs)
        processor = MultiStrategyProcessor(group)

        # AAPL should get mock_buy and mock_hold
        aapl_strategies = processor.get_strategies_for_symbol("AAPL")
        strategy_names = [a.name for a, _ in aapl_strategies]
        assert "mock_buy" in strategy_names
        assert "mock_hold" in strategy_names
        assert "mock_sell" not in strategy_names

        # GOOGL should only get mock_hold (no specific strategies)
        googl_strategies = processor.get_strategies_for_symbol("GOOGL")
        assert len(googl_strategies) == 1
        assert googl_strategies[0][0].name == "mock_hold"

    def test_generate_signals(self, sample_data: pd.DataFrame) -> None:
        """Test generating signals from multiple strategies."""
        group = create_strategy_group(["mock_buy", "mock_hold"])
        processor = MultiStrategyProcessor(group)

        signals = processor.generate_signals(sample_data, "AAPL")

        assert len(signals) == 2
        actions = {s.action for _, s in signals}
        assert SignalAction.BUY in actions
        assert SignalAction.HOLD in actions

    def test_process_symbol(self, sample_data: pd.DataFrame) -> None:
        """Test end-to-end signal processing."""
        group = create_strategy_group(["mock_buy", "mock_hold"])
        processor = MultiStrategyProcessor(group)

        signal = processor.process_symbol(sample_data, "AAPL")

        # With weighted aggregation, BUY should win over HOLD
        assert signal.action == SignalAction.BUY
        assert signal.symbol == "AAPL"
        assert "aggregation" in signal.metadata


class TestAggregationMethods:
    """Tests for different aggregation methods."""

    def test_majority_all_agree(self, sample_data: pd.DataFrame) -> None:
        """Test majority when all strategies agree."""
        group = create_strategy_group(
            ["mock_buy", "mock_buy"],
            aggregation="majority",
        )
        processor = MultiStrategyProcessor(group)

        signal = processor.process_symbol(sample_data, "AAPL")

        assert signal.action == SignalAction.BUY
        assert signal.confidence == 1.0  # 100% agreement

    def test_majority_mixed(self, sample_data: pd.DataFrame) -> None:
        """Test majority with mixed signals."""
        group = create_strategy_group(
            ["mock_buy", "mock_buy", "mock_sell"],
            aggregation="majority",
        )
        processor = MultiStrategyProcessor(group)

        signal = processor.process_symbol(sample_data, "AAPL")

        assert signal.action == SignalAction.BUY
        assert abs(signal.confidence - 2 / 3) < 0.01  # 2/3 agreement

    def test_weighted_by_confidence(self, sample_data: pd.DataFrame) -> None:
        """Test weighted aggregation considers confidence."""
        # Register strategies with different confidence levels
        register_strategy("mock_buy_high", MockBuyStrategy, {"confidence": 0.9})
        register_strategy("mock_sell_low", MockSellStrategy, {"confidence": 0.3})

        group = create_strategy_group(
            [
                {"name": "mock_buy_high", "weight": 1.0},
                {"name": "mock_sell_low", "weight": 1.0},
            ],
            aggregation="weighted",
            min_confidence=0.3,  # Lower threshold to not block the signal
        )
        processor = MultiStrategyProcessor(group)

        signal = processor.process_symbol(sample_data, "AAPL")

        # Higher confidence BUY should win (0.9 vs 0.3)
        assert signal.action == SignalAction.BUY

    def test_unanimous_agreement(self, sample_data: pd.DataFrame) -> None:
        """Test unanimous when all agree."""
        group = create_strategy_group(
            ["mock_buy", "mock_buy"],
            aggregation="unanimous",
        )
        processor = MultiStrategyProcessor(group)

        signal = processor.process_symbol(sample_data, "AAPL")

        assert signal.action == SignalAction.BUY
        assert "unanimous" in signal.metadata.get("aggregation", "").lower()

    def test_unanimous_disagreement(self, sample_data: pd.DataFrame) -> None:
        """Test unanimous holds when there's disagreement."""
        group = create_strategy_group(
            ["mock_buy", "mock_sell"],
            aggregation="unanimous",
        )
        processor = MultiStrategyProcessor(group)

        signal = processor.process_symbol(sample_data, "AAPL")

        # Should HOLD when no consensus
        assert signal.action == SignalAction.HOLD

    def test_any_triggers_action(self, sample_data: pd.DataFrame) -> None:
        """Test 'any' triggers on first non-hold signal."""
        group = create_strategy_group(
            ["mock_hold", "mock_buy", "mock_hold"],
            aggregation="any",
        )
        processor = MultiStrategyProcessor(group)

        signal = processor.process_symbol(sample_data, "AAPL")

        assert signal.action == SignalAction.BUY

    def test_any_prioritizes_sell(self, sample_data: pd.DataFrame) -> None:
        """Test 'any' prioritizes sell for safety."""
        group = create_strategy_group(
            ["mock_buy", "mock_sell"],
            aggregation="any",
        )
        processor = MultiStrategyProcessor(group)

        signal = processor.process_symbol(sample_data, "AAPL")

        # SELL should take priority over BUY
        assert signal.action == SignalAction.SELL

    def test_first_uses_priority(self, sample_data: pd.DataFrame) -> None:
        """Test 'first' uses first strategy's signal."""
        group = create_strategy_group(
            ["mock_sell", "mock_buy", "mock_hold"],
            aggregation="first",
        )
        processor = MultiStrategyProcessor(group)

        signal = processor.process_symbol(sample_data, "AAPL")

        assert signal.action == SignalAction.SELL


class TestCreateStrategyGroup:
    """Tests for the create_strategy_group helper."""

    def test_simple_string_list(self) -> None:
        """Test creating group from simple strategy names."""
        group = create_strategy_group(["mock_buy", "mock_hold"])

        assert len(group.allocations) == 2
        assert group.allocations[0].name == "mock_buy"
        assert group.allocations[1].name == "mock_hold"

    def test_dict_config(self) -> None:
        """Test creating group from dict configs."""
        group = create_strategy_group(
            [
                {"name": "mock_buy", "weight": 0.6},
                {"name": "mock_sell", "weight": 0.4, "symbols": ["AAPL"]},
            ]
        )

        # Weights get normalized, so check relative ratio
        buy_alloc = group.allocations[0]
        sell_alloc = group.allocations[1]
        assert buy_alloc.weight > sell_alloc.weight
        assert sell_alloc.symbols == ["AAPL"]

    def test_mixed_config(self) -> None:
        """Test creating group from mixed string and dict."""
        group = create_strategy_group(
            [
                "mock_hold",
                {"name": "mock_buy", "params": {"confidence": 0.8}},
            ]
        )

        assert group.allocations[0].name == "mock_hold"
        assert group.allocations[1].parameters == {"confidence": 0.8}

    def test_custom_aggregation(self) -> None:
        """Test setting custom aggregation method."""
        group = create_strategy_group(
            ["mock_buy"],
            aggregation="unanimous",
            min_confidence=0.7,
        )

        assert group.aggregation == AggregationMethod.UNANIMOUS
        assert group.min_confidence == 0.7


class TestMinConfidenceThreshold:
    """Tests for minimum confidence threshold."""

    def test_below_threshold_holds(self, sample_data: pd.DataFrame) -> None:
        """Test that signals below threshold result in HOLD."""
        # Use a low confidence buy strategy
        register_strategy("mock_buy_very_low", MockBuyStrategy, {"confidence": 0.2})

        # Single low-confidence buy strategy with high threshold
        group = create_strategy_group(
            [{"name": "mock_buy_very_low"}],
            aggregation="weighted",
            min_confidence=0.5,  # Threshold higher than signal confidence
        )
        processor = MultiStrategyProcessor(group)

        signal = processor.process_symbol(sample_data, "AAPL")

        # Low confidence should result in HOLD with threshold message
        assert signal.action == SignalAction.HOLD
        assert (
            "threshold" in signal.reason.lower()
            or "confidence" in signal.reason.lower()
        )


class TestStrategyMetadata:
    """Tests for signal metadata in multi-strategy mode."""

    def test_signal_contains_strategy_info(self, sample_data: pd.DataFrame) -> None:
        """Test that individual signals contain strategy info."""
        group = create_strategy_group(["mock_buy"])
        processor = MultiStrategyProcessor(group)

        signals = processor.generate_signals(sample_data, "AAPL")

        assert len(signals) == 1
        _, signal = signals[0]
        assert "strategy" in signal.metadata
        assert signal.metadata["strategy"] == "mock_buy"

    def test_aggregated_signal_contains_metadata(
        self, sample_data: pd.DataFrame
    ) -> None:
        """Test that aggregated signal contains aggregation metadata."""
        group = create_strategy_group(["mock_buy", "mock_hold"])
        processor = MultiStrategyProcessor(group)

        signal = processor.process_symbol(sample_data, "AAPL")

        assert "aggregation" in signal.metadata
        assert "num_strategies" in signal.metadata
        assert signal.metadata["num_strategies"] == 2


class TestDisabledStrategies:
    """Tests for disabled strategies."""

    def test_disabled_strategy_ignored(self, sample_data: pd.DataFrame) -> None:
        """Test that disabled strategies are not used."""
        allocs = [
            StrategyAllocation(name="mock_buy", enabled=True),
            StrategyAllocation(name="mock_sell", enabled=False),
        ]
        group = StrategyGroup(allocations=allocs)
        processor = MultiStrategyProcessor(group)

        # Only mock_buy should be active
        assert len(processor._strategies) == 1
        assert "mock_buy" in processor._strategies

        signal = processor.process_symbol(sample_data, "AAPL")
        assert signal.action == SignalAction.BUY
