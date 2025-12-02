"""Tests for transaction cost modeling."""

from __future__ import annotations

from decimal import Decimal

from trader.engine.costs import CostModel, CostReport


class TestCostModel:
    """Tests for CostModel."""

    def test_zero_cost_model(self) -> None:
        """Test zero cost model has no costs."""
        model = CostModel.zero_cost()

        cost = model.calculate_entry_cost(100.0, 100)
        assert cost == Decimal("0")

        cost = model.calculate_exit_cost(100.0, 100)
        assert cost == Decimal("0")

    def test_commission_per_trade(self) -> None:
        """Test fixed commission per trade."""
        model = CostModel(commission_per_trade=1.00)

        cost = model.calculate_entry_cost(100.0, 100)
        assert cost == Decimal("1.00")

        cost = model.calculate_entry_cost(200.0, 50)
        assert cost == Decimal("1.00")

    def test_commission_per_share(self) -> None:
        """Test per-share commission."""
        model = CostModel(commission_per_share=0.01)

        cost = model.calculate_entry_cost(100.0, 100)
        assert cost == Decimal("1.00")  # 100 shares * $0.01

        cost = model.calculate_entry_cost(100.0, 500)
        assert cost == Decimal("5.00")  # 500 shares * $0.01

    def test_spread_cost(self) -> None:
        """Test bid-ask spread cost."""
        model = CostModel(spread_pct=0.001)  # 10 bps spread

        # Entry pays half the spread
        cost = model.calculate_entry_cost(100.0, 100)
        # Trade value = 10000, half spread = 10000 * 0.001 / 2 = 5
        assert cost == Decimal("5.00")

    def test_slippage_cost(self) -> None:
        """Test slippage cost."""
        model = CostModel(slippage_pct=0.0005)  # 5 bps slippage

        cost = model.calculate_entry_cost(100.0, 100)
        # Trade value = 10000, slippage = 10000 * 0.0005 = 5
        assert cost == Decimal("5.00")

    def test_combined_costs(self) -> None:
        """Test multiple cost components combined."""
        model = CostModel(
            commission_per_trade=1.00,
            commission_per_share=0.005,
            spread_pct=0.001,
            slippage_pct=0.0005,
        )

        cost = model.calculate_entry_cost(100.0, 100)
        # Commission: 1.00
        # Per share: 100 * 0.005 = 0.50
        # Spread: 10000 * 0.001 / 2 = 5.00
        # Slippage: 10000 * 0.0005 = 5.00
        # Total: 11.50
        assert cost == Decimal("11.50")

    def test_round_trip_cost(self) -> None:
        """Test round-trip cost calculation."""
        model = CostModel(commission_per_trade=1.00)

        cost = model.calculate_round_trip_cost(100.0, 105.0, 100)
        # Entry: 1.00
        # Exit: 1.00
        # Total: 2.00
        assert cost == Decimal("2.00")

    def test_effective_entry_price_buy(self) -> None:
        """Test effective entry price for buy orders."""
        model = CostModel(spread_pct=0.001)  # 10 bps

        effective = model.get_effective_entry_price(100.0, 100, "buy")
        # Cost = 5.00 for 100 shares = 0.05 per share
        # Effective = 100.00 + 0.05 = 100.05
        assert effective == Decimal("100.05")

    def test_effective_entry_price_sell(self) -> None:
        """Test effective entry price for short sells."""
        model = CostModel(spread_pct=0.001)

        effective = model.get_effective_entry_price(100.0, 100, "sell")
        # For shorts, effective price is lower
        assert effective == Decimal("99.95")

    def test_effective_exit_price_sell(self) -> None:
        """Test effective exit price for closing longs."""
        model = CostModel(spread_pct=0.001)

        effective = model.get_effective_exit_price(100.0, 100, "sell")
        # Selling gets you less than market price
        assert effective == Decimal("99.95")

    def test_effective_exit_price_buy(self) -> None:
        """Test effective exit price for covering shorts."""
        model = CostModel(spread_pct=0.001)

        effective = model.get_effective_exit_price(100.0, 100, "buy")
        # Covering shorts costs more
        assert effective == Decimal("100.05")

    def test_retail_investor_preset(self) -> None:
        """Test retail investor cost preset."""
        model = CostModel.retail_investor()

        assert model.commission_per_trade == 0.0
        assert model.spread_pct == 0.0005  # 5 bps
        assert model.slippage_pct == 0.0002  # 2 bps

    def test_active_trader_preset(self) -> None:
        """Test active trader cost preset."""
        model = CostModel.active_trader()

        assert model.commission_per_trade == 1.00
        assert model.spread_pct == 0.0003  # 3 bps
        assert model.market_impact_pct == 0.0001  # 1 bp

    def test_institutional_preset(self) -> None:
        """Test institutional cost preset."""
        model = CostModel.institutional(avg_daily_volume=1_000_000)

        assert model.commission_per_trade == 0.50
        assert model.commission_per_share == 0.001
        assert model.avg_daily_volume == 1_000_000

    def test_high_frequency_preset(self) -> None:
        """Test HFT cost preset (with rebates)."""
        model = CostModel.high_frequency()

        # HFT gets rebates (negative commission)
        assert model.commission_per_trade < 0
        assert model.spread_pct == 0.0

    def test_summary(self) -> None:
        """Test cost model summary."""
        model = CostModel(
            commission_per_trade=1.00,
            spread_pct=0.0005,
            slippage_pct=0.0002,
        )

        summary = model.summary()

        assert summary["commission_per_trade"] == 1.00
        assert summary["spread_pct_bps"] == 5.0
        assert summary["slippage_pct_bps"] == 2.0

    def test_market_impact_scaling(self) -> None:
        """Test market impact scales with trade size."""
        model = CostModel(market_impact_pct=0.001)  # 10 bps

        # Larger trades have more impact
        small_cost = model.calculate_entry_cost(100.0, 100)  # $10k trade
        large_cost = model.calculate_entry_cost(100.0, 1000)  # $100k trade

        # Large trade should have more than linear impact due to sqrt scaling
        assert large_cost > small_cost


class TestCostReport:
    """Tests for CostReport."""

    def test_total_costs(self) -> None:
        """Test total costs calculation."""
        report = CostReport(
            total_commission=Decimal("10.00"),
            total_spread_cost=Decimal("25.00"),
            total_slippage=Decimal("15.00"),
            total_market_impact=Decimal("5.00"),
        )

        assert report.total_costs == Decimal("55.00")

    def test_cost_per_trade(self) -> None:
        """Test average cost per trade."""
        report = CostReport(
            total_commission=Decimal("100.00"),
            num_trades=10,
        )

        assert report.cost_per_trade == Decimal("10.00")

    def test_cost_per_trade_zero_trades(self) -> None:
        """Test cost per trade with zero trades."""
        report = CostReport(num_trades=0)
        assert report.cost_per_trade == Decimal("0")

    def test_cost_as_pct_of_gross(self) -> None:
        """Test costs as percentage of gross P&L."""
        report = CostReport(
            total_commission=Decimal("100.00"),
            gross_pnl=Decimal("1000.00"),
        )

        assert report.cost_as_pct_of_gross == 10.0

    def test_cost_as_pct_of_gross_zero_pnl(self) -> None:
        """Test cost percentage with zero P&L."""
        report = CostReport(
            total_commission=Decimal("100.00"),
            gross_pnl=Decimal("0"),
        )

        assert report.cost_as_pct_of_gross == 0.0

    def test_summary(self) -> None:
        """Test cost report summary."""
        report = CostReport(
            total_commission=Decimal("50.00"),
            total_spread_cost=Decimal("30.00"),
            total_slippage=Decimal("20.00"),
            num_trades=5,
            gross_pnl=Decimal("500.00"),
            net_pnl=Decimal("400.00"),
        )

        summary = report.summary()

        assert summary["total_commission"] == 50.00
        assert summary["total_costs"] == 100.00
        assert summary["num_trades"] == 5
        assert summary["cost_per_trade"] == 20.00
