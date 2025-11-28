---
name: trading-strategy
description: Expert guidance on designing, implementing, and testing algorithmic trading strategies. Covers backtesting best practices, performance metrics (Sharpe, Sortino, drawdown), risk management, position sizing, and avoiding common pitfalls like overfitting. Use when developing trading strategies, analyzing backtest results, implementing risk controls, or debugging strategy logic.
---

# Trading Strategy Development

## Core Principles

### Backtesting Best Practices
- Always use out-of-sample testing (train/test split)
- Account for transaction costs, slippage, and market impact
- Test across different market regimes (bull, bear, sideways)
- Beware of look-ahead bias - only use data available at decision time
- Avoid survivorship bias in historical data

### Performance Metrics

| Metric | Good | Great | Formula |
|--------|------|-------|---------|
| Sharpe Ratio | >1.0 | >2.0 | (Return - RiskFree) / StdDev |
| Sortino Ratio | >1.5 | >3.0 | (Return - RiskFree) / DownsideStdDev |
| Max Drawdown | <20% | <10% | Peak to trough decline |
| Win Rate | >50% | >60% | Winning trades / Total trades |
| Profit Factor | >1.5 | >2.0 | Gross profit / Gross loss |

### Position Sizing Rules
- Never risk more than 1-2% of portfolio on a single trade
- Scale position size based on volatility (ATR-based sizing)
- Reduce exposure during high correlation periods
- Use Kelly Criterion for optimal sizing (with fractional Kelly for safety)

### Common Pitfalls to Avoid
1. **Overfitting**: Too many parameters, curve-fitting to historical data
2. **Data snooping**: Testing many strategies until one works by chance
3. **Ignoring costs**: Transaction fees, slippage, bid-ask spread
4. **Survivorship bias**: Only testing on stocks that still exist
5. **Look-ahead bias**: Using future data in decisions

## Strategy Types

### Trend Following
- Moving average crossovers (SMA, EMA)
- Breakout strategies (Donchian channels)
- Momentum indicators (RSI, MACD)

### Mean Reversion
- Bollinger Band bounces
- RSI oversold/overbought
- Pairs trading (cointegration)

### Statistical Arbitrage
- Market-neutral strategies
- Factor-based investing
- Cross-sectional momentum

## Code Patterns

### Signal Generation
```python
@dataclass
class Signal:
    action: str  # 'buy', 'sell', 'hold'
    symbol: str
    quantity: float | None = None
    confidence: float = 1.0
    reason: str = ""
    stop_loss: float | None = None
    take_profit: float | None = None
```

### Indicator Caching
Calculate indicators once, not on every bar:
```python
def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data['sma_fast'] = data['close'].rolling(self.fast_period).mean()
    data['sma_slow'] = data['close'].rolling(self.slow_period).mean()
    return data
```

## Testing Strategies

1. **Unit test** signal generation logic with known inputs
2. **Backtest** on historical data with realistic assumptions
3. **Walk-forward** analysis for robustness
4. **Paper trade** for 2-4 weeks before live trading
5. **Start small** - scale up only after consistent results
