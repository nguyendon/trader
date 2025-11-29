---
name: alpaca-api
description: Expert guidance on using the Alpaca trading API (alpaca-py SDK). Covers authentication, market data fetching, order execution, position management, and paper vs live trading. Use when implementing broker integrations, fetching market data, or debugging Alpaca API issues.
---

# Alpaca API Integration

## Authentication

```python
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient

# Trading client (for orders, positions, account)
trading_client = TradingClient(
    api_key="APCA-API-KEY-ID",
    secret_key="APCA-API-SECRET-KEY",
    paper=True,  # Set False for live trading
)

# Data client (for market data)
data_client = StockHistoricalDataClient(
    api_key="APCA-API-KEY-ID",
    secret_key="APCA-API-SECRET-KEY",
)
```

## Environment Variables
```bash
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
ALPACA_PAPER=true  # or false for live
```

## Market Data

### Fetching Historical Bars
```python
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

request = StockBarsRequest(
    symbol_or_symbols=["AAPL", "MSFT"],
    timeframe=TimeFrame.Day,
    start=datetime(2024, 1, 1),
    end=datetime(2024, 12, 31),
)
bars = data_client.get_stock_bars(request)
```

### TimeFrame Options
- `TimeFrame.Minute` - 1 minute bars
- `TimeFrame.Hour` - 1 hour bars
- `TimeFrame.Day` - Daily bars
- `TimeFrame.Week` - Weekly bars
- `TimeFrame.Month` - Monthly bars

## Order Execution

### Market Order
```python
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

order = MarketOrderRequest(
    symbol="AAPL",
    qty=10,
    side=OrderSide.BUY,
    time_in_force=TimeInForce.DAY,
)
result = trading_client.submit_order(order)
```

### Limit Order
```python
from alpaca.trading.requests import LimitOrderRequest

order = LimitOrderRequest(
    symbol="AAPL",
    qty=10,
    side=OrderSide.BUY,
    time_in_force=TimeInForce.GTC,
    limit_price=150.00,
)
```

### Order Types
- `MarketOrderRequest` - Execute immediately at market price
- `LimitOrderRequest` - Execute at limit price or better
- `StopOrderRequest` - Trigger at stop price, execute as market
- `StopLimitOrderRequest` - Trigger at stop, execute as limit

### Time in Force
- `DAY` - Cancel at end of day
- `GTC` - Good til cancelled
- `IOC` - Immediate or cancel
- `FOK` - Fill or kill

## Position Management

```python
# Get all positions
positions = trading_client.get_all_positions()

# Get specific position
position = trading_client.get_open_position("AAPL")

# Close position
trading_client.close_position("AAPL")

# Close all positions
trading_client.close_all_positions()
```

## Account Info

```python
account = trading_client.get_account()
print(f"Equity: {account.equity}")
print(f"Cash: {account.cash}")
print(f"Buying Power: {account.buying_power}")
print(f"Day Trade Count: {account.daytrade_count}")
```

## Paper vs Live Trading

| Feature | Paper | Live |
|---------|-------|------|
| Real money | No | Yes |
| Market data | Same | Same |
| Order execution | Simulated | Real |
| PDT rules | Enforced | Enforced |

**Always test on paper first!**

## Common Issues

1. **PDT Rule**: 3+ day trades in 5 days requires $25k equity
2. **Market Hours**: Orders only fill during market hours (9:30-4:00 ET)
3. **Rate Limits**: 200 requests/minute for trading API
4. **Data Delays**: Free tier has 15-min delay on some data
