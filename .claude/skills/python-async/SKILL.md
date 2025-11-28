---
name: python-async
description: Expert guidance on async Python development including asyncio patterns, concurrent execution, error handling, testing async code, and performance optimization. Use when writing async functions, designing concurrent systems, debugging async issues, or optimizing I/O-bound operations.
---

# Async Python Development

## Core Patterns

### Basic Async Function
```python
async def fetch_data(symbol: str) -> pd.DataFrame:
    async with aiohttp.ClientSession() as session:
        async with session.get(f"/api/bars/{symbol}") as response:
            data = await response.json()
            return pd.DataFrame(data)
```

### Running Multiple Tasks Concurrently
```python
# Good - runs concurrently
async def fetch_multiple(symbols: list[str]) -> dict[str, pd.DataFrame]:
    tasks = [fetch_data(symbol) for symbol in symbols]
    results = await asyncio.gather(*tasks)
    return dict(zip(symbols, results))

# Bad - runs sequentially
async def fetch_sequential(symbols: list[str]):
    results = {}
    for symbol in symbols:
        results[symbol] = await fetch_data(symbol)  # Waits each time!
    return results
```

### Error Handling with gather
```python
async def fetch_with_errors(symbols: list[str]):
    tasks = [fetch_data(s) for s in symbols]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for symbol, result in zip(symbols, results):
        if isinstance(result, Exception):
            logger.error(f"Failed to fetch {symbol}: {result}")
        else:
            yield symbol, result
```

### Context Managers
```python
class AsyncBroker:
    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()

# Usage
async with AsyncBroker() as broker:
    await broker.submit_order(...)
```

### Async Iterators
```python
class DataStream:
    async def __aiter__(self):
        return self

    async def __anext__(self) -> Bar:
        data = await self._fetch_next()
        if data is None:
            raise StopAsyncIteration
        return data

# Usage
async for bar in data_stream:
    process(bar)
```

## Common Mistakes

### 1. Blocking the Event Loop
```python
# BAD - blocks event loop
async def bad_example():
    time.sleep(5)  # Blocks!
    requests.get(url)  # Blocks!

# GOOD - async alternatives
async def good_example():
    await asyncio.sleep(5)
    async with aiohttp.ClientSession() as session:
        await session.get(url)
```

### 2. Not Awaiting Coroutines
```python
# BAD - coroutine never runs
async def bad():
    fetch_data("AAPL")  # Warning: coroutine never awaited

# GOOD
async def good():
    await fetch_data("AAPL")
```

### 3. Creating Too Many Tasks
```python
# BAD - 10,000 concurrent connections
tasks = [fetch(url) for url in urls]  # len(urls) = 10,000
await asyncio.gather(*tasks)

# GOOD - limit concurrency
semaphore = asyncio.Semaphore(100)

async def limited_fetch(url):
    async with semaphore:
        return await fetch(url)

tasks = [limited_fetch(url) for url in urls]
await asyncio.gather(*tasks)
```

## Testing Async Code

### pytest-asyncio
```python
import pytest

@pytest.mark.asyncio
async def test_fetch_data():
    result = await fetch_data("AAPL")
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0

@pytest.mark.asyncio
async def test_broker_connection():
    async with AsyncBroker() as broker:
        assert await broker.is_connected()
```

### Mocking Async Functions
```python
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_with_mock():
    mock_fetch = AsyncMock(return_value=pd.DataFrame({'close': [100]}))

    with patch('trader.data.fetcher.fetch_data', mock_fetch):
        result = await some_function_that_fetches()
        mock_fetch.assert_called_once_with("AAPL")
```

## Performance Tips

1. **Use connection pooling** - reuse aiohttp sessions
2. **Batch requests** - combine multiple API calls when possible
3. **Use asyncio.wait_for** - add timeouts to prevent hanging
4. **Profile with asyncio debug mode** - `PYTHONASYNCIODEBUG=1`

## Entry Points

```python
# Script entry point
async def main():
    async with AsyncBroker() as broker:
        await run_strategy(broker)

if __name__ == "__main__":
    asyncio.run(main())
```
