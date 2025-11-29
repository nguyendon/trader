---
name: pydantic-patterns
description: Best practices for Pydantic v2 models, validation, settings management, and serialization. Use when creating data models, config classes, API schemas, or debugging validation errors.
---

# Pydantic Patterns (v2)

## Basic Model

```python
from pydantic import BaseModel, Field
from decimal import Decimal

class Order(BaseModel):
    symbol: str
    quantity: int = Field(gt=0)
    price: Decimal | None = None
    side: str = Field(pattern="^(buy|sell)$")
```

## Validation

### Field Constraints
```python
from pydantic import Field

class Trade(BaseModel):
    symbol: str = Field(min_length=1, max_length=10)
    quantity: int = Field(gt=0, le=10000)
    price: Decimal = Field(ge=0, decimal_places=2)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
```

### Custom Validators
```python
from pydantic import field_validator, model_validator

class Order(BaseModel):
    symbol: str
    limit_price: Decimal | None = None
    order_type: str

    @field_validator("symbol")
    @classmethod
    def uppercase_symbol(cls, v: str) -> str:
        return v.upper()

    @model_validator(mode="after")
    def check_limit_price(self) -> "Order":
        if self.order_type == "limit" and self.limit_price is None:
            raise ValueError("Limit orders require a price")
        return self
```

## Settings Management

```python
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # Required
    alpaca_api_key: SecretStr
    alpaca_secret_key: SecretStr

    # Optional with defaults
    alpaca_paper: bool = True
    max_position_pct: float = 0.10

    @property
    def has_credentials(self) -> bool:
        return bool(
            self.alpaca_api_key.get_secret_value()
            and self.alpaca_secret_key.get_secret_value()
        )
```

### Accessing Secrets
```python
settings = Settings()
# Wrong: print(settings.alpaca_api_key)  # Shows SecretStr('**********')
# Right:
api_key = settings.alpaca_api_key.get_secret_value()
```

## Serialization

### To Dict/JSON
```python
model = Order(symbol="AAPL", quantity=10)

# To dict
model.model_dump()
model.model_dump(exclude_none=True)
model.model_dump(by_alias=True)

# To JSON
model.model_dump_json()
```

### From Dict/JSON
```python
Order.model_validate({"symbol": "AAPL", "quantity": 10})
Order.model_validate_json('{"symbol": "AAPL", "quantity": 10}')
```

## Common Patterns

### Optional with Default Factory
```python
from pydantic import Field

class Config(BaseModel):
    symbols: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
```

### Immutable Models
```python
class ImmutableTrade(BaseModel):
    model_config = {"frozen": True}

    symbol: str
    price: Decimal
```

### Aliases (for API compatibility)
```python
class ApiResponse(BaseModel):
    order_id: str = Field(alias="orderId")
    filled_qty: int = Field(alias="filledQty")

# Parse with aliases
ApiResponse.model_validate({"orderId": "123", "filledQty": 10})
```

## Type Hints

```python
from typing import Literal
from decimal import Decimal

class Signal(BaseModel):
    action: Literal["buy", "sell", "hold"]
    symbol: str
    quantity: int | None = None
    price: Decimal | None = None
```

## Error Handling

```python
from pydantic import ValidationError

try:
    order = Order(symbol="", quantity=-1)
except ValidationError as e:
    print(e.errors())
    # [{'type': 'string_too_short', 'loc': ('symbol',), ...},
    #  {'type': 'greater_than', 'loc': ('quantity',), ...}]
```
