# Default: list available commands
default:
    @just --list

# Install dependencies
install:
    uv sync --all-extras

# Run tests
test *ARGS:
    uv run pytest {{ARGS}}

# Run tests with coverage
test-cov *ARGS:
    uv run pytest --cov=src/trader --cov-report=term-missing {{ARGS}}

# Run linting
lint:
    uv run ruff check src/ tests/

# Fix linting issues
lint-fix:
    uv run ruff check --fix src/ tests/

# Format code
format:
    uv run ruff format src/ tests/

# Check formatting without changing
format-check:
    uv run ruff format --check src/ tests/

# Run type checking
typecheck:
    uv run mypy src/

# Run lint + typecheck + test
check: lint typecheck test

# Run format + lint + typecheck + test
all: format lint typecheck test

# Run the CLI
trader *ARGS:
    uv run trader {{ARGS}}

# Run backtest
backtest SYMBOL *ARGS:
    uv run trader backtest {{SYMBOL}} {{ARGS}}

# Run paper trading
paper *ARGS:
    uv run trader paper {{ARGS}}

# Clean up caches and venv
clean:
    rm -rf .venv .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
    find . -type d -name __pycache__ -exec rm -rf {} +
