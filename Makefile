.PHONY: install test lint format typecheck check all clean

# Install dependencies
install:
	uv sync --all-extras

# Run tests
test:
	uv run pytest

# Run tests with coverage
test-cov:
	uv run pytest --cov=src/trader --cov-report=term-missing

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

# Clean up
clean:
	rm -rf .venv .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
