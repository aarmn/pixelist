# Default variables
python := "python"
uvx := "uvx"

# Default recipe (runs when `just` is called without arguments)
default:
    @just --list

# Clean build artifacts
clean:
    rm -rf dist/* build/* *.egg-info

# Run ruff linter
lint:
    ruff check .

# Format code with ruff
format:
    ruff format .

# Build package
build: clean
    {{uvx}} --from build pyproject-build.exe

# Upload to PyPI
upload: build
    {{uvx}} twine upload dist/* --verbose

# Install package in development mode
install:
    pip install -e ".[dev]"

# Run tests
test:
    pytest

# Full check - runs lint, test, and build
check: lint test build
    @echo "All checks passed!"

# Development setup
setup:
    pip install -U pip
    pip install ruff pytest build twine