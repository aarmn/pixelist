# Variables
python := "python"
current_version := `uvx --from toml-cli toml get --toml-path pyproject.toml project.version`

# Default recipe shows available commands
default:
    @just --list

# Clean all build and cache artifacts
clean:
    @echo "🧹 Cleaning build artifacts..."
    rm -rf dist/ build/ *.egg-info/ 
    rm -rf .coverage coverage.xml htmlcov/
    rm -rf .pytest_cache/ .ruff_cache/
    rm -rf .venv/ venv/

# Setup development environment
setup: clean
    @echo "🔧 Setting up development environment..."
    pip install -U pip
    pip install -e ".[all]"

# Install in development mode
install:
    @echo "📦 Installing package..."
    pip install -e ".[all]"

# Lint code with ruff
lint:
    @echo "🔍 Linting code..."
    ruff check .

# Format code with ruff
format:
    @echo "✨ Formatting code..."
    ruff format .

# Run tests with coverage
test:
    @echo "🧪 Running tests..."
    pytest --cov=pixelist --cov-report=xml --cov-report=html

# Build package
build: clean
    @echo "🏗️ Building package..."
    uvx --from build pyproject-build.exe

# Upload to PyPI
publish: ci
    @echo "🚀 Publishing to PyPI..."
    twine upload dist/*

# Version bumping commands
bump-patch:
    #!/usr/bin/env python
    major, minor, patch = map(int, '{{current_version}}'.split('.'))
    new_version = f"{major}.{minor}.{patch + 1}"
    print(f"🔖 Bumping patch to {new_version}")
    just _release new_version

bump-minor:
    #!/usr/bin/env python
    major, minor, patch = map(int, '{{current_version}}'.split('.'))
    new_version = f"{major}.{minor + 1}.0"
    print(f"🔖 Bumping minor to {new_version}")
    just _release new_version

bump-major:
    #!/usr/bin/env python
    major, minor, patch = map(int, '{{current_version}}'.split('.'))
    new_version = f"{major + 1}.0.0"
    print(f"🔖 Bumping major to {new_version}")
    just _release new_version

# Internal commands
_update-version NEW_VERSION:
    #!/usr/bin/env python
    import re
    with open('pyproject.toml', 'r') as f:
        content = f.read()
    new_content = re.sub(r'version = "[^"]*"', f'version = "{NEW_VERSION}"', content)
    with open('pyproject.toml', 'w') as f:
        f.write(new_content)

_release VERSION: ci
    @echo "📝 Preparing release v{{VERSION}}..."
    @just _update-version {{VERSION}}
    git add pyproject.toml
    git commit -m "🔖 Release v{{VERSION}}"
    git tag -a v{{VERSION}} -m "Release v{{VERSION}}"
    git push origin main v{{VERSION}}

# CI/CD commands
ci: lint test build
    @echo "✅ CI checks passed!"

# Test GitHub Actions locally
act:
    @echo "🔄 Testing GitHub Actions..."
    act -j test

# Full check - for pre-commit
check: format lint test build
    @echo "✨ All checks passed!"

# Create new branch
branch NAME:
    git checkout -b {{NAME}}
    git push -u origin {{NAME}}