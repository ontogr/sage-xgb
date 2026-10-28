.PHONY: help sync test cov cov-report lint-typing lint-style lint-fmt lint-check lint-typos lint-actions lint-all security-bandit security-audit security build wheel-test clean

export UV_LOCKED = 1

help:
	@echo "Available make targets:"
	@echo "  make help             - Show this help"
	@echo "  make sync             - Sync project and install all dependencies"
	@echo "  make test             - Run unit tests with coverage"
	@echo "  make cov              - Run tests and generate coverage reports"
	@echo "  make cov-report       - Generate the HTML coverage report"
	@echo "  make lint-typing      - Type check with ty"
	@echo "  make lint-style       - Lint with ruff (check only)"
	@echo "  make lint-fmt         - Format code and lint with auto-fixes"
	@echo "  make lint-check       - Check formatting and lint without modifying files"
	@echo "  make lint-typos       - Check for typos"
	@echo "  make lint-actions     - Audit GitHub Actions workflows with zizmor"
	@echo "  make lint-all         - Run formatting, linting, type checking, typos and workflow audit"
	@echo "  make security-bandit  - Run Bandit security scan"
	@echo "  make security-audit   - Audit the locked dependencies with uv audit"
	@echo "  make security         - Run all security scans"
	@echo "  make build            - Build sdist and wheel into dist/"
	@echo "  make wheel-test       - Install the built wheel in a clean environment and import it"
	@echo "  make clean            - Remove build artifacts, caches and run outputs"

sync:
	uv sync --locked --all-groups --all-extras

test:
	uv run pytest tests/unit \
		--cov=sage_xgb \
		--cov-report=term-missing \
		--cov-report=xml \
		--junitxml=pytest-results.xml

cov-report:
	uv run coverage html

cov: test cov-report

lint-typing:
	uv run ty check sage_xgb/ tests

lint-style:
	uv run ruff check .

lint-fmt:
	uv run ruff format .
	uv run ruff check --fix --unsafe-fixes .

lint-check:
	uv run ruff format --check .
	uv run ruff check .

lint-typos:
	uv run typos

lint-actions:
	uvx zizmor .github/workflows/

lint-all: lint-fmt lint-typing lint-typos lint-actions

security-bandit:
	uv run bandit -c pyproject.toml -r sage_xgb/ --severity-level high --confidence-level high

security-audit:
	uv audit --preview-features audit-command

security: security-bandit security-audit

build:
	rm -rf dist
	uv build

# The wheel installs with --no-deps because resolving its dependencies from PyPI
# would pull CUDA builds of torch. The import check needs only the package itself.
wheel-test: build
	uv run twine check dist/*
	rm -rf .wheelenv
	uv venv .wheelenv
	uv pip install --python .wheelenv/bin/python --no-deps dist/*.whl
	cd /tmp && "$(CURDIR)/.wheelenv/bin/python" -c "import sage_xgb; print(sage_xgb.__version__)"

clean:
	rm -rf .coverage coverage.xml htmlcov dist build .wheelenv artifacts reports *.egg-info pytest-results.xml pytest-integration-results.xml pytest-e2e-results.xml
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -type d -name .ty_cache -exec rm -rf {} +
