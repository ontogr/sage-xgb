.PHONY: help sync test test-e2e test-integration cov cov-report lint-typing lint-style lint-fmt lint-check lint-typos lint-actions lint-all security-bandit security-audit security build wheel-test clean download sample train train-sample tune report mlflow-ui docker-build

export UV_LOCKED = 1

help:
	@echo "Available make targets:"
	@echo "  make help             - Show this help"
	@echo "  make sync             - Sync project and install all dependencies"
	@echo "  make test             - Run unit tests with coverage"
	@echo "  make test-e2e         - Run end-to-end stage tests on the committed sample"
	@echo "  make test-integration - Run integration tests (requires network and credentials)"
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
	@echo "  make download         - Download HI-Small from Kaggle (requires KAGGLE_* credentials)"
	@echo "  make sample           - Rebuild the committed sample from data/"
	@echo "  make train            - Run the full pipeline on HI-Small"
	@echo "  make train-sample     - Run the full pipeline on the committed sample"
	@echo "  make tune             - Tune hyperparameters on HI-Small"
	@echo "  make report           - Score the test and Tail splits once"
	@echo "  make mlflow-ui        - Open the MLflow UI on the local SQLite store"
	@echo "  make docker-build     - Build the train image"

sync:
	uv sync --locked --all-groups --all-extras

test:
	uv run pytest tests/unit \
		--cov=sage_xgb \
		--cov-report=term-missing \
		--cov-report=xml \
		--junitxml=pytest-results.xml

# Runs serially because the session fixtures run the pipeline once and share the
# result. The addopts override drops the xdist and rerun settings from pyproject.toml.
test-e2e:
	uv run pytest tests/e2e -v \
		-o "addopts=--strict-markers --strict-config --disable-socket --allow-unix-socket -ra" \
		--junitxml=pytest-e2e-results.xml

# Integration tests call real services, so sockets stay enabled.
test-integration:
	uv run pytest tests/integration -v \
		-o "addopts=--strict-markers --strict-config -ra" \
		--junitxml=pytest-integration-results.xml

cov-report:
	uv run coverage html

cov: test cov-report

lint-typing:
	uv run ty check sage_xgb/ scripts/ tests

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

download:
	uv run sage-xgb download --config configs/hi_small.yaml

sample:
	uv run python scripts/build_sample.py

train:
	uv run sage-xgb run --config configs/hi_small.yaml

train-sample:
	uv run sage-xgb run --config configs/sample.yaml

tune:
	uv run sage-xgb tune --config configs/hi_small.yaml

report:
	uv run sage-xgb report --config configs/hi_small.yaml

mlflow-ui:
	uv run mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db

docker-build:
	docker build --target train -t sage-xgb:train .
