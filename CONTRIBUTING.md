# Contributing to sage-xgb

Thank you for contributing.

## Getting started

You need Linux on x86_64, Python 3.11 or newer, and uv 0.12.23. `pyproject.toml` requires that exact uv version.

```bash
git clone https://github.com/ontogr/sage-xgb.git
cd sage-xgb
make sync
uv run pre-commit install
```

`make sync` installs the locked dependencies, the dev group and all extras. Unit tests do not need external services. End-to-end tests run the stages on the committed sample and need no network. Only the download stage needs Kaggle credentials, and it reads them from a local `.env` file. Do not put credentials in Git.

## Make and test a change

1. Fork the repository and create a branch from `main`.
2. Make the change and add or update tests for it. Unit tests live under `tests/unit/`. End-to-end tests live under `tests/e2e/`. Integration tests live under `tests/integration/`.
3. While you work, run one test file or one test with pytest. For example:

   ```bash
   uv run pytest tests/e2e/test_features.py -v
   ```

   A focused run gives faster feedback. It does not replace the full suites.
4. Before you open a pull request, run the checks that match the change:

   ```bash
   make test
   make test-e2e
   make lint-check
   make lint-typing
   make lint-typos
   make docs-check
   ```

   `make lint-all` applies formatting and unsafe lint fixes, so it changes files. Use it only when you want those changes.
5. Open a pull request. Use a [conventional commit](https://www.conventionalcommits.org/en/v1.0.0/) prefix in the title: `fix:`, `feat:`, `build:`, `chore:`, `ci:`, `docs:`, `style:`, `refactor:`, `perf:` or `test:`.

## Tests

- Unit tests do not touch the network. Sockets are disabled by default.
- End-to-end tests run each stage on the committed sample in `tests/fixtures/amlworld_hi_small_sample/`. Tests do not read the `data/` folder.
- Integration tests call real external services, so they go in `tests/integration/`. `tests/conftest.py` applies the `integration`, `enable_socket` and flaky rerun markers to tests in that folder. Do not add those markers by hand.
- Leakage behavior has its own tests in `tests/e2e/`. A change to the split, the features or the GNN must keep those tests passing.
- Aim for 80 to 90 percent coverage on core logic. Do not write assertion-free tests to reach a number.

## Style

- Ruff handles formatting and linting. The line length is 88.
- `ty` checks types. Add type annotations to public signatures.
- Write Google-style docstrings on public functions and classes.
- Comments explain why the code does something. They do not describe the steps of a change.

## Documentation

The documentation is plain Markdown under `docs/`, in four folders: `tutorials/`, `how-to/`, `reference/` and `explanation/`. Keep each page to one type. The `README.md` file links to the pages.

Link to other pages with relative file paths. Run `make docs-check` before you open a pull request. It fails when a relative link points to a missing file, or when a `python` code block does not parse.

Do not add metric values, result tables or timings to the documentation. The repository ships the training pipeline. It does not ship a benchmark.

## Setup commands

| Command | Description |
| --- | --- |
| `make sync` | Install the locked dependencies, the dev group and all extras |
| `make test` | Run the unit tests with coverage |
| `make test-e2e` | Run the end-to-end stage tests on the committed sample |
| `make test-integration` | Run the integration tests, which need external services |
| `make lint-check` | Check formatting and lint without changing files |
| `make lint-typing` | Type check with `ty` |
| `make lint-typos` | Check for typos |
| `make lint-all` | Format, apply lint fixes, type check, check typos and audit the workflows |
| `make docs-check` | Check the links and the Python code blocks in the docs |
| `make security` | Run Bandit and the dependency audit |
| `make build` | Build the source distribution and the wheel into `dist/` |
| `make wheel-test` | Build the wheel and import it in a clean environment |
| `make docker-build` | Build the `train` image |
| `make help` | List all targets |

Run `uv run pre-commit install` once to enable the commit hooks.

## Releasing

1. In a pull request, raise `version` in `pyproject.toml` and run `uv lock` so that `uv.lock` records the new version.
2. Add a section for the new version to `CHANGELOG.md`, with the changes in plain prose.
3. Run `make build` and `make wheel-test`, and run `make docker-build`.
4. After the pull request merges, tag the merged commit as `v` followed by the version in `pyproject.toml`.

## Reporting bugs and requesting features

Open an [issue](https://github.com/ontogr/sage-xgb/issues). For security vulnerabilities, see [SECURITY.md](SECURITY.md).
