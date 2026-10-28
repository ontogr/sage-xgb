# AGENTS.md

## Project

**`sage_xgb`** detects money-laundering transactions on the IBM AMLworld dataset by feeding GraphSAGE graph neural network output into an XGBoost classifier. The first milestone covers the HI-Small Variant.

Published to PyPI as `sage-xgb`; imported as `sage_xgb`.

**Layout**: flat. The package is `sage_xgb/` at the repo root and everything outside `sage_xgb/` (tests, data) is not part of the distribution.

- Python package: `sage_xgb`.
- License: Apache-2.0 for the code. The committed sample under `tests/fixtures/amlworld_hi_small_sample/` is redistributed AMLworld data under CDLA-Sharing-1.0 and has its own LICENSE file.
- Python: `>=3.11`.
- Package manager and command runner: `uv`.

## Pre-1.0 policy — breaking changes expected

Project is pre-1.0 with no stability promise. Breaking changes are allowed
and recommended if they produce better results. This section overrides any
stability or compatibility rule in this file or in a skill.

Default to a hard cut. Do not add backward-compat aliases, deprecated shims,
fallback parsing, migration bridges, or tests for the previous API unless the
user explicitly asks. Remove the old surface and write tests only for the
current behavior, as if the old API never existed.

Do not add backwards-compatibility shims, deprecation comments, or feature
flags for changes that have not been released yet. Just edit the code.

## Skills

Task-specific rules live in project skills under `.claude/skills/`. Use the
matching skill before you start that kind of work. An agent without skill
support can open `.claude/skills/<name>/SKILL.md` directly.

| When you are doing... | Use |
| --- | --- |
| Drafting or revising prose, including replies, instruction files, docs, comments, docstrings, commits, and PRs | `writing-plain` plus the relevant task skill |
| Adding or changing tests | `writing-tests` |
| Adding or updating docstrings or comments | `writing-docstrings` |
| Writing docs pages or user-facing Markdown | `writing-diataxis-docs` |
| Fixing type-check failures from `ty` | `fixing-ty-errors` |

The Pre-1.0 policy above overrides every skill. Where this file and a skill
overlap, the skill is authoritative. More specific task instructions control
implementation details.

For prose style, `writing-plain` is authoritative. Its docwriter rules take precedence over its STE guidance. Task skills govern technical content, structure, and required formatting, not competing prose styles. Preserve technical meaning, uncertainty, and requirement strength, and apply the shared skill's scope and format exceptions.

## Repository structure

This is a flat-layout Python repository. Do not introduce a `src/` directory.

```text
sage_xgb/               # package
configs/                # run configuration, one YAML file per dataset Variant
scripts/                # dev tools, such as the sample builder; not in the wheel
docs/                   # user-facing docs: tutorials, how-to, reference, explanation
tests/
  unit/                 # unit tests, no network
  e2e/                  # per-stage end-to-end tests on the committed sample
  integration/          # integration tests with real external services
  fixtures/             # committed test data, including amlworld_hi_small_sample/
data/                   # raw AMLworld files, gitignored
```

## Critical rules

- Do not add any documentation or comments that would not reside in a production codebase. Comments should be developer/user friendly not intermediate findings when performing a task.
- Do not include comments referring to internal plans, notes, or comments written by agents while performing a task.
- Keep the flat layout: `sage_xgb/` lives at the repository root. Do not add `src/`.
- Use `uv run` for Python commands and Makefile targets for standard checks.
- Preserve public APIs within a release; the Pre-1.0 policy above governs breaking changes.
- Inject dependencies through constructors or function parameters; do not add a dependency-injection framework.
- Match existing patterns before introducing new abstractions.
- Prefer small, surgical edits. Do not refactor adjacent code opportunistically.
- Verify changed behavior with an end-to-end or integration test, following the Testing requirements below.
- Do not commit secrets, `.env`, API keys, tokens, or credentials.
- Do not put patient, customer, partner, or private organization names in code, docs, issues, PRs, commit messages, or tests unless they are already public fixture data approved for that use.
- Do not commit real AMLworld data except the committed sample under `tests/fixtures/amlworld_hi_small_sample/`. Keep that sample's LICENSE file, and do not add Apache-2.0 headers to its files. Never copy files from `data/` into the repo.
- Do not commit unless the user explicitly asks.

ML correctness rules:

- Split: the train, validation, and test split is defined in exactly one module and nowhere else. It is day-aligned 6/2/2 (HI-Small: train Sep 1-6, validation Sep 7-8, test Sep 9-10). Other Variants need their own day boundaries. Exclude the Tail from headline metrics and report it as a separate slice. Open the test set once, for the final report only. Never tune, early-stop, or choose thresholds on test.
- Ordering: the raw transactions CSV is not chronological. Sort transactions with a stable sort on timestamp, with the raw row id as the tiebreak.
- Account identity: an Account is the pair (bank, account number). The account number alone is not unique, so never key on it alone.
- Patterns and labels: the Patterns file is for evaluation and analysis only. Never use it as a feature source. Labels never enter features.
- No leakage or train/serve skew: do not compute features over a whole split or the whole file, and do not feed in-sample model outputs into a downstream model.
- Reference code: rewrite code adapted from reference repos to follow these rules, and do not carry over leakage or skew from it. Do not add attribution comments, headers, or notices that name reference repos in source files.

## Development commands

Use `uv` for environment and command execution. Standard commands are exposed as
Makefile targets:

```bash
make sync                  # install all dependencies with uv sync
make test                  # run unit tests with coverage
make test-e2e              # run end-to-end stage tests on the committed sample
make test-integration      # run integration tests requiring external services
make lint-fmt              # format and auto-fix with Ruff
make lint-check            # check formatting and lint, no modifications
make lint-typing           # type check with ty
make lint-typos            # check for typos
make lint-all              # format + lint + type check + typos; applies unsafe fixes
make security              # run the security checks
make build                 # build the distribution packages
make wheel-test            # build the wheel and import it in a clean environment
make sample                # rebuild the committed sample from data/
make train                 # run the full pipeline on HI-Small
make train-sample          # run the full pipeline on the committed sample
make tune                  # tune hyperparameters on HI-Small
make report                # score the test and Tail splits once
make docs-check            # check doc links and Python code blocks
make clean                 # remove build artifacts and caches
```

Run focused tests while iterating:

```bash
uv run pytest tests/unit/<area>/test_<feature>.py::Test<Feature> -v
uv run pytest tests/integration/<area>/test_<feature>.py -v
```

If a tool such as `uv`,
`pytest`, `ruff`, or `ty` is missing, check for the repo `.venv`. If present,
activate it and retry. If no environment exists, ask before installing or using
alternate tooling.

## Coding standards

- Use DDD and Clean Architecture principles where the project already follows
  them.
- Default to functions and modules. Use classes only when there is genuine
  stateful behavior, shared `self` state, or a data container with behavior.
- Do not use classes only for namespacing, single-method wrappers, or grouping
  pure functions.
- Do not add abstractions for single-use code or hypothetical future needs.
  If you catch yourself adding "for future extensibility", stop. Ship the
  smallest code that satisfies this task; add configurability only when a
  second caller exists in this repo.
- Use `snake_case` for functions and variables, `PascalCase` for classes, and
  `test_*.py` for tests.
- Prefer descriptive names. Avoid abbreviations such as `ver` when `version` is
  clearer, or `rp` when `requires_python` is clearer.
- Prefer top-level imports. Use local imports only when needed to avoid cycles,
  optional dependency import costs, or test-specific patching patterns already
  used nearby.
- Add type hints to new and modified public APIs.
- Prefer `TypedDict` over broad `dict`/`Mapping` when documenting structured
  dictionaries in code.
- Keep state explicit. Do not dynamically add/read fields with `setattr` and
  `getattr` when normal attributes or models would be clearer.
- Implement relevant special methods such as `__repr__` or `__str__` when they
  improve debugging or match nearby domain models.
- Keep this project's own author attribution in module docstrings if present. This does not apply to reference repos (see the reference code rule).
- Prefer composition over inheritance for new code unless an existing framework
  or local interface expects inheritance.
- Use early returns to avoid deep nesting.
- Keep public APIs minimal and well-defined; do not expose internals.
- Prefix internal helpers with `_` when they carry no stability guarantee.
- Define `__all__` only in package `__init__.py` files that intentionally
  re-export public symbols. Do not put `__all__` in implementation modules.
- Do not access private members (`_`-prefixed attributes or methods) of other
  classes. Use public APIs or add an explicit interface when needed.
- Place module-level helper functions near the top of a file after imports.
  Within classes, place private/helper methods after public methods when that
  matches the surrounding file.

### Public API stability

The Pre-1.0 policy above controls. Within a release, public APIs are
contracts. Before changing a public function, class, constructor,
or exported symbol:

1. Check whether it is exported from a package `__init__.py`.
2. Search existing tests, examples, docs, and call sites.
3. Preserve function names, parameter names, positional argument behavior, and
   return shape when possible.
4. Add new parameters as keyword-only when practical:
   `*, new_parameter: str = "default"`.
5. Warn the user if a requested change is breaking or may require migration.

Ask: would this break code that used the package last week?

## Architecture and component boundaries

- Settings classes subclass `pydantic_settings.BaseSettings`, hold credentials
  and service URIs such as Kaggle credentials and MLflow URIs, load from the
  repo-root `.env`, and use clear environment prefixes such as `KAGGLE_` and
  `MLFLOW_`. Put run settings in the run config instead.
- Run configuration lives in one YAML file per dataset Variant. Pydantic models
  validate the file at startup. Override a value from the file with
  `--set key=value` on the command line.
- The `sage-xgb` CLI uses argparse, with one subcommand per pipeline stage.
- Wrap storage, external services, network I/O, and message queues when a
  project-native boundary is useful. Do not wrap stable utility or ML libraries
  merely to hide imports.
- Use monkeypatching/mocking for OS calls, time, third-party SDKs, and test-only
  seams. Add explicit injection seams when tests would otherwise patch project
  internals.
- Prefer dependency injection over monkeypatching project class attributes in
  tests. Monkeypatching third-party SDK boundaries is acceptable when it is the
  narrowest reliable test seam.

## Adding project components

Follow nearby components first.

### Pipeline stages

- Write each pipeline stage as a plain function and expose it as a CLI
  subcommand.
- Read and write parquet artifacts at paths taken from the run config.
- Keep stage logic pure where possible.
- Give every stage an end-to-end test that runs the CLI stage on the committed
  sample and checks the artifacts it writes.

## Testing requirements

Use the `writing-tests` skill before adding or changing tests.

- Never write unit tests after you write code.
- Highly prefer E2E or integration tests as the sole testing mechanism. Use them to verify complex features work. At the end of those tests, produce a verifiable and repeatable artifact. An E2E test runs a CLI stage on the committed sample and checks the artifacts it writes.
- Per-stage end-to-end tests live in `tests/e2e/`, and `make test-e2e` runs them serially with sockets disabled and no reruns.
- If you must test a system in isolation, first write down all the ways it could fail, then write the code.
- A test that breaks under a behavior-preserving refactor is asserting implementation, not behavior. Do not add it.
- Never delete or weaken a failing test to make the suite pass. Fix the code, or ask.
- Do not add tests that restate the code, assert mock call shapes, grep source or docs text, compute the expected value with the code under test, or only keep a test-only export or seam alive.
- Use `pytest`. Add `pytest-asyncio` only if an async component appears.
- Tests mirror the `sage_xgb/` package under `tests/unit/` and
  `tests/integration/`.
- Use one `Test<ClassName>` class per `test_<class>.py` file when practical.
- Use descriptive behavior names such as `test_returns_empty_when_no_results`.
- Use Arrange-Act-Assert structure.
- Test public interfaces and documented behavior, not private implementation
  details. Avoid accessing private members in tests unless no public seam exists.
- For tests over multiple inputs, use `@pytest.mark.parametrize`.
- Always read and copy the style of similar nearby tests before adding fixtures
  or helpers.
- For bug fixes, extend the E2E or integration test that covers the behavior so
  it fails without the fix. Add a unit test only when the rules above allow it.
- One focused test that fails without the fix is better than many shallow tests
  that do not prove behavior.
- Do not write assertion-free tests just to increase coverage.
- Target 80-90% coverage for core/domain logic; do not chase 100% coverage.
- Do not delete, skip, or weaken a failing test to hide the failure. Use `skip`
  or `xfail` only for a known, documented reason the user has accepted.
- Use only the committed sample under `tests/fixtures/amlworld_hi_small_sample/` as test data. Never read `data/` from tests.
- Leakage tests are a first-class test category. For example, a test drops every transaction after time `t`, recomputes the features, and checks that the features at `t` do not change.

### Unit tests

Unit tests are the exception. Write one only for isolated logic with real branches that an E2E or integration test cannot reach, and only under the rules above.

- Location: `tests/unit/...`.
- No network access. Sockets are disabled by default.
- Mock only external services, such as Kaggle and the MLflow server, at the client or API boundary.
- If a unit test raises `SocketBlockedError`, fix the mock; do not enable
  network access.
- Do not add manual `enable_socket` markers to unit tests.

### Integration tests

- Location: `tests/integration/...`.
- Use real external services only when integration behavior is under test.
- `integration`, `enable_socket`, and flaky rerun markers are auto-applied by
  `tests/conftest.py`; do not add them manually.
- Integration tests can rely on real environment variables or `.env`; do not
  commit secrets.

## Type checking

This project uses `ty`. Use the `fixing-ty-errors` skill when fixing type
errors.

- Run focused checks with `uv run ty check <target>` while iterating.
- Run `make lint-typing` for the repository typing target.
- Use narrowing (`isinstance`, `if x is None: raise`, `hasattr`) before casts.
- Never use `assert` for type narrowing.
- Use `cast()` only after structural validation when the checker cannot infer
  the type.
- Use `# type: ignore[code]` only for genuine third-party stub defects, and never
  as a first resort.
- Do not add helper abstractions solely to satisfy the type checker.

## Writing style

Use the `writing-plain` skill before drafting or revising prose. Write literal explanations with familiar words and precise technical terms. Keep necessary facts and qualifications. The shared skill combines plain-writing rules with compatible STE guidance; it does not claim ASD-STE100 compliance.

## Docstrings and comments

Use the `writing-docstrings` skill before adding or changing docstrings or
comments.

Google style governs docstring structure and formatting; `writing-plain` governs prose.

- Use Google-style docstrings with triple double quotes.
- Start every docstring with a one-line summary ending in punctuation.
- Keep docstring summary lines and comment lines within the 88-character limit;
  rewrite concisely rather than awkwardly wrapping.
- Public APIs, nontrivial functions, classes, exceptions, properties, settings,
  and test modules need useful docstrings.
- Docstrings document caller-visible behavior, edge cases, constraints, return
  values, raised interface exceptions, and side effects.
- Types belong in annotations, not repeated in docstrings unless annotations are
  insufficient to describe units, shapes, or accepted values.
- Inline comments explain why, not what. Do not add comments that restate code.
- Add comments only when they explain non-obvious business logic,
  domain-specific constraints, safety concerns, algorithms, performance
  tradeoffs, or library workarounds. Prefer clearer code over comments for
  ordinary control flow.
- Do not add AI-progress notes, implementation transcripts, or comments saying
  something now works.
- Do not put comments at the end of code lines; place them above the relevant
  line or block.
- Newly added code comments must be ASCII only.

## Documentation

Use the `writing-diataxis-docs` skill before writing user-facing docs.

- Update relevant docs when code behavior changes.
- The 88-character line limit is a Python rule (Ruff line length, docstrings,
  and comments). Do not hard-wrap prose in Markdown docs pages, `AGENTS.md`,
  `CLAUDE.md`, or PR/issue descriptions to 88 characters; write normal,
  unwrapped prose lines and let the renderer or editor soft-wrap.
- Keep tutorials, how-to guides, reference, and explanation content separated
  where practical.
- Link between documentation modes instead of mixing them into one page.
- When documenting research-based implementations, link the papers they follow.
  Docs may cite papers, but source-code comments must not attribute reference
  repos.
- Do not create new docs files unless the user requests them or the behavior
  change clearly requires user-facing documentation.

## Tooling, dependencies, and security

- Use `uv` for package management, virtual environments, and lockfiles.
- Keep dependency specs loose in `pyproject.toml`; rely on `uv.lock` for
  reproducibility.
- Add upper bounds only for known-breaking dependencies.
- Install PyTorch from the explicit `pytorch-cpu` index through
  `[tool.uv.sources]` for local installs, CI, and Docker. Never add CUDA builds.
- Bound torch to `>=2.14,<2.15`, the minor version of the `pyg-lib` 0.9 CPU wheels. Those wheels are tied to a torch minor version, so move both together.
- Do not add dependencies unless strictly required; justify any new dependency.
- Run `deptry` or an equivalent check before removing a dependency, and account
  for dynamic imports and plugin entry points.
- Use Ruff for linting and formatting. Line length is 88.
- Do not change Ruff, typing, pytest, or CI configuration unless explicitly
  asked.
- In CI or installation scripts, never pipe remote scripts into a shell. Download
  artifacts to files, pin external tools to explicit versions, and verify
  checksums before installation.
- Use configured ignores for generated or unowned code instead of manual edits.
- Treat Bandit findings as review items unless CI makes them blocking.
- Never commit `.env`, API keys, tokens, credentials, or local secrets.

## Workflow expectations

- If the request is ambiguous and multiple valid interpretations materially
  change the solution, ask one narrow clarifying question.
- Ask before changing public API shape, persistence, dependencies, or config
  keys. Otherwise make a reasonable assumption and proceed.
- Read `GLOSSARY.md` before changing domain terms or settled behavior.
- State assumptions when they affect the implementation. Surface tradeoffs and
  push back when a simpler or safer approach exists.
- Before changing code, inspect enough nearby code and tests to follow existing
  patterns.
- Do not assume existing code is always the right pattern. Follow good local
  patterns, but call out code smells or risky patterns when they affect the
  requested change.
- Keep every changed line traceable to the requested behavior.
- Do not add unrelated features, configuration, abstractions, or defensive code.
- If a solution starts becoming much larger than the problem requires, stop and
  simplify before continuing.
- Remove imports, variables, functions, or docs made obsolete by your change.
  Do not remove pre-existing dead code unless asked.
- If you notice unrelated issues, mention them separately instead of fixing them
  opportunistically.
- Prefer the non-mutating targets `make test`, `make lint-check`, and
  `make lint-typing`. Run `make lint-all` only for an explicit autofix run.
- Stop when the narrowest useful check passes, or report the concrete blocker.
- When asked to document or explain the codebase, describe what exists. Do not
  propose improvements unless asked.
- Define the narrowest useful verification check before running broad commands.

## Task completion guidelines

Use judgment based on scope, but default to these artifacts.

### Bug fixes

1. Reproduce the bug in an E2E or integration test that fails without the fix.
2. Implement the smallest correct fix.
3. Run the focused test.
4. Run lint/type checks when the touched area or risk justifies them.

### New features

1. Follow existing interfaces and component patterns.
2. Add an E2E or integration test that exercises the feature and leaves a
   verifiable, repeatable artifact.
3. Update docstrings and user-facing docs if behavior is public.
4. Run focused tests plus the narrowest broader check that proves integration.

### Refactors/internal changes

- Preserve public behavior unless the change itself intends a break under the
  Pre-1.0 policy; update tests to the current behavior and drop tests for
  removed APIs.
- Run existing tests that cover the refactored behavior.

### Documentation-only changes

- Follow `writing-plain` for prose and instruction files. Also use `writing-diataxis-docs` for user-facing documentation pages.
- Be concrete and concise. Include examples, command invocations, data formats,
  or short flows when they make behavior easier to understand.
- Explain the why behind non-obvious design decisions and tradeoffs.
- Use diagrams or short step-by-step flows for complex interactions.
- Verify links, commands, and referenced paths where practical.
- Do not change code unless required to make the docs true.

## Git and pull requests

- Do not commit unless explicitly asked.
- Do not revert, overwrite, or clean up user changes unless explicitly asked.
- Commit messages and PR titles should use a conventional commit prefix such as
  `fix:`, `feat:`, `docs:`, `refactor:`, `test:`, `build:`, `ci:`, `chore:`,
  `style:`, or `perf:`.
- Commit messages should be a single short imperative sentence with no trailing
  period, clear and informative.
- PR descriptions should follow the repository's pull request template.
- Do not add AI-agent attribution such as `Co-Authored-By: Claude`, `Generated
  with Claude Code`, or similar text to commits, PRs, comments, or issues.
- When writing public-facing PR descriptions, issues, comments, or commit
  messages, avoid emojis, hype, AI-sounding phrasing, and unnecessary bullet
  lists. Prefer concise human prose.

## Self-review checklist

Before handing off non-trivial changes, check:

- Correctness: the code does what the request says and handles relevant edge
  cases such as empty inputs, `None`, and boundary values.
- Leakage: no feature or model input uses information from after the scored transaction's timestamp or from its own label.
- Public contract: breaking changes follow the Pre-1.0 policy (hard cut, old
  surface removed, no compat shims); no leftover aliases or old-API tests
  remain.
- Tests: changed behavior has an E2E or integration test that fails when the
  behavior is broken and survives a behavior-preserving refactor.
- Style: new or modified public functions are typed, docstrings are updated when
  needed, and Ruff formatting/linting expectations are respected.
- Design: no unnecessary abstractions, private-member access, god objects,
  monster files, or speculative configurability were introduced.
- Safety: no secrets, unsafe deserialization, injection risks, resource leaks,
  or unnecessary hot-path allocations were introduced.
- Docs: user-facing behavior changes are reflected in relevant docs or
  docstrings.

## Do not do these things

- Do not introduce a `src/` layout.
- Do not use `pip`, `poetry`, or `conda` for project dependency operations.
- Do not use mypy; use `ty`.
- Do not add CUDA builds of torch.
- Do not add manual integration/socket markers that `tests/conftest.py` already
  auto-applies.
- Do not enable sockets in unit tests to avoid mocking.
- Do not commit secrets or `.env` files.
- Do not add new dependencies casually or pin upper bounds without a known
  compatibility reason.
- Do not pipe remote install scripts into a shell in CI or documentation.
- Do not refactor, reformat, or clean adjacent code just because it is nearby.
- Do not change public APIs without explicit approval unless the Pre-1.0 policy
  above allows the break; never keep the old API beside the new one, and update
  docs and tests to the current behavior.
- Do not change CI, Ruff, pytest, or typing config unless explicitly asked.
- Do not add AI attribution to commits, branches, PR descriptions, comments, or
  issues.
