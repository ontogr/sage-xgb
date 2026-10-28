"""Adds test markers by directory and keeps MLflow quiet and offline."""

import os

import pytest


os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Add markers to tests by directory.

    Each test under tests/integration/ gets the integration, enable_socket and
    flaky markers. Test files must not add these markers directly. Tests under
    tests/e2e/ get the slow marker and are skipped on xdist workers, because
    they share session fixtures that run the pipeline once; run `make test-e2e`.

    Args:
        config: The pytest run configuration.
        items: The collected test items.
    """
    for item in items:
        path = str(item.path).replace("\\", "/")
        if "tests/integration" in path:
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.enable_socket)
            item.add_marker(
                pytest.mark.flaky(
                    reruns=3, reruns_delay=30, rerun_except=[AssertionError]
                )
            )
        if "tests/e2e" in path:
            item.add_marker(pytest.mark.slow)
            if os.environ.get("PYTEST_XDIST_WORKER"):
                item.add_marker(
                    pytest.mark.skip(
                        reason="e2e tests share session fixtures; run `make test-e2e`"
                    )
                )
