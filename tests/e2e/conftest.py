"""Session fixtures that run the pipeline stages on the committed sample once."""

from collections.abc import Callable
from pathlib import Path

import pytest

from sage_xgb import pipeline
from sage_xgb.config import Config, load_config


SAMPLE_DIR = Path(__file__).parents[1] / "fixtures" / "amlworld_hi_small_sample"
SAMPLE_CONFIG = Path(__file__).parents[2] / "configs" / "sample.yaml"


def make_config(
    artifacts_dir: Path,
    *overrides: str,
    raw_dir: Path = SAMPLE_DIR,
    tracking: bool = False,
) -> Config:
    """Return the sample config writing into `artifacts_dir`.

    MLflow tracking is off unless `tracking` is true, because every new
    artifacts directory creates a new SQLite store and migrating one takes
    seconds.

    Args:
        artifacts_dir: Directory for this config's artifacts.
        *overrides: Extra `dotted.key=value` overrides.
        raw_dir: Directory that holds the raw files. Defaults to the sample.
        tracking: Whether to enable MLflow tracking.
    """
    return load_config(
        SAMPLE_CONFIG,
        [
            f"paths.artifacts_dir={artifacts_dir}",
            f"data.raw_dir={raw_dir}",
            "gnn.num_threads=1",
            f"tracking.enabled={str(tracking).lower()}",
            *overrides,
        ],
    )


@pytest.fixture(scope="session")
def config_factory() -> Callable[..., Config]:
    """Return `make_config` for tests that need their own artifacts directory."""
    return make_config


@pytest.fixture(scope="session")
def sample_config(tmp_path_factory: pytest.TempPathFactory) -> Config:
    """Return the shared sample config, with MLflow tracking on.

    No stage has run yet. It is the only config whose tracking store the
    tests read.
    """
    return make_config(tmp_path_factory.mktemp("artifacts"), tracking=True)


@pytest.fixture(scope="session")
def prepared_config(sample_config: Config) -> Config:
    """Run `prepare` on the sample and return the config."""
    pipeline.run(sample_config, through="prepare")
    return sample_config


@pytest.fixture(scope="session")
def featured_config(sample_config: Config) -> Config:
    """Run `prepare` and `features` on the sample and return the config."""
    pipeline.run(sample_config, through="features")
    return sample_config


@pytest.fixture(scope="session")
def gnn_config(sample_config: Config) -> Config:
    """Run `prepare`, `features` and `gnn` on the sample and return the config."""
    pipeline.run(sample_config, through="gnn")
    return sample_config
