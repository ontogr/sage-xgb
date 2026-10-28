"""MLflow boundary for pipeline runs.

This is the only module that imports MLflow. MLflow loads on first use, so
importing `sage_xgb` does not load it.
"""

from __future__ import annotations

import importlib
import os
import re
from collections.abc import Generator, Mapping
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from types import ModuleType

from sage_xgb.config import Config, dump_config
from sage_xgb.settings import MlflowSettings


# Anonymous MLflow telemetry stays off unless the environment sets it.
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

_PIPELINE_RUN_NAME = "run"
_PARAM_SECTIONS = ("data", "split", "features", "seed")
_PARAM_MAX_CHARS = 250
_SQLITE_PREFIX = "sqlite:///"
_INVALID_METRIC_CHARS = re.compile(r"[^A-Za-z0-9_\-. /]")
# MLflow writes the URI it is given back into MLFLOW_TRACKING_URI, so the setting
# is read once here, before this module makes any MLflow call.
_CONFIGURED_TRACKING_URI = MlflowSettings().tracking_uri


def _mlflow() -> ModuleType:
    """Return the MLflow module, importing it on first use."""
    return importlib.import_module("mlflow")


def _tracking_uri(config: Config) -> str:
    """Return the configured store URI, or the store in the artifacts directory."""
    if _CONFIGURED_TRACKING_URI:
        return _CONFIGURED_TRACKING_URI
    return f"{_SQLITE_PREFIX}{config.paths.artifacts_dir.resolve()}/mlflow.db"


def _store_missing(uri: str) -> bool:
    """Return true when `uri` names a SQLite file that does not exist yet."""
    path = uri.removeprefix(_SQLITE_PREFIX)
    return uri.startswith(_SQLITE_PREFIX) and not Path(path).exists()


def _activate(config: Config) -> str:
    """Select the MLflow store and return the id of the config's experiment.

    The experiment is created on first use, with its artifacts under
    `<artifacts_dir>/mlartifacts`. The artifacts directory is created first
    because the SQLite store lives in it.
    """
    mlflow = _mlflow()
    artifacts_dir = config.paths.artifacts_dir.resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(_tracking_uri(config))
    experiment = mlflow.get_experiment_by_name(config.tracking.experiment)
    if experiment is not None:
        return experiment.experiment_id
    return mlflow.create_experiment(
        config.tracking.experiment,
        artifact_location=(artifacts_dir / "mlartifacts").as_uri(),
    )


@contextmanager
def _run(config: Config, run_name: str) -> Generator[None]:
    """Open an MLflow run that nests under the active run when there is one."""
    if not config.tracking.enabled:
        yield
        return
    mlflow = _mlflow()
    experiment_id = _activate(config)
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=run_name,
        nested=mlflow.active_run() is not None,
    ):
        yield


def _flatten_params(config: Config) -> dict[str, str]:
    """Return the logged config settings as dotted keys and short strings."""
    settings = dump_config(config)
    params: dict[str, str] = {}
    for section in _PARAM_SECTIONS:
        _flatten_into(params, section, settings[section])
    return params


def _flatten_into(params: dict[str, str], key: str, value: object) -> None:
    """Store `value` under `key`, recursing into mappings with dotted keys."""
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            _flatten_into(params, f"{key}.{child_key}", child_value)
        return
    params[key] = str(value)[:_PARAM_MAX_CHARS]


def _metric_name(name: str) -> str:
    """Replace characters that MLflow rejects in metric names with underscores."""
    return _INVALID_METRIC_CHARS.sub("_", name)


def pipeline_run(config: Config) -> AbstractContextManager[None]:
    """Hold open the parent MLflow run named `run` for one pipeline run.

    Does nothing when `config.tracking.enabled` is false. An exception in the
    body marks the run as FAILED and propagates.
    """
    return _run(config, _PIPELINE_RUN_NAME)


def stage_run(config: Config, stage: str) -> AbstractContextManager[None]:
    """Hold open an MLflow run named after `stage`.

    The run nests under the active run, such as the one from `pipeline_run`,
    and is a top-level run otherwise. An exception in the body marks the run as
    FAILED and propagates. Does nothing when tracking is disabled.
    """
    return _run(config, stage)


def log_stage(
    config: Config,
    stage: str,
    *,
    duration_seconds: float,
    metrics: Mapping[str, float],
    manifest: Path | None,
) -> None:
    """Log the settings, metrics, duration and manifest of a finished stage.

    Call it inside `stage_run` for the same stage. The values go to the active
    run. The settings are the `data`, `split`, `features` and `seed` sections,
    flattened to dotted keys with values cut to 250 characters. Metric names
    that MLflow rejects get underscores in place of the invalid characters.

    Args:
        config: The run config.
        stage: Stage name, used as the artifact folder of the manifest.
        duration_seconds: Wall-clock seconds the stage took.
        metrics: Metrics that the stage returned.
        manifest: Manifest file to attach, or None to attach no file.
    """
    if not config.tracking.enabled:
        return
    mlflow = _mlflow()
    mlflow.log_params(_flatten_params(config))
    mlflow.log_metrics(
        {"duration_seconds": duration_seconds}
        | {_metric_name(name): value for name, value in metrics.items()}
    )
    if manifest is not None:
        mlflow.log_artifact(str(manifest), artifact_path=stage)


def list_run_names(config: Config) -> list[str]:
    """Return the name of each run in the config's experiment, one per run.

    Returns an empty list when tracking is disabled, when the SQLite store does
    not exist yet, or when the experiment does not exist. Listing never creates
    a store.
    """
    if not config.tracking.enabled:
        return []
    uri = _tracking_uri(config)
    if _store_missing(uri):
        return []
    mlflow = _mlflow()
    mlflow.set_tracking_uri(uri)
    experiment = mlflow.get_experiment_by_name(config.tracking.experiment)
    if experiment is None:
        return []
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id], output_format="list"
    )
    return [run.info.run_name for run in runs]
