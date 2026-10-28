"""Run the pipeline stages in order, skipping stages that are up to date."""

from __future__ import annotations

import importlib
import logging
import time
from collections.abc import Callable, Sequence

from sage_xgb import artifacts, tracking
from sage_xgb.config import Config


logger = logging.getLogger(__name__)

STAGE_FUNCTIONS: dict[str, str] = {
    "download": "sage_xgb.data.download:run",
    "prepare": "sage_xgb.data.prepare:run",
    "features": "sage_xgb.features.table:run",
    "gnn": "sage_xgb.gnn.crossfit:run",
    "train-xgb": "sage_xgb.train_xgb:run",
}

_StageFunction = Callable[[Config], dict[str, float]]


def _stage_index(stage: str) -> int:
    """Return the position of a pipeline stage and raise for an unknown name."""
    if stage not in artifacts.PIPELINE_STAGES:
        msg = (
            f"unknown pipeline stage {stage!r}, expected one of "
            f"{', '.join(artifacts.PIPELINE_STAGES)}"
        )
        raise ValueError(msg)
    return artifacts.PIPELINE_STAGES.index(stage)


def _resolve_stage(stage: str) -> _StageFunction:
    """Import the module of a stage on first use and return its `run` function."""
    module_name, _, function_name = STAGE_FUNCTIONS[stage].partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def _timing_table(timings: Sequence[tuple[str, str, float]]) -> str:
    """Format `(stage, status, seconds)` rows as a table with a total row."""
    lines = [f"{'stage':<12}{'status':<9}{'seconds':>9}"]
    lines += [
        f"{stage:<12}{status:<9}{seconds:>9.1f}" for stage, status, seconds in timings
    ]
    total = sum(seconds for _, _, seconds in timings)
    lines.append(f"{'total':<21}{total:>9.1f}")
    return "\n".join(lines)


def execute_stage(config: Config, stage: str) -> dict[str, float]:
    """Run one stage inside an MLflow stage run and return its metrics.

    The stage function runs first. Then its duration, metrics and manifest are
    logged to the stage run. A stage that raises leaves a FAILED run and logs
    nothing more.

    Args:
        config: The run config.
        stage: A key of `STAGE_FUNCTIONS`.

    Returns:
        The metrics that the stage function returned.

    Raises:
        ValueError: `stage` is not a key of `STAGE_FUNCTIONS`.
    """
    if stage not in STAGE_FUNCTIONS:
        msg = f"unknown stage {stage!r}, expected one of {', '.join(STAGE_FUNCTIONS)}"
        raise ValueError(msg)
    stage_function = _resolve_stage(stage)
    manifest = artifacts.manifest_path(config, stage)
    manifest.unlink(missing_ok=True)
    with tracking.stage_run(config, stage):
        started = time.perf_counter()
        metrics = stage_function(config)
        duration_seconds = time.perf_counter() - started
        tracking.log_stage(
            config,
            stage,
            duration_seconds=duration_seconds,
            metrics=metrics,
            manifest=manifest if manifest.exists() else None,
        )
    logger.info("stage %s took %.1f s", stage, duration_seconds)
    return metrics


def run(
    config: Config,
    *,
    force: bool = False,
    from_stage: str | None = None,
    through: str | None = None,
) -> None:
    """Run the pipeline stages in order and skip the ones that are up to date.

    The stages run from `prepare` up to and including `through`, which defaults
    to the last stage. A stage runs when `force` is true, when it is at or after
    `from_stage`, or when `artifacts.is_fresh` is false. Other stages are skipped
    and logged. `report` is never part of a run. When the run ends, a table of
    the stage timings is logged.

    Args:
        config: The run config.
        force: Rerun every stage.
        from_stage: Rerun this stage and every stage after it.
        through: Last stage to run.

    Raises:
        ValueError: `from_stage` or `through` is not a pipeline stage.
    """
    stages = artifacts.PIPELINE_STAGES
    last_index = len(stages) - 1 if through is None else _stage_index(through)
    from_index = len(stages) if from_stage is None else _stage_index(from_stage)
    timings: list[tuple[str, str, float]] = []
    with tracking.pipeline_run(config):
        for index, stage in enumerate(stages[: last_index + 1]):
            if force or index >= from_index or not artifacts.is_fresh(config, stage):
                started = time.perf_counter()
                execute_stage(config, stage)
                timings.append((stage, "ran", time.perf_counter() - started))
            else:
                logger.info("skipping %s, up to date", stage)
                timings.append((stage, "skipped", 0.0))
    logger.info("stage timings\n%s", _timing_table(timings))
