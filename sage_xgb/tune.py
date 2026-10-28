"""Hyperparameter search on validation PR-AUC with a resumable Optuna study.

Each trial fits the GNN on every cross-fit block but the last, scores the last
block and the validation days, and trains XGBoost on the last block. The trial
reads only train and validation labels, so test and Tail labels never change
the study or the written config.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import optuna
import yaml
from pydantic import ValidationError

from sage_xgb import artifacts, pipeline, split, train_xgb
from sage_xgb.config import Config, ConfigError, SpaceEntry, dump_config
from sage_xgb.gnn import crossfit


logger = logging.getLogger(__name__)


def _set_dotted(raw: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a value at an existing dotted config path such as `gnn.hidden`."""
    *parents, leaf = dotted_key.split(".")
    node = raw
    for part in parents:
        node = node.get(part, {})
    if leaf not in node:
        msg = f"tune.space key {dotted_key!r} is not a config field"
        raise ConfigError(msg)
    node[leaf] = value


def _suggest(trial: optuna.Trial, key: str, entry: SpaceEntry) -> Any:
    """Sample one searched value with the distribution named by `entry.kind`."""
    if entry.kind == "categorical":
        return trial.suggest_categorical(key, entry.choices or [])
    low, high = entry.low, entry.high
    if low is None or high is None:
        msg = f"tune.space.{key} needs low and high"
        raise ConfigError(msg)
    if entry.kind == "int":
        return trial.suggest_int(key, int(low), int(high))
    return trial.suggest_float(key, low, high, log=entry.kind == "log_float")


def _trial_config(config: Config, params: dict[str, Any]) -> Config:
    """Return a validated copy of `config` with the sampled values applied.

    Raises:
        optuna.TrialPruned: The sampled values break a config rule, such as a
            GNN layer count that no longer matches `gnn.neighbors`.
    """
    raw = dump_config(config)
    for dotted_key, value in params.items():
        _set_dotted(raw, dotted_key, value)
    try:
        return Config.model_validate(raw)
    except ValidationError as error:
        logger.warning("pruned trial, sampled values break the config: %s", error)
        raise optuna.TrialPruned(str(error)) from error


class _DeadlineReachedError(Exception):
    """Error that ends the study when the time limit passes inside a trial."""


def _objective_factory(
    config: Config, deadline: float
) -> Callable[[optuna.Trial], float]:
    """Return the objective that scores one proxy trial by validation PR-AUC.

    The proxy GNN fit scores the last cross-fit block and the validation days.
    XGBoost trains on the last block and early-stops on validation. The objective
    checks `deadline` (a `time.monotonic` value) before each of those steps. A
    step that has started runs to its end.

    Raises:
        _DeadlineReachedError: The deadline passed before a step could start.
    """
    blocks = split.crossfit_blocks(config.split)
    fit_id = len(blocks) - 1

    def objective(trial: optuna.Trial) -> float:
        params = {
            key: _suggest(trial, key, entry) for key, entry in config.tune.space.items()
        }
        trial_config = _trial_config(config, params)
        _check_deadline(deadline)
        scores = crossfit.score_fits(trial_config, fits=[fit_id], also_score=("val",))
        _check_deadline(deadline)
        result = train_xgb.train(trial_config, gnn_scores=scores, train_days=blocks[-1])
        return result.val_metrics["pr_auc"]

    return objective


def _check_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise _DeadlineReachedError


def _count_completed(study: optuna.Study) -> int:
    """Return the number of trials that finished without pruning or failing."""
    return sum(
        trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
    )


def _count_finished(study: optuna.Study) -> int:
    """Return the trials that count toward `tune.n_trials`.

    Completed and pruned trials count. A pruned trial hit a config rule, so
    sampling it again would fail the same way. Failed trials and trials left
    running by a stopped process do not count, so a later call replaces them.
    """
    counted = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    return sum(trial.state in counted for trial in study.trials)


_FINGERPRINT_SECTIONS = ("data", "split", "features", "gnn", "xgb", "seed")


def _fingerprint(config: Config) -> str:
    """Hash the settings that change a trial's score.

    The hash covers the data, split, seed, feature, GNN and XGBoost settings and
    the search space. It leaves out the searched keys' own values, the trial
    count, the time limit and the paths, so `best.yaml` can be tuned again.
    """
    raw = dump_config(config)
    for dotted_key in config.tune.space:
        *parents, leaf = dotted_key.split(".")
        node = raw
        for name in parents:
            node = node[name]
        node.pop(leaf, None)
    payload = {name: raw[name] for name in _FINGERPRINT_SECTIONS}
    payload["space"] = config.model_dump(mode="json")["tune"]["space"]
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def _open_study(config: Config, storage: str) -> optuna.Study:
    """Create the study or resume it, with a sampler seed that moves on resume.

    A new sampler restarts its random stream. Reusing `config.seed` on resume
    would replay the draws of the first trials and repeat their parameters, so
    the seed moves forward by the number of trials already recorded.

    Raises:
        ConfigError: The study was created under settings that differ from
            `config`, so its trials do not score the same objective.
    """
    summaries = optuna.study.get_all_study_summaries(storage=storage)
    recorded = sum(
        summary.n_trials
        for summary in summaries
        if summary.study_name == config.variant
    )
    study = optuna.create_study(
        study_name=config.variant,
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.seed + recorded),
        load_if_exists=True,
    )
    fingerprint = _fingerprint(config)
    stored = study.user_attrs.get("fingerprint")
    if stored is None:
        study.set_user_attr("fingerprint", fingerprint)
    elif stored != fingerprint:
        msg = (
            f"the study in {storage} was created with different data, split, seed, "
            "model or search space settings; use a new artifacts directory or "
            "delete the study file"
        )
        raise ConfigError(msg)
    return study


def _write_best(config: Config, best_params: dict[str, Any]) -> Path:
    """Write the complete config with the best trial's values to `best.yaml`.

    The file has no `extends` key, so it loads on its own and
    `sage-xgb run --config` refits the pipeline from it.
    """
    raw = dump_config(config)
    for dotted_key, value in best_params.items():
        _set_dotted(raw, dotted_key, value)
    path = artifacts.stage_path(config, "tune", "best.yaml")
    scratch = path.with_suffix(".yaml.tmp")
    scratch.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    os.replace(scratch, path)
    return path


def run(config: Config) -> dict[str, float]:
    """Search `tune.space` on validation PR-AUC and write `best.yaml`.

    The shared inputs come from `pipeline.run(config, through="features")`.
    Trials go into the SQLite study `tune/study.db` in the stage directory.
    A call runs `tune.n_trials` minus the completed and pruned trials already in
    the study, so a study that already met the target gets no new trials. Failed
    trials and trials left running by a stopped process are replaced. Calls for the
    same variant and artifacts directory wait for each other.
    The time limit `tune.timeout_minutes` starts when the call starts and includes
    the shared stages. A fit or training step already running is not interrupted,
    so the call can end after the limit by the length of that step. A later call with
    the same artifacts directory and the same objective settings resumes the study.

    Args:
        config: The run config. `xgb.use_gnn` must be true, because each
            trial scores the GNN proxy.

    Returns:
        `best_val_pr_auc` for the best completed trial, `n_trials` for every
        trial recorded in the study, and `n_completed` for the completed ones.

    Raises:
        ConfigError: `xgb.use_gnn` is false or is a searched key, a `tune.space`
            entry names a key that is not a config field, or the study was
            created with different settings.
        RuntimeError: No trial completed within the limits.
    """
    if not config.xgb.use_gnn:
        msg = "tune scores the GNN proxy, so xgb.use_gnn must be true"
        raise ConfigError(msg)
    if "xgb.use_gnn" in config.tune.space:
        msg = "tune scores the GNN proxy, so xgb.use_gnn cannot be a searched key"
        raise ConfigError(msg)
    deadline = time.monotonic() + config.tune.timeout_minutes * 60
    pipeline.run(config, through="features")
    artifacts.ensure_stage_dir(config, "tune")
    with artifacts.exclusive_lock(artifacts.stage_path(config, "tune", ".lock")):
        return _search(config, deadline)


def _search(config: Config, deadline: float) -> dict[str, float]:
    """Run the missing trials of the study and write `best.yaml`."""
    study_db = artifacts.stage_path(config, "tune", "study.db")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = _open_study(config, f"sqlite:///{study_db}")
    completed = _count_completed(study)
    remaining = config.tune.n_trials - _count_finished(study)
    time_left = deadline - time.monotonic()
    if remaining > 0 and time_left > 0:
        try:
            study.optimize(
                _objective_factory(config, deadline),
                n_trials=remaining,
                timeout=time_left,
            )
        except _DeadlineReachedError:
            logger.warning("tune stopped at the time limit inside a trial")
        completed = _count_completed(study)
    if completed == 0:
        msg = f"no tune trial completed in {study_db}; check the log for pruned trials"
        raise RuntimeError(msg)
    best_path = _write_best(config, study.best_params)
    logger.info(
        "best trial %d with validation PR-AUC %.4f and params %s, written to %s",
        study.best_trial.number,
        study.best_value,
        study.best_params,
        best_path,
    )
    return {
        "best_val_pr_auc": study.best_value,
        "n_trials": len(study.trials),
        "n_completed": completed,
    }
