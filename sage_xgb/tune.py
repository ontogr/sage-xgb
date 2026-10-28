"""Hyperparameter search on validation PR-AUC with a resumable Optuna study.

Each trial fits the GNN on every cross-fit block but the last, scores the last
block and the validation days, and trains XGBoost on the last block. The trial
reads only train and validation labels, so test and Tail labels never change
the study or the written config.
"""

from __future__ import annotations

import logging
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
        if entry.choices is None:
            msg = f"tune.space.{key} is categorical and needs choices"
            raise ConfigError(msg)
        return trial.suggest_categorical(key, entry.choices)
    if entry.low is None or entry.high is None:
        msg = f"tune.space.{key} needs low and high"
        raise ConfigError(msg)
    if entry.kind == "int":
        return trial.suggest_int(key, int(entry.low), int(entry.high))
    return trial.suggest_float(
        key, entry.low, entry.high, log=entry.kind == "log_float"
    )


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


def _objective_factory(config: Config) -> Callable[[optuna.Trial], float]:
    """Return the objective that scores one proxy trial by validation PR-AUC.

    The proxy GNN fit scores the last cross-fit block and the validation days.
    XGBoost trains on the last block and early-stops on validation.
    """
    blocks = split.crossfit_blocks(config.split)
    fit_id = len(blocks) - 1

    def objective(trial: optuna.Trial) -> float:
        params = {
            key: _suggest(trial, key, entry) for key, entry in config.tune.space.items()
        }
        trial_config = _trial_config(config, params)
        scores = crossfit.score_fits(trial_config, fits=[fit_id], also_score=("val",))
        result = train_xgb.train(trial_config, gnn_scores=scores, train_days=blocks[-1])
        return result.val_metrics["pr_auc"]

    return objective


def _count_completed(study: optuna.Study) -> int:
    """Return the number of trials that finished without pruning or failing."""
    return sum(
        trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
    )


def _open_study(config: Config, storage: str) -> optuna.Study:
    """Create the study or resume it, with a sampler seed that moves on resume.

    A new sampler restarts its random stream. Reusing `config.seed` on resume
    would replay the draws of the first trials and repeat their parameters, so
    the seed moves forward by the number of trials already recorded.
    """
    summaries = optuna.study.get_all_study_summaries(storage=storage)
    recorded = sum(
        summary.n_trials
        for summary in summaries
        if summary.study_name == config.variant
    )
    return optuna.create_study(
        study_name=config.variant,
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.seed + recorded),
        load_if_exists=True,
    )


def _write_best(config: Config, best_params: dict[str, Any]) -> Path:
    """Write the complete config with the best trial's values to `best.yaml`.

    The file has no `extends` key, so it loads on its own and
    `sage-xgb run --config` refits the pipeline from it.
    """
    raw = dump_config(config)
    for dotted_key, value in best_params.items():
        _set_dotted(raw, dotted_key, value)
    path = artifacts.stage_path(config, "tune", "best.yaml")
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def run(config: Config) -> dict[str, float]:
    """Search `tune.space` on validation PR-AUC and write `best.yaml`.

    The shared inputs come from `pipeline.run(config, through="features")`.
    Trials go into the SQLite study `tune/study.db` in the stage directory.
    A call runs `tune.n_trials` minus the trials already in the study, pruned
    ones included, so a study that already met the target gets no new trials.
    The call also stops after `tune.timeout_minutes`. A later call with
    the same artifacts directory resumes the study.

    Args:
        config: The run config. `xgb.use_gnn` must be true, because each
            trial scores the GNN proxy.

    Returns:
        `best_val_pr_auc` for the best completed trial, `n_trials` for every
        trial recorded in the study, and `n_completed` for the completed ones.

    Raises:
        ConfigError: `xgb.use_gnn` is false, or a `tune.space` entry is
            incomplete or names a key that is not a config field.
        RuntimeError: No trial completed within the limits.
    """
    if not config.xgb.use_gnn:
        msg = "tune scores the GNN proxy, so xgb.use_gnn must be true"
        raise ConfigError(msg)
    pipeline.run(config, through="features")
    artifacts.ensure_stage_dir(config, "tune")
    study_db = artifacts.stage_path(config, "tune", "study.db")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = _open_study(config, f"sqlite:///{study_db}")
    completed = _count_completed(study)
    remaining = config.tune.n_trials - len(study.trials)
    if remaining > 0:
        study.optimize(
            _objective_factory(config),
            n_trials=remaining,
            timeout=config.tune.timeout_minutes * 60,
        )
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
