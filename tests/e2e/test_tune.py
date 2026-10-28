"""End-to-end tests for the tune stage on the committed sample.

The sample config runs three trials. One module-scoped fixture runs tune three
times in one artifacts directory: a first run, a rerun at the same target and a
resume with `tune.n_trials` raised to four. A second fixture runs tune on a
copy of the sample whose test and Tail labels are flipped. Each fixture keeps
plain records of the study, so the tests do not depend on the order they run in.
"""

import csv
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import optuna
import pytest

from sage_xgb import artifacts, tune
from sage_xgb.config import Config, ConfigError, dump_config, load_config


SAMPLE_DIR = Path(__file__).parents[1] / "fixtures" / "amlworld_hi_small_sample"
TRANS_FILE = "HI-Small_Trans.csv"
TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M"
# The per-call limit is set high so that a slow machine cannot cut a run short.
# Trial counts are what these tests check, not the time limit.
GENEROUS_TIMEOUT = "tune.timeout_minutes=60"


@dataclass(frozen=True)
class _TrialRecord:
    """A trial as stored in the study, kept as plain data for comparisons."""

    number: int
    state: optuna.trial.TrialState
    params: dict[str, Any]
    value: float | None


@dataclass(frozen=True)
class _TuneScenario:
    """Study records after each tune call in one shared artifacts directory."""

    config: Config
    first_result: dict[str, float]
    first: list[_TrialRecord]
    after_rerun: list[_TrialRecord]
    after_resume: list[_TrialRecord]


@dataclass(frozen=True)
class _FlippedTune:
    """Trials from tune on a sample copy whose holdout labels are flipped."""

    flipped_rows: int
    trials: list[_TrialRecord]


def _storage_url(config: Config) -> str:
    return f"sqlite:///{artifacts.stage_path(config, 'tune', 'study.db')}"


def _load_study(config: Config) -> optuna.Study:
    return optuna.load_study(study_name=config.variant, storage=_storage_url(config))


def _trial_records(config: Config) -> list[_TrialRecord]:
    return [
        _TrialRecord(trial.number, trial.state, trial.params, trial.value)
        for trial in _load_study(config).trials
    ]


def _complete(records: list[_TrialRecord]) -> list[_TrialRecord]:
    return [
        record for record in records if record.state == optuna.trial.TrialState.COMPLETE
    ]


def _copy_with_flipped_holdout_labels(destination: Path, holdout_start: date) -> int:
    """Copy the sample and flip `Is Laundering` for rows dated on or after a day.

    Args:
        destination: Directory to create. It must not exist yet.
        holdout_start: First date whose labels are flipped.

    Returns:
        The number of flipped rows.
    """
    shutil.copytree(SAMPLE_DIR, destination)
    trans_path = destination / TRANS_FILE
    with trans_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    flipped = 0
    for row in rows[1:]:
        if datetime.strptime(row[0], TIMESTAMP_FORMAT).date() >= holdout_start:
            row[-1] = "0" if row[-1] == "1" else "1"
            flipped += 1
    with trans_path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle, lineterminator="\n").writerows(rows)
    return flipped


@pytest.fixture(scope="module")
def tune_scenario(
    tmp_path_factory: pytest.TempPathFactory,
    config_factory: Callable[..., Config],
) -> _TuneScenario:
    """Run tune three times in one artifacts directory and record the study."""
    config = config_factory(tmp_path_factory.mktemp("tune_artifacts"), GENEROUS_TIMEOUT)
    first_result = tune.run(config)
    first = _trial_records(config)
    tune.run(config)
    after_rerun = _trial_records(config)
    resumed = config_factory(
        config.paths.artifacts_dir, "tune.n_trials=4", GENEROUS_TIMEOUT
    )
    tune.run(resumed)
    after_resume = _trial_records(resumed)
    return _TuneScenario(config, first_result, first, after_rerun, after_resume)


@pytest.fixture(scope="module")
def flipped_tune(
    tmp_path_factory: pytest.TempPathFactory,
    config_factory: Callable[..., Config],
) -> _FlippedTune:
    """Run tune on a copy of the sample with test and Tail labels flipped."""
    raw_dir = tmp_path_factory.mktemp("flipped_raw") / "sample"
    config = config_factory(
        tmp_path_factory.mktemp("flipped_artifacts"), GENEROUS_TIMEOUT, raw_dir=raw_dir
    )
    flipped_rows = _copy_with_flipped_holdout_labels(raw_dir, config.split.val_end)
    tune.run(config)
    return _FlippedTune(flipped_rows, _trial_records(config))


class TestTune:
    """Tune on the sample: trial counts, the best config file and isolation."""

    def test_completes_configured_trials(self, tune_scenario: _TuneScenario) -> None:
        """The study holds the configured number of trials."""
        assert len(_complete(tune_scenario.first)) == 3

    def test_returns_best_value_and_trial_counts(
        self, tune_scenario: _TuneScenario
    ) -> None:
        """The result reports the best validation PR-AUC and the trial counts."""
        result = tune_scenario.first_result
        best_first = max(record.value for record in _complete(tune_scenario.first))
        assert result["best_val_pr_auc"] == best_first
        assert result["n_trials"] == 3
        assert result["n_completed"] == 3

    def test_writes_best_yaml_with_best_trial_values(
        self, tune_scenario: _TuneScenario
    ) -> None:
        """The written config loads and carries the best trial's values."""
        config = tune_scenario.config
        best = load_config(artifacts.stage_path(config, "tune", "best.yaml"))
        best_raw = dump_config(best)
        for dotted_key, value in _load_study(config).best_params.items():
            section, field = dotted_key.split(".")
            assert best_raw[section][field] == value

    def test_rerun_at_target_adds_no_trials(self, tune_scenario: _TuneScenario) -> None:
        """A rerun with the target already met leaves the study unchanged."""
        assert tune_scenario.after_rerun == tune_scenario.first

    def test_raised_target_resumes_study_and_keeps_existing_trials(
        self, tune_scenario: _TuneScenario
    ) -> None:
        """Raising the target adds trials and keeps the recorded ones."""
        assert len(_complete(tune_scenario.after_resume)) == 4
        kept = tune_scenario.after_resume[: len(tune_scenario.first)]
        assert kept == tune_scenario.first

    def test_resumed_trials_do_not_repeat_earlier_parameters(
        self, tune_scenario: _TuneScenario
    ) -> None:
        """Trials added on resume sample new parameters, not replayed ones."""
        earlier = [record.params for record in tune_scenario.first]
        added = tune_scenario.after_resume[len(tune_scenario.first) :]
        assert added
        assert all(record.params not in earlier for record in added)

    def test_flipped_test_and_tail_labels_leave_trials_unchanged(
        self, tune_scenario: _TuneScenario, flipped_tune: _FlippedTune
    ) -> None:
        """Test and Tail labels do not change any completed trial."""
        assert flipped_tune.flipped_rows > 0
        assert flipped_tune.trials == tune_scenario.first


class TestTuneSettings:
    """Rejected settings, a changed study, interrupted trials and the time limit."""

    def test_study_from_other_settings_is_refused(
        self, tune_scenario: _TuneScenario
    ) -> None:
        """Refuse to resume a study after a setting that changes scores changes."""
        before = _trial_records(tune_scenario.config)
        changed = tune_scenario.config.model_copy(
            update={"seed": tune_scenario.config.seed + 1}
        )

        with pytest.raises(ConfigError, match="different"):
            tune.run(changed)

        assert _trial_records(tune_scenario.config) == before

    def test_searching_use_gnn_is_refused(
        self, tmp_path: Path, config_factory: Callable[..., Config]
    ) -> None:
        """Refuse a search space that could switch the GNN off."""
        config = config_factory(
            tmp_path,
            "tune.space={xgb.use_gnn: {kind: categorical, choices: [true, false]}}",
        )

        with pytest.raises(ConfigError, match="xgb.use_gnn"):
            tune.run(config)

    def test_failed_and_abandoned_trials_do_not_count_toward_the_target(
        self, tmp_path: Path, config_factory: Callable[..., Config]
    ) -> None:
        """Run a full target of new trials next to a failed and a running record."""
        config = config_factory(tmp_path, "tune.n_trials=1", GENEROUS_TIMEOUT)
        artifacts.ensure_stage_dir(config, "tune")
        study = optuna.create_study(
            study_name=config.variant,
            storage=_storage_url(config),
            direction="maximize",
        )
        study.add_trial(optuna.trial.create_trial(state=optuna.trial.TrialState.FAIL))
        study.ask()

        result = tune.run(config)

        assert result["n_completed"] == 1
        assert result["n_trials"] == 3

    def test_time_limit_covers_the_shared_stages(
        self, tmp_path: Path, config_factory: Callable[..., Config]
    ) -> None:
        """Start no trial when the limit passes during the shared stages."""
        config = config_factory(tmp_path, "tune.timeout_minutes=0.00001")

        with pytest.raises(RuntimeError, match="no tune trial completed"):
            tune.run(config)

    def test_best_yaml_is_replaced_without_leaving_a_temporary_file(
        self, tune_scenario: _TuneScenario
    ) -> None:
        """Leave only best.yaml after the write."""
        directory = artifacts.stage_dir(tune_scenario.config, "tune")

        assert not list(directory.glob("*.tmp"))
        assert (directory / "best.yaml").exists()
