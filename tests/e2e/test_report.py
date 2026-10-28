"""End-to-end tests for the report stage on the committed sample.

The report stage is the only code that reads test and Tail labels. These tests
run it on the session `pipeline_config`. Every report call appends to the
shared history file, so the history test compares the line count of its own
two calls with the count before them.
"""

import json
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest
from sklearn.metrics import average_precision_score

from sage_xgb import artifacts, split, train_xgb
from sage_xgb.config import Config
from sage_xgb.data import prepare
from sage_xgb.evaluation import report
from sage_xgb.features import table
from sage_xgb.gnn import crossfit


def _report_path(config: Config, name: str) -> Path:
    return artifacts.stage_path(config, "report", name)


def _history_lines(config: Config) -> list[str]:
    return (
        _report_path(config, "history.jsonl").read_text(encoding="utf-8").splitlines()
    )


class TestReport:
    """Slices, intervals and history written by the report stage."""

    def test_writes_test_and_tail_slices_with_metrics(
        self, pipeline_config: Config
    ) -> None:
        """Write both slices to report.json and return the test metrics."""
        flat = report.run(pipeline_config)

        written = json.loads(_report_path(pipeline_config, "report.json").read_text())
        for name in ("test", "tail"):
            assert written[name]["positives"] > 0
            assert written[name]["pr_auc"] is not None
        assert flat["test_pr_auc"] == pytest.approx(written["test"]["pr_auc"])

    def test_test_pr_auc_matches_an_independent_calculation(
        self, pipeline_config: Config
    ) -> None:
        """Match the test PR-AUC to scikit-learn on labels read straight from disk."""
        report.run(pipeline_config)

        bundle = train_xgb.load_bundle(
            artifacts.stage_dir(pipeline_config, "train-xgb")
        )
        rows = (
            prepare.scan_transactions(pipeline_config)
            .filter(pl.col("split") == "test")
            .select("row_id", "is_laundering")
            .collect()
        )
        frame = train_xgb.join_inputs(
            rows,
            table.read_features(pipeline_config),
            crossfit.read_scores(pipeline_config)
            if bundle.metadata["use_gnn"]
            else None,
        )
        expected = average_precision_score(
            frame["is_laundering"].to_numpy(), bundle.predict(frame)
        )
        written = json.loads(_report_path(pipeline_config, "report.json").read_text())
        assert written["test"]["pr_auc"] == pytest.approx(expected)
        assert 0.0 < expected < 1.0

    def test_intervals_contain_the_point_pr_auc_on_the_test_slice(
        self, pipeline_config: Config
    ) -> None:
        """Place the point PR-AUC of the test slice inside its bootstrap interval."""
        report.run(pipeline_config)

        written = json.loads(_report_path(pipeline_config, "report.json").read_text())
        low, high = written["test"]["intervals"]["pr_auc"]
        assert low <= written["test"]["pr_auc"] <= high

    def test_each_call_appends_one_history_line_per_slice(
        self, pipeline_config: Config
    ) -> None:
        """Append a record for each slice read, naming the bundle that was read."""
        report.run(pipeline_config)
        before = len(_history_lines(pipeline_config))

        report.run(pipeline_config)

        lines = _history_lines(pipeline_config)
        assert len(lines) == before + 2
        added = [json.loads(line) for line in lines[-2:]]
        booster = artifacts.stage_path(
            pipeline_config, "train-xgb", train_xgb.BOOSTER_FILE
        )
        assert [record["slice"] for record in added] == ["test", "tail"]
        for record in added:
            assert record["bundle_digest"] == artifacts.file_digest(booster)
            assert datetime.fromisoformat(record["read_at"]).tzinfo is not None

    def test_stale_config_is_refused_before_any_label_is_read(
        self, pipeline_config: Config
    ) -> None:
        """Raise without reading labels when the config no longer matches the bundle."""
        report.run(pipeline_config)
        before = _history_lines(pipeline_config)
        stale = pipeline_config.model_copy(update={"seed": pipeline_config.seed + 1})

        with pytest.raises(artifacts.StaleArtifactError):
            report.run(stale)

        assert _history_lines(pipeline_config) == before

    @pytest.mark.parametrize("name", ["test", "tail"])
    def test_holdout_labels_cannot_be_read_outside_the_report(
        self, pipeline_config: Config, name: str
    ) -> None:
        """Refuse to load test or Tail labels through the split module."""
        frame = prepare.scan_transactions(pipeline_config)

        with pytest.raises(split.HoldoutAccessError):
            split.labelled_rows(frame, (name,))
