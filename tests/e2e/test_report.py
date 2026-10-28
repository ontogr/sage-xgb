"""End-to-end tests for the report stage on the committed sample.

The report stage is the only code that reads test and Tail labels. These tests
run it on the session `pipeline_config`. Every report call appends to the
shared history file, so the history test compares the line count of its own
two calls with the count before them.
"""

import json
from pathlib import Path

import pytest

from sage_xgb import artifacts, split, train_xgb
from sage_xgb.config import Config
from sage_xgb.data import prepare
from sage_xgb.evaluation import report


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

    def test_intervals_contain_the_point_pr_auc_on_the_test_slice(
        self, pipeline_config: Config
    ) -> None:
        """Place the point PR-AUC of the test slice inside its bootstrap interval."""
        report.run(pipeline_config)

        written = json.loads(_report_path(pipeline_config, "report.json").read_text())
        low, high = written["test"]["intervals"]["pr_auc"]
        assert low <= written["test"]["pr_auc"] <= high

    def test_each_call_appends_one_history_line_with_the_bundle_digest(
        self, pipeline_config: Config
    ) -> None:
        """Append one history record per call, naming the bundle that was read."""
        report.run(pipeline_config)
        before = len(_history_lines(pipeline_config))

        report.run(pipeline_config)

        lines = _history_lines(pipeline_config)
        assert len(lines) == before + 1
        latest = json.loads(lines[-1])
        booster = artifacts.stage_path(
            pipeline_config, "train-xgb", train_xgb.BOOSTER_FILE
        )
        assert latest["bundle_digest"] == artifacts.file_digest(booster)
        assert "read_at" in latest

    @pytest.mark.parametrize("name", ["test", "tail"])
    def test_holdout_labels_cannot_be_read_outside_the_report(
        self, pipeline_config: Config, name: str
    ) -> None:
        """Refuse to load test or Tail labels unless the caller is the report."""
        frame = prepare.scan_transactions(pipeline_config)

        with pytest.raises(split.HoldoutAccessError):
            split.labelled_rows(frame, (name,))
