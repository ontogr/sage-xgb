"""Report stage: score the test and Tail slices and write the report.

This module is the only code that reads test and Tail labels. It scores each
slice with the bundle from the `train-xgb` stage and applies the validation
threshold stored in that bundle. Each label read appends one line to
`history.jsonl` before the slice is scored, so the number of label reads stays on
record even when a later step fails. A lock file serializes runs of the same
variant.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import shutil
import tempfile
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

import numpy as np
import polars as pl

from sage_xgb import artifacts, split, train_xgb
from sage_xgb.config import Config
from sage_xgb.data import prepare
from sage_xgb.evaluation import metrics
from sage_xgb.features import table
from sage_xgb.gnn import crossfit


logger = logging.getLogger(__name__)

_REPORT_FILE = "report.json"
_HISTORY_FILE = "history.jsonl"
_LOCK_FILE = ".lock"
_SLICES = split.HOLDOUT_SPLITS


class SliceReport(TypedDict):
    """Metrics of one labelled slice.

    Attributes:
        rows: Rows in the slice.
        positives: Illicit transactions in the slice.
        pr_auc: PR-AUC, or None when the slice has no positives.
        f1: F1 at the validation threshold, or None when the slice has no
            positives.
        precision_at_k: Precision among the top `positives` rows, or None when
            the slice has no positives.
        intervals: 95 percent Poisson bootstrap intervals keyed by metric name,
            or None when the slice has no positives.
    """

    rows: int
    positives: int
    pr_auc: float | None
    f1: float | None
    precision_at_k: float | None
    intervals: dict[str, tuple[float, float]] | None


def _slice_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float, seed: int
) -> SliceReport:
    rows = int(labels.size)
    positives = int(labels.sum())
    if positives == 0:
        return {
            "rows": rows,
            "positives": 0,
            "pr_auc": None,
            "f1": None,
            "precision_at_k": None,
            "intervals": None,
        }
    return {
        "rows": rows,
        "positives": positives,
        "pr_auc": metrics.pr_auc(labels, scores),
        "f1": metrics.f1_at(labels, scores, threshold),
        "precision_at_k": metrics.precision_at_k(labels, scores, positives),
        "intervals": metrics.poisson_bootstrap(
            labels, scores, threshold, positives, seed=seed
        ),
    }


def _holdout_rows(transactions: pl.LazyFrame, name: str) -> pl.DataFrame:
    """Collect the row ids and labels of one holdout slice."""
    if name not in split.HOLDOUT_SPLITS:
        msg = f"{name!r} is not a holdout slice"
        raise ValueError(msg)
    return (
        transactions.filter(pl.col("split") == name)
        .select("row_id", "is_laundering")
        .collect()
    )


def _score_slice(
    *,
    config: Config,
    transactions: pl.LazyFrame,
    name: str,
    bundle: train_xgb.Bundle,
    bundle_digest: str,
    history: Path,
    features: pl.DataFrame,
    gnn_scores: pl.DataFrame | None,
) -> SliceReport:
    rows = _holdout_rows(transactions, name)
    reads = _append_history(history, bundle_digest, name)
    logger.info("holdout labels have been read %d times", reads)
    frame = train_xgb.join_inputs(rows, features, gnn_scores)
    labels = frame["is_laundering"].to_numpy()
    scores = bundle.predict(frame)
    return _slice_metrics(labels, scores, bundle.threshold, config.seed)


def _append_history(path: Path, bundle_digest: str, slice_name: str) -> int:
    """Append one read record and return the number of records in the file."""
    record = {
        "bundle_digest": bundle_digest,
        "read_at": datetime.now(UTC).isoformat(),
        "slice": slice_name,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return len(path.read_text(encoding="utf-8").splitlines())


@contextmanager
def _locked(directory: Path) -> Generator[None]:
    """Hold an exclusive lock on the report directory until the block ends."""
    with (directory / _LOCK_FILE).open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextmanager
def _bundle_snapshot(config: Config) -> Generator[Path]:
    """Copy the `train-xgb` directory and yield the copy.

    Raises:
        StaleArtifactError: The stage manifest changed while the copy was made.
    """
    source = artifacts.stage_dir(config, "train-xgb")
    before = artifacts.manifest_digest(config, "train-xgb")
    with tempfile.TemporaryDirectory() as scratch:
        copy = Path(scratch) / "train-xgb"
        shutil.copytree(source, copy)
        if artifacts.manifest_digest(config, "train-xgb") != before:
            msg = "stage 'train-xgb' changed while the report read it"
            raise artifacts.StaleArtifactError(msg)
        yield copy


def run(config: Config) -> dict[str, float]:
    """Score the test and Tail slices once and write `report.json`.

    Every stage up to `train-xgb` must be up to date for `config`. The bundle is
    copied before it is loaded, so the scores and the recorded digest describe
    the same files. The threshold comes from the bundle, so this function never
    chooses a threshold. The report directory receives `report.json` and one new
    line in `history.jsonl` for each slice whose labels were read. The stage has
    no manifest. Runs for the same variant wait for each other.

    Args:
        config: The run config.

    Returns:
        Scalar metrics with keys such as `test_pr_auc` and `tail_positives`.
        A slice without positives contributes no metric keys.

    Raises:
        StaleArtifactError: A stage is missing or out of date, or `train-xgb`
            was rewritten during the run.
    """
    directory = artifacts.ensure_stage_dir(config, "report")
    with _locked(directory):
        artifacts.require_fresh(config, "train-xgb")
        with _bundle_snapshot(config) as snapshot:
            bundle = train_xgb.load_bundle(snapshot)
            bundle_digest = artifacts.file_digest(snapshot / train_xgb.BOOSTER_FILE)
        features = table.read_features(config)
        gnn_scores = (
            crossfit.read_scores(config) if bundle.metadata["use_gnn"] else None
        )
        transactions = prepare.scan_transactions(config)
        slices = {
            name: _score_slice(
                config=config,
                transactions=transactions,
                name=name,
                bundle=bundle,
                bundle_digest=bundle_digest,
                history=directory / _HISTORY_FILE,
                features=features,
                gnn_scores=gnn_scores,
            )
            for name in _SLICES
        }
        artifacts.require_fresh(config, "train-xgb")
        report = {
            "bundle_digest": bundle_digest,
            "threshold": bundle.threshold,
            **slices,
        }
        _write_report(directory / _REPORT_FILE, report)
    return {
        f"{name}_{key}": float(value)
        for name, slice_report in slices.items()
        for key, value in slice_report.items()
        if isinstance(value, int | float)
        if key != "intervals"
    }


def _write_report(path: Path, report: Mapping[str, object]) -> None:
    """Write `report` to `path` by replacing the file in one step."""
    scratch = path.with_suffix(".tmp")
    scratch.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", "utf-8")
    os.replace(scratch, path)
