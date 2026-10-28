"""Report stage: score the test and Tail slices and write the report.

This module is the only code that reads test and Tail labels. It scores each
slice with the bundle from the `train-xgb` stage and applies the validation
threshold stored in that bundle. Every run appends one line to `history.jsonl`,
so the number of label reads stays on record.
"""

from __future__ import annotations

import json
import logging
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
_SLICES = ("test", "tail")


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


def _score_slice(
    *,
    config: Config,
    transactions: pl.LazyFrame,
    name: str,
    bundle: train_xgb.Bundle,
    features: pl.DataFrame,
    gnn_scores: pl.DataFrame | None,
) -> SliceReport:
    labelled = split.labelled_rows(transactions, (name,), allow_holdout=True)
    rows = labelled.select("row_id", "is_laundering").collect()
    frame = train_xgb.join_inputs(rows, features, gnn_scores)
    labels = frame["is_laundering"].to_numpy()
    scores = bundle.predict(frame)
    return _slice_metrics(labels, scores, bundle.threshold, config.seed)


def _append_history(path: Path, bundle_digest: str) -> int:
    """Append one read record and return the number of records in the file."""
    record = {"bundle_digest": bundle_digest, "read_at": datetime.now(UTC).isoformat()}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return len(path.read_text(encoding="utf-8").splitlines())


def run(config: Config) -> dict[str, float]:
    """Score the test and Tail slices once and write `report.json`.

    The `train-xgb` stage must have run. The threshold comes from the bundle, so
    this function never chooses a threshold. The report directory receives
    `report.json` and one new line in `history.jsonl`. The stage has no
    manifest.

    Args:
        config: The run config.

    Returns:
        Scalar metrics with keys such as `test_pr_auc` and `tail_positives`.
        A slice without positives contributes no metric keys.
    """
    bundle = train_xgb.load_bundle(artifacts.stage_dir(config, "train-xgb"))
    features = table.read_features(config)
    gnn_scores = crossfit.read_scores(config) if bundle.metadata["use_gnn"] else None
    transactions = prepare.scan_transactions(config)
    slices = {
        name: _score_slice(
            config=config,
            transactions=transactions,
            name=name,
            bundle=bundle,
            features=features,
            gnn_scores=gnn_scores,
        )
        for name in _SLICES
    }
    bundle_digest = artifacts.file_digest(
        artifacts.stage_path(config, "train-xgb", train_xgb.BOOSTER_FILE)
    )
    directory = artifacts.ensure_stage_dir(config, "report")
    report = {"bundle_digest": bundle_digest, "threshold": bundle.threshold, **slices}
    (directory / _REPORT_FILE).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    reads = _append_history(directory / _HISTORY_FILE, bundle_digest)
    logger.info("test labels have been read %d times", reads)
    return {
        f"{name}_{key}": float(value)
        for name, slice_report in slices.items()
        for key, value in slice_report.items()
        if isinstance(value, int | float)
        if key != "intervals"
    }
