"""Train the XGBoost stage and write the model bundle.

XGBoost reads the feature table and, when `xgb.use_gnn` is true, the
out-of-fold GNN scores. It trains on the training rows of the training days and
early-stops on the validation rows. The F1-maximising threshold on validation is
stored in the bundle, and the report stage applies that threshold unchanged.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import polars as pl
import xgboost as xgb

from sage_xgb import artifacts, split
from sage_xgb.config import Config
from sage_xgb.data import prepare, vocab
from sage_xgb.evaluation import metrics
from sage_xgb.features import table
from sage_xgb.gnn import crossfit


logger = logging.getLogger(__name__)

BOOSTER_FILE = "booster.json"
_FEATURE_COLUMNS_FILE = "feature_columns.json"
_METADATA_FILE = "metadata.json"
_METRICS_FILE = "metrics_val.json"
_ROW_COLUMNS = ("row_id", "split", "day", "is_laundering")


class BundleMetadata(TypedDict):
    """Content of `metadata.json`.

    Attributes:
        config_hash: Hash of the config sections that the bundle depends on.
        training_days: 1-based days whose rows trained the model.
        n_train_rows: Training rows used to fit the model.
        n_val_rows: Validation rows used for early stopping and the threshold.
        n_train_positives: Illicit rows among the training rows.
        use_gnn: Whether the GNN score columns are features.
        vocabulary_hash: Hash of the fixed vocabularies and currency rates.
        threshold: Validation F1 threshold. Scores at or above it are positive.
        best_iteration: Zero-based boosting round that early stopping selected.
        trained_on: Short text of the training days, such as "days 3-6".
    """

    config_hash: str
    training_days: list[int]
    n_train_rows: int
    n_val_rows: int
    n_train_positives: int
    use_gnn: bool
    vocabulary_hash: str
    threshold: float
    best_iteration: int
    trained_on: str


@dataclass(frozen=True)
class Bundle:
    """A trained booster with the feature order and threshold it needs.

    Attributes:
        booster: The trained XGBoost booster.
        feature_columns: Feature names in the order the booster reads them.
        threshold: Validation F1 threshold. Scores at or above it are positive.
        metadata: Content of `metadata.json`.
    """

    booster: xgb.Booster
    feature_columns: list[str]
    threshold: float
    metadata: BundleMetadata

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        """Return the probability that each row is positive.

        Args:
            frame: Rows with every name in `feature_columns`. Other columns are
                ignored.

        Returns:
            One probability per row, computed with the boosting rounds up to
            the stored best iteration.

        Raises:
            KeyError: `frame` lacks one or more feature columns. The message
                names every missing column. Missing values are never filled.
        """
        missing = [name for name in self.feature_columns if name not in frame.columns]
        if missing:
            msg = f"frame lacks feature columns: {missing}"
            raise KeyError(msg)
        matrix = np.asarray(
            frame.select(self.feature_columns).to_numpy(), dtype=np.float32
        )
        dmatrix = xgb.DMatrix(matrix, feature_names=self.feature_columns)
        rounds = self.metadata["best_iteration"] + 1
        return self.booster.predict(dmatrix, iteration_range=(0, rounds))


@dataclass(frozen=True)
class TrainResult:
    """Output of `train`, held in memory until `run` writes it.

    Attributes:
        bundle: The trained bundle.
        val_metrics: Validation metrics with the keys `pr_auc`, `f1`,
            `precision_at_k`, `threshold`, `positives` and `rows`.
        n_train_rows: Training rows used to fit the model.
        n_val_rows: Validation rows.
        n_train_positives: Illicit rows among the training rows.
        training_days: 1-based days whose rows trained the model.
    """

    bundle: Bundle
    val_metrics: dict[str, float]
    n_train_rows: int
    n_val_rows: int
    n_train_positives: int
    training_days: list[int]


def _feature_columns(config: Config, features: pl.DataFrame) -> list[str]:
    columns = [name for name in features.columns if name != "row_id"]
    if config.xgb.use_gnn:
        columns += [
            crossfit.GNN_SCORE_COLUMN,
            *crossfit.gnn_embedding_columns(config.gnn.embedding_dim),
        ]
    return columns


def _read_training_frames(
    config: Config,
    features: pl.DataFrame,
    gnn_scores: pl.DataFrame | None,
    train_days: list[int],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    labelled = split.labelled_rows(prepare.scan_transactions(config), ("train", "val"))
    is_training_row = (pl.col("split") == "train") & pl.col("day").is_in(train_days)
    is_validation_row = pl.col("split") == "val"
    rows = (
        labelled.filter(is_training_row | is_validation_row)
        .select(_ROW_COLUMNS)
        .collect()
    )
    joined = join_inputs(rows, features, gnn_scores)
    train_frame = joined.filter(pl.col("split") == "train")
    val_frame = joined.filter(pl.col("split") == "val")
    return train_frame, val_frame


def _check_training_frames(
    train_frame: pl.DataFrame,
    val_frame: pl.DataFrame,
    train_days: list[int],
) -> tuple[int, int]:
    if val_frame.is_empty():
        msg = "validation rows are empty, so there is nothing to early-stop on"
        raise ValueError(msg)
    n_train_positives = int(train_frame["is_laundering"].sum())
    if n_train_positives == 0:
        msg = f"training rows on days {train_days} have no illicit transactions"
        raise ValueError(msg)
    n_val_positives = int(val_frame["is_laundering"].sum())
    if n_val_positives == 0:
        msg = "validation rows have no illicit transactions"
        raise ValueError(msg)
    return n_train_positives, n_val_positives


def _dmatrix(frame: pl.DataFrame, feature_columns: list[str]) -> xgb.DMatrix:
    matrix = np.asarray(frame.select(feature_columns).to_numpy(), dtype=np.float32)
    labels = frame["is_laundering"].cast(pl.Float32).to_numpy()
    return xgb.DMatrix(matrix, label=labels, feature_names=feature_columns)


def _booster_params(config: Config) -> dict[str, Any]:
    return {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "nthread": config.xgb.num_threads,
        "seed": config.seed,
        "max_depth": config.xgb.max_depth,
        "eta": config.xgb.learning_rate,
        "subsample": config.xgb.subsample,
        "colsample_bytree": config.xgb.colsample_bytree,
        "min_child_weight": config.xgb.min_child_weight,
        "lambda": config.xgb.reg_lambda,
        "scale_pos_weight": config.xgb.scale_pos_weight,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_bundle(directory: Path) -> Bundle:
    """Load the bundle that `run` wrote to a `train-xgb` stage directory.

    Args:
        directory: The `train-xgb` stage directory.

    Returns:
        The loaded bundle.
    """
    booster = xgb.Booster()
    booster.load_model(directory / BOOSTER_FILE)
    feature_columns: list[str] = _read_json(directory / _FEATURE_COLUMNS_FILE)
    metadata: BundleMetadata = _read_json(directory / _METADATA_FILE)
    return Bundle(
        booster=booster,
        feature_columns=feature_columns,
        threshold=metadata["threshold"],
        metadata=metadata,
    )


def _require_matches(rows: pl.DataFrame, other: pl.DataFrame, label: str) -> None:
    unmatched = rows.join(other.select("row_id"), on="row_id", how="anti")
    if unmatched.is_empty():
        return
    first_row_id = unmatched["row_id"][0]
    msg = f"{unmatched.height} rows have no {label} row, first row_id {first_row_id}"
    raise ValueError(msg)


def join_inputs(
    rows: pl.DataFrame,
    features: pl.DataFrame,
    gnn_scores: pl.DataFrame | None,
) -> pl.DataFrame:
    """Join rows with their feature values and GNN scores.

    The training and the report stage both build their inputs with this
    function, so they get the same columns in the same order.

    Args:
        rows: Frame with a `row_id` column and any other columns to keep.
        features: The feature table, keyed by `row_id`.
        gnn_scores: The GNN score table, keyed by `row_id`. Pass None to leave
            out the GNN columns. A `fit_id` column is dropped.

    Returns:
        The joined rows, sorted by `row_id`.

    Raises:
        ValueError: A row has no feature row, or has no GNN score when
            `gnn_scores` is given.
    """
    _require_matches(rows, features, "feature")
    joined = rows.join(features, on="row_id", how="inner")
    if gnn_scores is None:
        return joined.sort("row_id")
    _require_matches(rows, gnn_scores, "GNN score")
    return joined.join(gnn_scores.drop("fit_id"), on="row_id", how="inner").sort(
        "row_id"
    )


def train(
    config: Config,
    *,
    gnn_scores: pl.DataFrame | None,
    train_days: Sequence[int],
) -> TrainResult:
    """Train XGBoost on the training days and early-stop on validation.

    Only the `train` and `val` splits are read. Training uses the `train` rows
    whose `day` is in `train_days`. Early stopping and the threshold use every
    `val` row. Nothing is written to disk.

    Args:
        config: The run config. `xgb` and `gnn` set the model, and `seed` seeds
            XGBoost.
        gnn_scores: Out-of-fold GNN scores, as returned by
            `crossfit.read_scores` or `crossfit.score_fits`. Required when
            `xgb.use_gnn` is true and ignored otherwise.
        train_days: 1-based days whose training rows fit the model.

    Returns:
        The bundle, the validation metrics and the row counts.

    Raises:
        ValueError: `xgb.use_gnn` is true and `gnn_scores` is None, a row lacks
            a feature or GNN score, the validation rows are empty or have no
            illicit transactions, or the training rows have no illicit
            transactions.
    """
    if config.xgb.use_gnn and gnn_scores is None:
        msg = "xgb.use_gnn is true, so gnn_scores is required"
        raise ValueError(msg)
    gnn = gnn_scores if config.xgb.use_gnn else None
    days = list(train_days)
    features = table.read_features(config)
    feature_columns = _feature_columns(config, features)
    train_frame, val_frame = _read_training_frames(config, features, gnn, days)
    n_train_positives, n_val_positives = _check_training_frames(
        train_frame, val_frame, days
    )

    dtrain = _dmatrix(train_frame, feature_columns)
    dval = _dmatrix(val_frame, feature_columns)
    booster = xgb.train(
        _booster_params(config),
        dtrain,
        num_boost_round=config.xgb.n_estimators,
        evals=[(dval, "val")],
        early_stopping_rounds=config.xgb.early_stopping_rounds,
        verbose_eval=False,
    )
    best_iteration = int(booster.best_iteration)
    val_labels = val_frame["is_laundering"].to_numpy()
    val_scores = booster.predict(dval, iteration_range=(0, best_iteration + 1))
    threshold = metrics.best_f1_threshold(val_labels, val_scores)
    val_metrics: dict[str, float] = {
        "pr_auc": metrics.pr_auc(val_labels, val_scores),
        "f1": metrics.f1_at(val_labels, val_scores, threshold),
        "precision_at_k": metrics.precision_at_k(
            val_labels, val_scores, n_val_positives
        ),
        "threshold": threshold,
        "positives": n_val_positives,
        "rows": val_frame.height,
    }
    metadata: BundleMetadata = {
        "config_hash": artifacts.config_hash(
            config, ("xgb", "features", "split", "seed")
        ),
        "training_days": days,
        "n_train_rows": train_frame.height,
        "n_val_rows": val_frame.height,
        "n_train_positives": n_train_positives,
        "use_gnn": config.xgb.use_gnn,
        "vocabulary_hash": vocab.vocabulary_hash(),
        "threshold": threshold,
        "best_iteration": best_iteration,
        "trained_on": f"days {min(days)}-{max(days)}",
    }
    logger.info(
        "trained XGBoost on %d rows, best iteration %d",
        train_frame.height,
        best_iteration,
    )
    return TrainResult(
        bundle=Bundle(
            booster=booster,
            feature_columns=feature_columns,
            threshold=threshold,
            metadata=metadata,
        ),
        val_metrics=val_metrics,
        n_train_rows=train_frame.height,
        n_val_rows=val_frame.height,
        n_train_positives=n_train_positives,
        training_days=days,
    )


def run(config: Config) -> dict[str, float]:
    """Run the `train-xgb` stage and write the model bundle.

    The stage writes `booster.json`, `feature_columns.json`, `metadata.json` and
    `metrics_val.json` to its stage directory, then its manifest. The `prepare`
    and `features` stages must have run. The `gnn` stage must have run when
    `xgb.use_gnn` is true.

    Args:
        config: The run config.

    Returns:
        Validation PR-AUC, F1 and precision at k, and the training row and
        illicit transaction counts.
    """
    gnn_scores = crossfit.read_scores(config) if config.xgb.use_gnn else None
    result = train(
        config,
        gnn_scores=gnn_scores,
        train_days=split.xgb_training_days(config.split),
    )
    directory = artifacts.ensure_stage_dir(config, "train-xgb")
    result.bundle.booster.save_model(directory / BOOSTER_FILE)
    _write_json(directory / _FEATURE_COLUMNS_FILE, result.bundle.feature_columns)
    _write_json(directory / _METADATA_FILE, result.bundle.metadata)
    _write_json(directory / _METRICS_FILE, result.val_metrics)
    artifacts.write_manifest(
        config,
        "train-xgb",
        [BOOSTER_FILE, _FEATURE_COLUMNS_FILE, _METADATA_FILE, _METRICS_FILE],
    )
    return {
        "val_pr_auc": result.val_metrics["pr_auc"],
        "val_f1": result.val_metrics["f1"],
        "val_precision_at_k": result.val_metrics["precision_at_k"],
        "n_train_rows": float(result.n_train_rows),
        "n_train_positives": float(result.n_train_positives),
    }
