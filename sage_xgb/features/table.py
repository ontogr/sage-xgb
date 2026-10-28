"""The feature table: row features next to the trailing-window features.

`build_features` is pure and works on any in-memory frame, so the leakage tests
can call it on perturbed copies. `run` reads the prepared Transactions, drops
the label and writes `features.parquet` for the `features` stage.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from sage_xgb import artifacts, split
from sage_xgb.config import Config
from sage_xgb.data import prepare, vocab
from sage_xgb.features.windows import (
    MINUTES_SINCE_LAST_COLUMNS,
    WINDOW_FEATURE_COLUMNS,
    window_features,
)


FEATURES_FILE = "features.parquet"

ROW_FEATURE_COLUMNS: tuple[str, ...] = (
    "hour",
    "weekday",
    "log_amount_usd",
    "amount_ratio",
    "same_bank",
    "is_cross_currency",
    "is_self_transfer",
    "paid_currency_code",
    "payment_format_code",
)
FEATURE_COLUMNS: tuple[str, ...] = (
    *ROW_FEATURE_COLUMNS,
    *WINDOW_FEATURE_COLUMNS,
    *MINUTES_SINCE_LAST_COLUMNS,
)


def _code(column: pl.Expr, vocabulary: Sequence[str]) -> pl.Expr:
    """Return the position of each value in `vocabulary`.

    The positions come from the fixed tuples, so the codes do not depend on the
    data. A value outside the vocabulary makes the expression raise.
    """
    return column.replace_strict(list(vocabulary), list(range(len(vocabulary))))


def _row_features(transactions: pl.DataFrame) -> pl.DataFrame:
    """Return `row_id` and the row features of each Transaction."""
    return transactions.select(
        "row_id",
        pl.col("timestamp").dt.hour().alias("hour"),
        pl.col("timestamp").dt.weekday().alias("weekday"),
        pl.col("amount_usd").log1p().alias("log_amount_usd"),
        pl.when(pl.col("amount_paid") != 0)
        .then(pl.col("amount_received") / pl.col("amount_paid"))
        .alias("amount_ratio"),
        (pl.col("src_bank") == pl.col("dst_bank")).alias("same_bank"),
        (pl.col("paid_currency") != pl.col("received_currency")).alias(
            "is_cross_currency"
        ),
        (pl.col("src") == pl.col("dst")).alias("is_self_transfer"),
        _code(pl.col("paid_currency"), vocab.CURRENCIES).alias("paid_currency_code"),
        _code(pl.col("payment_format"), vocab.PAYMENT_FORMATS).alias(
            "payment_format_code"
        ),
    )


def build_features(
    transactions: pl.DataFrame, exclude: Sequence[str] = ()
) -> pl.DataFrame:
    """Return the feature table of `transactions`, sorted by `row_id`.

    The window features are computed over the rows passed in, so a Transaction's
    features depend on the earlier Transactions in that frame. Rows in the frame
    may be in any order. The `is_laundering` column is never read, so it may be
    present or absent.

    Args:
        transactions: Prepared Transactions with `row_id`, `timestamp`, `src`,
            `dst`, `src_node`, `dst_node`, `src_bank`, `dst_bank`, `amount_paid`,
            `amount_received`, `paid_currency`, `received_currency`,
            `payment_format` and `amount_usd`.
        exclude: Feature column names to leave out of the table.

    Returns:
        One row per Transaction with `row_id` (UInt32) and the feature columns in
        contract order. Every feature column is Float32, and nulls are kept.

    Raises:
        ValueError: A name in `exclude` is not a feature column.
    """
    unknown = [name for name in exclude if name not in FEATURE_COLUMNS]
    if unknown:
        msg = f"exclude names unknown feature columns: {unknown}"
        raise ValueError(msg)
    kept = [name for name in FEATURE_COLUMNS if name not in exclude]
    return (
        _row_features(transactions)
        .join(window_features(transactions), on="row_id", validate="1:1")
        .select("row_id", *[pl.col(name).cast(pl.Float32) for name in kept])
        .sort("row_id")
    )


def run(config: Config) -> dict[str, float]:
    """Write `features.parquet` and its manifest for the prepared Transactions.

    The label is dropped before any feature is computed. The file is written
    with zstd compression and omits the `features.exclude` columns.

    Returns:
        `rows` is the number of Transactions and `columns` is the number of
        feature columns, which excludes `row_id`.
    """
    prepared = split.unlabelled_rows(prepare.scan_transactions(config)).collect()
    features = build_features(prepared, exclude=config.features.exclude)
    directory = artifacts.ensure_stage_dir(config, "features")
    features.write_parquet(directory / FEATURES_FILE, compression="zstd")
    artifacts.write_manifest(config, "features", [FEATURES_FILE])
    return {"rows": float(features.height), "columns": float(features.width - 1)}


def read_features(config: Config) -> pl.DataFrame:
    """Read the `features.parquet` written by `run`."""
    return pl.read_parquet(artifacts.stage_path(config, "features", FEATURES_FILE))
