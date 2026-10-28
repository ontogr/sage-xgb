"""Read the raw AMLworld Transactions CSV into typed columns.

The raw header names two columns `Account`, so the columns are renamed by
position. Banks and account numbers stay strings, so leading zeros survive.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from sage_xgb.data.vocab import CURRENCIES, PAYMENT_FORMATS, VocabularyError


_RAW_SCHEMA = {
    "timestamp": pl.String,
    "src_bank": pl.String,
    "src_account": pl.String,
    "dst_bank": pl.String,
    "dst_account": pl.String,
    "amount_received": pl.Float64,
    "received_currency": pl.String,
    "amount_paid": pl.Float64,
    "paid_currency": pl.String,
    "payment_format": pl.String,
    "is_laundering": pl.Int8,
}

_VOCABULARIES: dict[str, tuple[str, ...]] = {
    "received_currency": CURRENCIES,
    "paid_currency": CURRENCIES,
    "payment_format": PAYMENT_FORMATS,
}


def scan_raw(raw_dir: Path, name: str) -> pl.LazyFrame:
    """Return the Transactions CSV of a Variant as a lazy frame.

    Args:
        raw_dir: Directory that holds the raw AMLworld files.
        name: Variant name, such as `HI-Small`. The file read is
            `{name}_Trans.csv`.

    Returns:
        One row per CSV data row, with `row_id` (UInt32) first. `row_id` is the
        0-based data-row index, assigned before any sort or filter runs.
        `timestamp` is parsed to a datetime with minute resolution.
    """
    # The header repeats `Account`, so the column names come from this module
    # and the header line is skipped.
    return (
        pl.scan_csv(
            raw_dir / f"{name}_Trans.csv",
            has_header=False,
            skip_rows=1,
            new_columns=list(_RAW_SCHEMA),
            schema_overrides=list(_RAW_SCHEMA.values()),
        )
        .with_row_index("row_id")
        .with_columns(
            pl.col("timestamp").str.to_datetime("%Y/%m/%d %H:%M", time_unit="us")
        )
    )


def check_vocabulary(frame: pl.LazyFrame) -> None:
    """Raise if a categorical column holds a value outside its vocabulary.

    The check reads the distinct values of the currency and payment format
    columns in one pass over the file. Nothing is mapped to a fallback code.

    Args:
        frame: Raw transactions from `scan_raw`.

    Raises:
        VocabularyError: A column holds a value that the vocabulary does not
            list. The message names the column and the unknown values.
    """
    distinct = (
        frame.select([pl.col(column).unique().implode() for column in _VOCABULARIES])
        .collect()
        .to_dicts()[0]
    )
    for column, allowed in _VOCABULARIES.items():
        unknown = sorted(
            (value for value in distinct[column] if value not in allowed), key=str
        )
        if unknown:
            msg = (
                f"unknown values {unknown!r} in column {column!r}; "
                "add them to sage_xgb.data.vocab"
            )
            raise VocabularyError(msg)
