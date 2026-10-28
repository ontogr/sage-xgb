"""The only module that turns a timestamp into a split or a day block.

Every other module asks this one. Test and Tail labels stay unopened until
`report` asks for them with `allow_holdout=True`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import overload

import polars as pl

from sage_xgb.config import SplitConfig


SPLITS: tuple[str, ...] = ("train", "val", "test", "tail")
HOLDOUT_SPLITS: tuple[str, ...] = ("test", "tail")


class HoldoutAccessError(PermissionError):
    """Error for a read of test or Tail labels without permission."""


def day_expr(timestamp: pl.Expr, split: SplitConfig) -> pl.Expr:
    """Return the 1-based day number since `split.window_start`."""
    elapsed = timestamp.dt.date() - pl.lit(split.window_start)
    return (elapsed.dt.total_days() + 1).cast(pl.Int16)


def split_expr(timestamp: pl.Expr, split: SplitConfig) -> pl.Expr:
    """Return the split name of each timestamp: train, val, test or tail."""
    date = timestamp.dt.date()
    return (
        pl.when(date < split.train_end)
        .then(pl.lit("train"))
        .when(date < split.val_end)
        .then(pl.lit("val"))
        .when(date < split.test_end)
        .then(pl.lit("test"))
        .otherwise(pl.lit("tail"))
    )


@overload
def add_day_and_split(frame: pl.DataFrame, split: SplitConfig) -> pl.DataFrame: ...
@overload
def add_day_and_split(frame: pl.LazyFrame, split: SplitConfig) -> pl.LazyFrame: ...
def add_day_and_split(
    frame: pl.DataFrame | pl.LazyFrame, split: SplitConfig
) -> pl.DataFrame | pl.LazyFrame:
    """Add the `day` and `split` columns computed from `timestamp`."""
    return frame.with_columns(
        day=day_expr(pl.col("timestamp"), split),
        split=split_expr(pl.col("timestamp"), split),
    )


def crossfit_blocks(split: SplitConfig) -> list[list[int]]:
    """Return the cross-fit day blocks as lists of 1-based day numbers."""
    return [list(block) for block in split.crossfit_blocks]


def training_days(split: SplitConfig) -> list[int]:
    """Return every training day, in order."""
    return [day for block in split.crossfit_blocks for day in block]


def xgb_training_days(split: SplitConfig) -> list[int]:
    """Return the days XGBoost trains on: every block after the first.

    The first block only provides history and the first GNN fit, so it has no
    out-of-fold score.
    """
    return [day for block in split.crossfit_blocks[1:] for day in block]


@overload
def labelled_rows(
    frame: pl.DataFrame, splits: Sequence[str], *, allow_holdout: bool = False
) -> pl.DataFrame: ...
@overload
def labelled_rows(
    frame: pl.LazyFrame, splits: Sequence[str], *, allow_holdout: bool = False
) -> pl.LazyFrame: ...
def labelled_rows(
    frame: pl.DataFrame | pl.LazyFrame,
    splits: Sequence[str],
    *,
    allow_holdout: bool = False,
) -> pl.DataFrame | pl.LazyFrame:
    """Return the rows of the requested splits, with `is_laundering` kept.

    Args:
        frame: Prepared transactions with `split` and `is_laundering`.
        splits: Split names to keep.
        allow_holdout: Must be true to ask for `test` or `tail`. Only `report`
            sets it.

    Raises:
        ValueError: A split name is unknown.
        HoldoutAccessError: `test` or `tail` was requested without permission.
    """
    unknown = [name for name in splits if name not in SPLITS]
    if unknown:
        msg = f"unknown split names: {unknown}"
        raise ValueError(msg)
    held_out = [name for name in splits if name in HOLDOUT_SPLITS]
    if held_out and not allow_holdout:
        msg = f"labels of {held_out} may only be read by the report stage"
        raise HoldoutAccessError(msg)
    return frame.filter(pl.col("split").is_in(list(splits)))


@overload
def unlabelled_rows(frame: pl.DataFrame) -> pl.DataFrame: ...
@overload
def unlabelled_rows(frame: pl.LazyFrame) -> pl.LazyFrame: ...
def unlabelled_rows(frame: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    """Return all rows of every split with the label column removed."""
    return frame.drop("is_laundering")
