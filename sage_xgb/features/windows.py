"""Trailing-window features for each Account role.

Each Transaction becomes two events: an `out` event for its sender and an `in`
event for its receiver. The events of one Account form one stream. Every event
row carries the out and in statistics of its Account, so a Transaction reads
its sender's `in` values and its receiver's `out` values from the same kind of
row. A window is the half-open interval [t - w, t) on minute timestamps, so
Transactions in one minute never count each other.
"""

from __future__ import annotations

from datetime import timedelta

import polars as pl


SIDES: tuple[str, ...] = ("src", "dst")
ROLES: tuple[str, ...] = ("out", "in")
STATS: tuple[str, ...] = ("count", "usd_sum", "usd_max")
WINDOWS: dict[str, timedelta] = {
    "1h": timedelta(hours=1),
    "24h": timedelta(hours=24),
    "7d": timedelta(days=7),
}

WINDOW_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    f"{side}_{role}_{stat}_{window}"
    for side in SIDES
    for role in ROLES
    for stat in STATS
    for window in WINDOWS
)
MINUTES_SINCE_LAST_COLUMNS: tuple[str, ...] = tuple(
    f"{side}_{role}_minutes_since_last" for side in SIDES for role in ROLES
)

_EVENT_COLUMNS: tuple[str, ...] = tuple(
    [
        f"{role}_{stat}_{window}"
        for role in ROLES
        for stat in STATS
        for window in WINDOWS
    ]
    + [f"{role}_minutes_since_last" for role in ROLES]
)


def _is_role(role: str) -> pl.Expr:
    """Return a boolean expression that is true on events of `role`."""
    if role == "out":
        return pl.col("is_out")
    return ~pl.col("is_out")


def _event_stream(transactions: pl.DataFrame) -> pl.LazyFrame:
    """Return two events per Transaction, sorted by minute, row id and side.

    Sorting by minute first lets the rolling windows and the asof joins read
    each Account in time order. The row id and the side flag break ties, so the
    order does not depend on the input order.
    """
    frame = transactions.lazy()
    sent = frame.select(
        "row_id",
        "timestamp",
        pl.lit(True).alias("is_src"),
        pl.lit(True).alias("is_out"),
        pl.col("src_node").alias("account"),
        pl.col("amount_usd").alias("usd"),
    )
    received = frame.select(
        "row_id",
        "timestamp",
        pl.lit(False).alias("is_src"),
        pl.lit(False).alias("is_out"),
        pl.col("dst_node").alias("account"),
        pl.col("amount_usd").alias("usd"),
    )
    return pl.concat([sent, received]).sort("timestamp", "row_id", "is_src")


def _rolling_columns() -> list[pl.Expr]:
    """Return the trailing count, sum and max of each role for every window.

    Each expression runs per Account over the window [t - w, t). The event at
    exactly t - w is included and events in the same minute are not. Count and
    sum use inputs that are 0 on the other role, so a window with no event sums
    to 0. Max uses an input that is null on the other role, and a rolling max
    skips nulls, so a window with no event of the role gives null.
    """
    columns: list[pl.Expr] = []
    for window, duration in WINDOWS.items():
        for role in ROLES:
            is_role = _is_role(role)
            count = is_role.cast(pl.Float64)
            usd = pl.when(is_role).then(pl.col("usd")).otherwise(0.0)
            usd_max = pl.when(is_role).then(pl.col("usd"))
            columns += [
                count.rolling_sum_by("timestamp", duration, closed="left")
                .over("account")
                .cast(pl.Float32)
                .alias(f"{role}_count_{window}"),
                usd.rolling_sum_by("timestamp", duration, closed="left")
                .over("account")
                .cast(pl.Float32)
                .alias(f"{role}_usd_sum_{window}"),
                usd_max.rolling_max_by("timestamp", duration, closed="left")
                .over("account")
                .cast(pl.Float32)
                .alias(f"{role}_usd_max_{window}"),
            ]
    return columns


def _with_minutes_since_last(events: pl.LazyFrame, role: str) -> pl.LazyFrame:
    """Add the minutes since the previous event of `role` on the same Account.

    The previous event must come from an earlier minute, so a Transaction never
    sees a same-minute event here. The result is null when the Account has no
    earlier event of that role.
    """
    previous = events.filter(_is_role(role)).select(
        "account", "timestamp", pl.col("timestamp").alias("previous_timestamp")
    )
    # Polars cannot check sortedness of a join with `by`. The event stream is
    # sorted in _event_stream, so the check is skipped.
    return (
        events.join_asof(
            previous,
            on="timestamp",
            by="account",
            strategy="backward",
            allow_exact_matches=False,
            check_sortedness=False,
        )
        .with_columns(
            (pl.col("timestamp") - pl.col("previous_timestamp"))
            .dt.total_minutes()
            .cast(pl.Float32)
            .alias(f"{role}_minutes_since_last")
        )
        .drop("previous_timestamp")
    )


def _side_frame(events: pl.DataFrame, side: str) -> pl.DataFrame:
    """Return one row per Transaction for `side`, with prefixed column names."""
    on_side = pl.col("is_src") if side == "src" else ~pl.col("is_src")
    return events.filter(on_side).select(
        "row_id", *[pl.col(name).alias(f"{side}_{name}") for name in _EVENT_COLUMNS]
    )


def window_features(transactions: pl.DataFrame) -> pl.DataFrame:
    """Return the trailing-window features of every Transaction.

    Args:
        transactions: Transactions with `row_id` (unique), `timestamp` (minute
            resolution), `src_node`, `dst_node` and `amount_usd`. The rows may
            come in any order.

    Returns:
        One row per Transaction, sorted by `row_id`. The columns are `row_id`,
        then the `WINDOW_FEATURE_COLUMNS` and `MINUTES_SINCE_LAST_COLUMNS` names
        as Float32. A window with no event gives count 0, sum 0 and a null max.
        Minutes since last are null when the Account has no earlier event of
        that role.
    """
    events = _event_stream(transactions)
    for role in ROLES:
        events = _with_minutes_since_last(events, role)
    events = events.with_columns(_rolling_columns())
    collected = events.select("row_id", "is_src", *_EVENT_COLUMNS).collect()
    return (
        _side_frame(collected, "src")
        .join(_side_frame(collected, "dst"), on="row_id", validate="1:1")
        .select("row_id", *WINDOW_FEATURE_COLUMNS, *MINUTES_SINCE_LAST_COLUMNS)
        .sort("row_id")
    )
