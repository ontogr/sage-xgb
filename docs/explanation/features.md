# Features

The `features` stage writes one row for each Transaction. The row holds nine row features, computed from the Transaction alone. It also holds 36 window features and four minutes-since-last features, computed from the earlier Transactions of its two Accounts. Every window covers a half-open interval of minutes that ends just before the Transaction's own minute. No feature reads a label.

## Row features

| Column | Content |
| --- | --- |
| `hour` | Hour of the day of the minute timestamp, from 0 to 23 |
| `weekday` | Day of the week from the timestamp |
| `log_amount_usd` | Natural log of one plus the USD amount |
| `amount_ratio` | Received amount divided by paid amount, or null when the paid amount is zero |
| `same_bank` | True when the sender and the receiver have the same bank |
| `is_cross_currency` | True when the paid currency differs from the received currency |
| `is_self_transfer` | True when the sender and the receiver are the same Account |
| `paid_currency_code` | Position of the paid currency in the fixed currency list |
| `payment_format_code` | Position of the payment format in the fixed format list |

The codes come from the fixed tuples in `sage_xgb/data/vocab.py`. A value outside those tuples fails the run.

## Two events for each Transaction

A Transaction becomes two events. The sender gets an `out` event, and the receiver gets an `in` event. The events of one Account form one stream, sorted by minute. The `_event_stream` function in `sage_xgb/features/windows.py` builds the two events.

```python
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
```

Each Transaction takes its `src` columns from its sender's event and its `dst` columns from its receiver's event. Each of those events carries the statistics of its own Account, so one row holds the sender's values and the receiver's values side by side.

## Window features

The windows are 1 hour, 24 hours and 7 days. Each window has three statistics for each role: a count, a sum of USD amounts and a maximum USD amount. The column name combines the parts in this order: side, role, statistic and window. For example, `src_out_count_1h` is the number of sent Transactions of the sender in the last hour, and `dst_in_usd_max_7d` is the largest received amount of the receiver in the last week.

The `_rolling_columns` function computes the statistics.

```python
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
```

The `closed="left"` argument makes each window run from `t - w` up to, but not including, `t`. An event exactly `w` before `t` counts, and an event at `t` does not. Because the window ignores the events of its own minute, a Transaction never counts itself or any other Transaction in its minute.

A window with no event gives count 0 and sum 0. The max of such a window is null, because the rolling max skips the null inputs of the other role.

## Minutes since the last event

The `minutes_since_last` columns give the gap in minutes from the previous event of the same role on the same Account. The `_with_minutes_since_last` function finds the previous event with an as-of join.

```python
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
```

The `allow_exact_matches=False` argument makes the join skip events in the same minute. The result is null when no earlier event of that role exists.

## Whole-table computation

The `build_features` function computes the window columns in one vectorized pass over the whole prepared table. Each row still reads only events from earlier minutes of its own Accounts, so the output for a row does not depend on rows that come after it.

The end-to-end test in `tests/e2e/test_features.py` checks this property. It removes the rows after a cut time, recomputes the features, and checks that the remaining features are unchanged. The test that flips every label checks that the labels play no part.

## Output

The `features.parquet` file holds the `row_id` column, which is the key of each Transaction, followed by the 49 feature columns. Every feature column is Float32. The [artifact reference](../reference/artifacts.md) lists the groups.

The `features.exclude` config key removes names from the file. The name must be one of the feature columns. A name that is not a feature column fails the stage. The [config reference](../reference/config.md) describes the key.

## Next steps

- [Leakage rules](leakage-rules.md) lists the other rules that protect the features.
- [GNN to XGBoost hand-off](gnn-to-xgboost-handoff.md) explains the GNN columns that join the features.
