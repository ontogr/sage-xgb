# Leakage rules

A model leaks when it uses information it would not have at the time of a scored Transaction. That information includes later Transactions, the label of the scored Transaction, and the labels of the test and Tail splits. The rules keep each of these out of the model. Each rule has code in the repository that enforces it, and [Tests that check these rules](#tests-that-check-these-rules) names the tests for the main rules.

## Order the Transactions by time

The raw CSV is not in time order. The `prepare` stage sorts the rows by timestamp, and it uses the raw row id to break ties. The sort is in `sage_xgb/data/prepare.py`, in the `run` function, as `raw.sort(["timestamp", "row_id"])`.

The row id is the zero-based index of the data row in the CSV. The tiebreak keeps the order the same on every run. No feature value depends on the row id, which only orders rows that share a timestamp.

## Count an Account by its bank and number

An Account is the pair of bank and account number. The number alone is not unique across banks. The `prepare` stage builds the Account key as the bank and the number joined with a colon, `src=pl.concat_str(["src_bank", "src_account"], separator=":")`, and it uses the same form for the receiver.

## Read earlier minutes only

Every window feature covers the interval `[t - w, t)`. The window includes the event exactly `w` before `t`, and it excludes every event in the minute `t` itself. The [features page](features.md) shows the code. Two Transactions in the same minute do not see each other. The window excludes the whole minute, so the features do not depend on the order of rows within a minute.

## Keep labels out of features

The `features` stage removes the label before it computes anything. The `run` function in `sage_xgb/features/table.py` reads the prepared table through the unlabelled loader and then builds the features.

```python
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
```

The `unlabelled_rows` function in `sage_xgb/split.py` drops the label column.

```python
def unlabelled_rows(frame: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    """Return all rows of every split with the label column removed."""
    return frame.drop("is_laundering")
```

The scan drops the label column before the table is collected, so no label is loaded. The `build_features` function never reads `is_laundering`, so the table is the same with or without that column.

## Ask for labels by split

Only the stages that fit a model read labels, and they ask for the splits they need. The `labelled_rows` function in `sage_xgb/split.py` returns the rows of the requested splits with the label column kept.

```python
def labelled_rows(
    frame: pl.DataFrame | pl.LazyFrame, splits: Sequence[str]
) -> pl.DataFrame | pl.LazyFrame:
    """Return the rows of the requested splits, with `is_laundering` kept.

    Args:
        frame: Prepared transactions with `split` and `is_laundering`.
        splits: Split names to keep. `test` and `tail` are refused.

    Raises:
        ValueError: A split name is unknown.
        HoldoutAccessError: `test` or `tail` was requested.
    """
    unknown = [name for name in splits if name not in SPLITS]
    if unknown:
        msg = f"unknown split names: {unknown}"
        raise ValueError(msg)
    held_out = [name for name in splits if name in HOLDOUT_SPLITS]
    if held_out:
        msg = f"labels of {held_out} may only be read by the report stage"
        raise HoldoutAccessError(msg)
    return frame.filter(pl.col("split").is_in(list(splits)))
```

The training stages ask for `train` and `val`. A call that asks for `test` or `tail` raises `HoldoutAccessError`. The `report` stage reads those two splits with its own reader in `sage_xgb/evaluation/report.py`, which is the only code that collects their labels.

Stages that need the Transactions of every split but no labels use the unlabelled loader, such as `features` and the GNN scoring.

## Keep the scored day out of its own graph

A GNN score for day `d` uses the snapshot of day `d`, and the snapshot holds only the earlier days. The `snapshot_bounds` function sets the slice of the table that the snapshot may read.

```python
def snapshot_bounds(table: EdgeTable, day: int, config: GnnConfig) -> tuple[int, int]:
    """Return the table slice [lo, hi) of the Transactions in Snapshot(day)."""
    horizon = day - config.snapshot_lag_days
    lo = int(np.searchsorted(table.day, horizon - config.history_days, side="left"))
    hi = int(np.searchsorted(table.day, horizon, side="left"))
    return lo, hi
```

The upper bound is `horizon`, and the search uses `side="left"`, so no row of day `horizon` or later is in the slice. A day-`d` Transaction is therefore absent from its own message passing, and so are the edges of every later day.

## Keep out-of-fold scores out of their own fit

Each GNN score for training rows comes from a fit that never trained on the row's day. The check runs after all fits finish, and it raises an error when a row fails. The [hand-off](gnn-to-xgboost-handoff.md#out-of-fold-check) quotes the check.

## Fix the GNN epochs

The GNN fit has no early stopping on validation. Validation labels are newer than the days that the early cross-fit folds score, so using them to stop a fold would let a later label shape an earlier score. The number of epochs comes from the config instead, and the same count applies to every fit.

## Use fixed vocabularies and rates

The codes for currencies and payment formats come from fixed tuples in `sage_xgb/data/vocab.py`. The USD rates are fixed constants too. The module says so in its docstring. Nothing is fitted from the data being processed, so a cut in the data cannot change a category code. A value outside the vocabulary stops the run, and the run does not map it to a fallback code.

## Keep the Patterns file out of the pipeline

The Patterns file lists the labelled laundering patterns. The pipeline never opens it, and the features never use it. The file is for evaluation and analysis only.

## Tests that check these rules

The end-to-end tests run the stages on the committed sample and check the behavior of each rule.

- `tests/e2e/test_features.py` removes the history after a cut time, recomputes the features, and checks that the remaining features do not change. It also checks that a same-minute Transaction does not change its pair, that the window edge counts the event exactly `w` earlier, that the row order does not change any feature, and that flipping every label leaves the features unchanged.
- `tests/e2e/test_run.py` flips the labels of the test and Tail splits and checks that the stage outputs do not change.
- `tests/e2e/test_tune.py` flips the test and Tail labels and checks that the tune study does not change.
- `tests/e2e/test_gnn.py` checks that snapshots hold no edge from the scored day or later, that the out-of-fold guard rejects a leaking score, and that future labels and edges do not change earlier scores.
- `tests/e2e/test_report.py` checks that test labels cannot be read outside the report stage.

## Known limits

Two limits remain.

- The prepared Transactions file holds the labels for every split. The protection is in the code that reads the file. The file itself has no protection.
- The features stage computes each window column in one pass over the whole prepared table. Each row still reads only earlier minutes of its own Accounts, and the truncation test checks that a row's features do not depend on later rows.
