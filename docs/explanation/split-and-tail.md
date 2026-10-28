# Split and Tail

Every Transaction belongs to one of four splits: `train`, `val`, `test` or `tail`. The split comes from the calendar day of the Transaction, and the day boundaries come from the config. The `sage_xgb/split.py` module is the only code that turns a timestamp into a split or a day number.

## Day boundaries

The config gives the first day of the Variant and three exclusive end dates. The `split.py` module numbers the days from the first day, starting at 1.

```python
def day_expr(timestamp: pl.Expr, split: SplitConfig) -> pl.Expr:
    """Return the 1-based day number since `split.window_start`."""
    elapsed = timestamp.dt.date() - pl.lit(split.window_start)
    return (elapsed.dt.total_days() + 1).cast(pl.Int16)
```

The split comes from the same date, compared with the end dates.

```python
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
```

With the shipped HI-Small config, the training split covers days 1 to 6, the validation split covers days 7 and 8, and the test split covers days 9 and 10. The Tail covers every later day. The [config reference](../reference/config.md) gives the dates.

## Why whole days

Every split starts and ends on a day boundary, so no day belongs to two splits. Each split holds whole days, and the GNN snapshots are built from whole days too. A snapshot for day `d` holds the Transactions of the earlier days in its history window, so a day-level cut gives an exact graph for each day. A cut inside a day would need a partial snapshot, and the rules for that partial day would be harder to check.

The boundaries are chosen so the training, validation and test splits hold about 60, 20 and 20 percent of the rows. The choice follows from the day counts, so a Variant gets its own boundaries. The [add a Variant](../how-to/add-a-variant.md) how-to explains how to choose them.

## The Tail

The Tail is every Transaction on or after `split.test_end`. A Tail Transaction counts as history only for later Tail Transactions, because no other split comes after it. The model never trains on it, and the validation threshold never uses it.

The report stage scores the Tail slice on its own. Rows after the nominal window can have a different illicit rate and a different mix of patterns. A model could score well on them for reasons that do not carry over to the window. Keeping the Tail in its own slice lets the test slice stay a fixed window.

## Test labels are read once

The test and Tail labels are read only by the report stage. The `labelled_rows` function in `split.py` refuses those splits, and the report stage reads them with its own reader.

The report stage appends one line to `report/history.jsonl` for each slice it reads, and writes the line before it scores the slice. The log prints the number of lines after each read. The file is the count of times the labels of a slice have been read.

The rule is enforced in code. The prepared Transactions file holds the labels for every split, so a person who opens the file can see them. The code keeps labels out of the model as described in [Leakage rules](leakage-rules.md).

## Cross-fit blocks

The training days also form the `crossfit_blocks` in the config. Each block is a run of days, and the blocks cover the training days in order. The GNN stage uses the blocks to make out-of-fold scores, which the [hand-off page](gnn-to-xgboost-handoff.md) explains. The validation check in the config requires at least two blocks, and it requires the blocks to list every training day once.

## Next steps

- [Leakage rules](leakage-rules.md) explains the guards around the split.
- [Features](features.md) explains why the windows never reach past the Transaction's own minute.
