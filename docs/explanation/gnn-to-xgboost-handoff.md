# GNN to XGBoost hand-off

The `gnn` stage writes one score for each Transaction that a cross-fit fit scores. The `train-xgb` stage reads those scores as features. The scores are made in cross-fit folds, and each score comes from a model that did not train on its own day. The scores are logits, so they are not probabilities.

## What the GNN learns

The GNN is a supervised edge model with a [GraphSAGE](https://arxiv.org/abs/1706.02216) encoder. It reads a graph whose nodes are Accounts and whose edges are Transactions. An encoder gives each Account a state from its neighborhood. The edge head reads the states of the sender and the receiver, together with the attributes of the Transaction, and produces one logit for that Transaction and one embedding of `gnn.embedding_dim` values.

The encoder uses one message layer per entry in `gnn.neighbors`. The `gnn.encoder` key chooses the layer type, and [Encoder choice](#encoder-choice) compares the two types.

## Encoder choice

The `gnn.encoder` key sets the GraphSAGE layer type of the encoder. Both values use the same input layers, daily snapshots, neighbor sampling, loss and edge head. They differ in how a layer builds its messages.

| Value | Layer | What a layer computes |
| --- | --- | --- |
| `edge_sage` (default) | `EdgeSageConv`, defined in `sage_xgb/gnn/model.py` | For each edge, a linear map of the neighbor state and the edge encoding. Forward and reverse edges use separate maps. The mean of these messages is added to a linear map of the node's own state, and ReLU is applied. |
| `sage` | `SAGEConv` from PyTorch Geometric | A linear map of the mean of the neighbor states, added to a second linear map of the node's own state, and ReLU is applied. The messages carry no edge attributes. |

The edge encoding is a learned layer over the attributes of a Transaction. Under both values, the edge head reads the attributes of each Transaction. Under `sage`, the attributes reach the edge head but not the messages.

Each Transaction is stored as one forward edge and one reverse edge under both values. A forward message comes from a neighbor that sent money to the Account, and a reverse message comes from a neighbor that received money from the Account.

## Daily snapshots

The graph for day `d` is a snapshot. It holds the Transactions of the `gnn.history_days` days that end on the day before day `d - gnn.snapshot_lag_days`. The `snapshot_bounds` function in `sage_xgb/gnn/graph.py` gives the slice of the sorted table that the snapshot uses.

```python
def snapshot_bounds(table: EdgeTable, day: int, config: GnnConfig) -> tuple[int, int]:
    """Return the table slice [lo, hi) of the Transactions in Snapshot(day)."""
    horizon = day - config.snapshot_lag_days
    lo = int(np.searchsorted(table.day, horizon - config.history_days, side="left"))
    hi = int(np.searchsorted(table.day, horizon, side="left"))
    return lo, hi
```

With the default lag of `0` and the default history of 7 days, the snapshot for day 5 holds the Transactions of days 1 to 4. The Transactions of day 5 are not in it. A scored Transaction is therefore never part of the message passing that scores it, and neither is any later Transaction.

A lag of `1` leaves out the day before as well. A larger lag makes the graph older and the scores less fresh. Each day needs its own snapshot, so the stage builds one graph per scored day.

## Training on one day at a time

The fit trains one day at a time. For each day, the stage builds that day's snapshot and runs the sampled neighborhood of each labelled seed Transaction through it. The seed Transactions of the day are the only ones with a loss term.

The sampler keeps every positive Transaction of the day and a random share of the negatives. The `_sample_day` function in `sage_xgb/gnn/train.py` draws the share.

```python
def _sample_day(
    rows: TrainRows, day: int, neg_sample_rate: float, rng: np.random.Generator
) -> np.ndarray:
    """Return indices into `rows` for one day: every positive, some negatives."""
    in_day = np.flatnonzero(rows.day == day)
    labels = rows.label[in_day]
    negatives = in_day[labels == 0]
    kept = negatives[rng.random(len(negatives)) < neg_sample_rate]
    return np.concatenate([in_day[labels == 1], kept])
```

The draw runs again for each epoch, so each epoch sees a different set of negatives. The loss is a supervised binary cross-entropy on the logit, and the model has no early stopping.

## Cross-fit blocks

The training days are cut into blocks. The config sets the blocks in `split.crossfit_blocks`, and the shipped value is `[[1, 2], [3, 4], [5, 6]]`. The `fold_plan` function in `sage_xgb/gnn/crossfit.py` turns the blocks into fits.

```python
def fold_plan(split_config: SplitConfig, also_score: Sequence[str] = ()) -> list[Fit]:
    """Return the forward-chained fits, 1-based by `fit_id`.

    Args:
        split_config: Split with the cross-fit day blocks.
        also_score: Extra splits that every fit scores, for example `("val",)`.
    """
    blocks = split.crossfit_blocks(split_config)
    fits = []
    for index in range(1, len(blocks)):
        train_days = tuple(day for block in blocks[:index] for day in block)
        fits.append(Fit(index, train_days, tuple(blocks[index]), tuple(also_score)))
    final_splits = ("val", "test", "tail")
    extra = tuple(name for name in also_score if name not in final_splits)
    fits.append(
        Fit(
            len(blocks),
            tuple(split.training_days(split_config)),
            (),
            final_splits + extra,
        )
    )
    return fits
```

With the shipped blocks, the plan has three fits.

| Fit | Trains on days | Scores |
| --- | --- | --- |
| 1 | 1 to 2 | Days 3 and 4 |
| 2 | 1 to 4 | Days 5 and 6 |
| 3 | 1 to 6 | Validation, test and Tail rows |

The fits chain forward. Fit 1 never sees days 3 to 6, so its scores for days 3 and 4 are out of fold. Fit 2 never sees days 5 and 6. The final fit trains on the whole training window, and it scores the rows that come after training: validation, test and Tail.

Days 1 and 2 have no out-of-fold score, because no fit scores them. They appear only as training data for the fits. The `train-xgb` stage therefore trains on days 3 to 6 only.

## Out-of-fold check

After the fits run, the stage checks each score against the fit that made it. A row must come from a fit whose training days all come before the row's day.

```python
def assert_out_of_fold(
    scores: pl.DataFrame, days: pl.DataFrame, plan: Sequence[Fit]
) -> None:
    """Raise if any row was scored by a fit that trained on its day or later.

    Args:
        scores: Frame with `row_id` and `fit_id`.
        days: Frame with `row_id` and `day` for every scored row.
        plan: The fits that produced the scores.

    Raises:
        ValueError: A row's day is not after every training day of its fit.
    """
    last_train_day = {fit.fit_id: max(fit.train_days) for fit in plan}
    joined = scores.select("row_id", "fit_id").join(days, on="row_id", how="left")
    limit = pl.col("fit_id").replace_strict(last_train_day, return_dtype=pl.Int16)
    leaked = joined.filter(pl.col("day") <= limit)
    if leaked.height:
        msg = f"{leaked.height} rows were scored by a fit that trained on their day"
        raise ValueError(msg)
```

The check guards the stage against a future change to the plan. The shipped plan passes it by construction.

## What the scores hold

`gnn_score` is the raw logit of the edge head. It is not a probability. The model trains on all positives and on a sample of negatives, so the balance of the classes in training differs from the balance in the data. The logit therefore has no calibrated meaning. XGBoost reads it as a feature, and the decision threshold comes from the XGBoost validation scores, so no calibration step is needed.

The `gnn_emb_*` columns hold the edge head embedding of each Transaction. Each embedding combines the sender state, the receiver state and the Transaction attributes. Each embedding belongs to one Transaction.

## Scores from early fits

Early fits train on fewer days than the final fit. The scores for days 3 to 6 therefore come from models that saw less training data than the final model, which scores the validation and test rows. XGBoost learns from scores that are out of fold for each row. The validation and test scores come from the final fit, which saw the full training window.

## Next steps

- [Training pipeline](training-pipeline.md) shows where the `gnn` stage sits in the run.
- [Config reference](../reference/config.md) lists the `gnn` keys that shape the model.
- [Scale the GNN](../how-to/scale-the-gnn.md) explains which keys change the size of the model.
