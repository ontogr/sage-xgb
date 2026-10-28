# Scale the GNN

The `gnn` section of the config sets the size of the graph neural network and how much work each GNN fit does. `gnn.hidden`, `gnn.layers`, `gnn.encoder` and `gnn.embedding_dim` set the number of parameters. The other keys set the work per fit, such as the number of sampled neighbors, the batch size, the number of passes and the share of negatives kept.

## Prerequisites

- A config file for the Variant. The `gnn` section of `configs/hi_small.yaml` and `configs/sample.yaml` holds the values that the configs use.
- Enough free memory for the setting you choose. Larger batches and wider layers need more memory.

## Choose the keys to change

| Key | Default | Larger value | Smaller value |
| --- | --- | --- | --- |
| `gnn.hidden` | `32` | Wider node and edge states, more parameters, more work per message | Narrower states, fewer parameters, less work per message |
| `gnn.neighbors` | `[5, 3]` | More sampled neighbors per layer, larger subgraphs per batch | Fewer sampled neighbors, smaller subgraphs per batch |
| `gnn.neg_sample_rate` | `0.02` | More negative Transactions kept each epoch | Fewer negative Transactions kept each epoch |
| `gnn.epochs` | `1` | More passes over the training days | Fewer passes over the training days |

The defaults come from `sage_xgb/config.py`. The HI-Small config sets `gnn.hidden` to 64, `gnn.neighbors` to `[10, 5]`, `gnn.neg_sample_rate` to 0.05 and `gnn.epochs` to 2, so its GNN is larger and does more work per fit than the defaults.

A larger `hidden` or `layers` value gives a bigger model. A larger `neighbors` list makes each batch hold a larger subgraph. A higher `neg_sample_rate` or a higher `epochs` gives each fit more training examples and more passes, so each fit takes longer.

`neighbors` must have one entry for each layer in `gnn.layers`. The config check rejects a list whose length differs from `layers`.

## Choose the encoder

`gnn.encoder` sets the GraphSAGE layer type of the message-passing layers. The value `edge_sage` uses `EdgeSageConv`, and the value `sage` uses PyTorch Geometric's `SAGEConv`. The input layers and the edge head have the same parameters under both values, so only the message-passing layers change.

A layer with width `h`, where `h` is `gnn.hidden`, has these parameters.

| Value | Layer | Parameters per layer |
| --- | --- | --- |
| `edge_sage` | `EdgeSageConv` | `5h^2 + 3h` |
| `sage` | `SAGEConv` | `2h^2 + h` |

With `gnn.hidden` set to 64 and `gnn.layers` set to 2, the message-passing layers hold 41,344 parameters under `edge_sage` and 16,512 under `sage`.

`sage` has fewer parameters per layer, but its messages do not carry the Transaction attributes. `edge_sage` carries them in its messages and has more parameters per layer.

Changing `gnn.encoder` reruns the GNN stage. A `gnn_fit_<n>.pt` file loads only into a model built with the same encoder.

## Change the training work

`gnn.neg_sample_rate` sets the fraction of negative training Transactions that each epoch keeps. Every positive Transaction is kept in every epoch. The sample is drawn again for each epoch, so each epoch sees a different set of negatives. The value must be greater than 0 and at most 1.

`gnn.epochs` sets the number of passes over the training days. The GNN has no early stopping on validation. The epoch count is therefore fixed in the config, and the value you pick is the number of passes that every fit makes.

`gnn.batch_size` sets the number of seed Transactions in each training batch. A smaller batch uses less memory for each step. A larger batch takes fewer steps.

`gnn.score_batch_size` sets the number of Accounts in each batch when the stage scores Transactions. Scoring does not change the model.

`gnn.lr` sets the learning rate of the Adam optimizer.

## Change the outputs

`gnn.embedding_dim` sets the width of the per-Transaction embedding from the edge head. The GNN stage writes one `gnn_emb_*` column for each dimension, and `train-xgb` uses each column as a feature. A larger value gives XGBoost more input columns.

`gnn.snapshot_lag_days` and `gnn.history_days` set which earlier Transactions form each day's graph. Changing either one changes the scores of every fit, so the GNN stage reruns. The graph of each day is defined in [Daily snapshots](../explanation/gnn-to-xgboost-handoff.md#daily-snapshots).

## Apply a change

Edit the `gnn` section of the config, or override one key on the command line.

```bash
uv run sage-xgb run --config configs/hi_small.yaml --set gnn.epochs=3
```

The GNN stage reruns when a key in the `gnn` section changes, because the stage hashes that section. The `train-xgb` stage reruns after it, because its upstream manifest changed. The `prepare` and `features` stages do not rerun.

## Check the change

Confirm that the stage wrote its files. The `gnn` directory holds `gnn_scores.parquet` and one `gnn_fit_<n>.pt` file for each fit. The [artifact reference](../reference/artifacts.md) lists the files.

## Next steps

- [Config reference](../reference/config.md) lists every `gnn` key with its default and validation rule.
- [Tune hyperparameters](tune-hyperparameters.md) searches the GNN keys with a time limit.
