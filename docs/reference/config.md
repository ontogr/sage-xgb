# Config reference

A run config is one YAML file per Variant. The file loads into a validated model at startup, so an unknown key or a wrong type stops the command before any stage runs. The defaults come from `sage_xgb/config.py`.

## Loading rules

- A top-level `extends` key names a parent file, relative to the child file. The child merges onto the parent. Mappings merge key by key, and lists and scalars in the child replace the parent value. The parent cannot itself use `extends`.
- Each `--set KEY=VALUE` override applies after the merge. The key is a dotted path, and the value parses as YAML.
- Unknown keys fail validation. The section models forbid extra keys, so a misspelled key is an error.

## Top-level keys

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `variant` | string | None, required | Name of the Variant. It names the artifacts directory and the tune study. |
| `seed` | integer | `7` | Seed for XGBoost, the GNN fits, the negative sampler, the tune study sampler and the bootstrap intervals in the report. |

## paths

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `paths.artifacts_dir` | path | `artifacts` | Root folder of all outputs. Relative paths resolve from the working directory. |

## data

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `data.name` | string | `HI-Small` | Prefix of the raw file names, such as `HI-Small_Trans.csv`. |
| `data.raw_dir` | path | `data` | Folder that holds the raw files. |
| `data.kaggle_dataset` | string | `ealtman2019/ibm-transactions-for-anti-money-laundering-aml` | Kaggle dataset that `download` reads. |

## split

Every date is a calendar day, and the end dates are exclusive.

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `split.window_start` | date | None, required | First day of the Variant. Day 1 is this day. |
| `split.train_end` | date | None, required | First day after the training split. |
| `split.val_end` | date | None, required | First day after the validation split. |
| `split.test_end` | date | None, required | First day after the test split. Rows from this day on are the Tail. |
| `split.crossfit_blocks` | list of lists of integers | None, required | Cross-fit day blocks, as 1-based day numbers. |

The model checks these rules when the file loads.

- The dates must satisfy `window_start < train_end < val_end < test_end`.
- `crossfit_blocks` must hold at least two blocks. Flattened, the blocks must list the days 1 to the number of training days, each once and in order.

Each block holds consecutive days, and the blocks follow each other in day order.

## features

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `features.exclude` | list of strings | `[]` | Feature column names to leave out of `features.parquet`. A name that is not a feature column fails the features stage. |

The [features page](../explanation/features.md) lists the feature column names.

## xgb

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `xgb.use_gnn` | boolean | `true` | When true, `train-xgb` reads the GNN scores and adds their columns to the features. When false, `train-xgb` reads no GNN output. The `run` command still checks the `gnn` stage in both cases, and runs it when its manifest is stale. |
| `xgb.n_estimators` | integer | `400` | Maximum number of boosting rounds. |
| `xgb.early_stopping_rounds` | integer | `30` | Number of rounds without a gain in validation PR-AUC before training stops. |
| `xgb.max_depth` | integer | `6` | Maximum depth of each tree. |
| `xgb.learning_rate` | float | `0.1` | Step size of each round, written as `eta` to XGBoost. |
| `xgb.subsample` | float | `0.8` | Fraction of training rows that each round samples. |
| `xgb.colsample_bytree` | float | `0.8` | Fraction of feature columns that each tree samples. |
| `xgb.min_child_weight` | float | `1.0` | Minimum sum of instance weight in a child node. |
| `xgb.reg_lambda` | float | `1.0` | L2 regularization on leaf weights, written as `lambda` to XGBoost. |
| `xgb.scale_pos_weight` | float | `1.0` | Weight of the positive class in the loss. |
| `xgb.num_threads` | integer | `8` | Threads for XGBoost. |

Training uses the `binary:logistic` objective, the `hist` tree method and the `aucpr` evaluation metric.

## gnn

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `gnn.hidden` | integer | `32` | Width of the node and edge states. The edge head reads three states of this width. |
| `gnn.layers` | integer | `2` | Number of message-passing layers. Must equal the length of `gnn.neighbors`. |
| `gnn.encoder` | `edge_sage` or `sage` | `edge_sage` | GraphSAGE layer type of the Account encoder. `edge_sage` is `EdgeSageConv`, whose messages carry the Transaction attributes and use separate weights for forward and reverse edges. `sage` is PyTorch Geometric's `SAGEConv`, whose messages carry neighbor states only. The edge head reads the Transaction attributes in both cases. |
| `gnn.embedding_dim` | integer | `8` | Width of the per-Transaction embedding from the edge head. The `train-xgb` stage uses each dimension as a feature column. |
| `gnn.neighbors` | list of integers | `[5, 3]` | Neighbors sampled for each layer. Must hold one entry for each layer. |
| `gnn.batch_size` | integer | `2048` | Seed Transactions in each training batch. |
| `gnn.score_batch_size` | integer | `8192` | Accounts in each batch when the stage scores Transactions. |
| `gnn.lr` | float | `0.005` | Learning rate of the Adam optimizer. |
| `gnn.epochs` | integer | `1` | Passes over the training days of each fit. There is no early stopping. |
| `gnn.neg_sample_rate` | float | `0.02` | Fraction of negative training Transactions kept in each epoch. Must be greater than 0 and at most 1. Every positive is kept. |
| `gnn.snapshot_lag_days` | integer | `0` | Days skipped between the end of a snapshot and the day it scores. `0` ends the snapshot on the day before the scored day. |
| `gnn.history_days` | integer | `7` | Number of days in each snapshot window. |
| `gnn.num_threads` | integer | `8` | Threads for PyTorch. |
| `gnn.num_workers` | integer | `0` | Sampler worker processes for training batches. `0` samples in the main process. |

The snapshot rule is in [Daily snapshots](../explanation/gnn-to-xgboost-handoff.md#daily-snapshots), and the fits are in [Cross-fit blocks](../explanation/gnn-to-xgboost-handoff.md#cross-fit-blocks).

## tracking

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `tracking.enabled` | boolean | `true` | When false, the stages make no MLflow calls. |
| `tracking.experiment` | string | `sage-xgb` | MLflow experiment that receives the runs. |

## tune

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `tune.n_trials` | integer | `20` | Number of trials that one study aims to reach. Completed and pruned trials count toward it. Failed trials and trials left running by a stopped process do not. |
| `tune.timeout_minutes` | float | `10.0` | Time limit of one call to the tune stage, in minutes. The limit starts when the call starts and includes the shared stages. A fit or training step that has started is not interrupted. |
| `tune.space` | mapping | The default space, with nine keys | Searched config paths, each with a range. |

Each `tune.space` entry has these fields.

| Field | Type | Meaning |
| --- | --- | --- |
| `kind` | `int`, `float`, `log_float` or `categorical` | Distribution of the sampled value |
| `low` | number | Lower bound, required for the numeric kinds. It must not exceed `high`, it must be above 0 for `log_float`, and it must be a whole number for `int` |
| `high` | number | Upper bound, required for the numeric kinds. It must be a whole number for `int` |
| `choices` | list | Allowed values, required for `categorical`. The list must not be empty |

The keys of `tune.space` are dotted config paths. The default space searches these keys.

| Key | Kind | Range or choices |
| --- | --- | --- |
| `gnn.hidden` | `categorical` | `16`, `32`, `64` |
| `gnn.lr` | `log_float` | `0.001` to `0.02` |
| `gnn.neg_sample_rate` | `log_float` | `0.01` to `0.2` |
| `xgb.max_depth` | `int` | `3` to `9` |
| `xgb.learning_rate` | `log_float` | `0.03` to `0.3` |
| `xgb.subsample` | `float` | `0.6` to `1.0` |
| `xgb.colsample_bytree` | `float` | `0.5` to `1.0` |
| `xgb.min_child_weight` | `log_float` | `1.0` to `20.0` |
| `xgb.reg_lambda` | `log_float` | `0.1` to `10.0` |

A `tune.space` mapping in a file replaces the whole default space. The `kind` value, the bounds and the choices of each entry are checked when the config loads. A key that is not a config field fails when the tune stage starts a trial. The `xgb.use_gnn` key cannot be searched.

## Values in the shipped configs

These values apply after `extends` merges the parent file.

| Key | Default | `configs/hi_small.yaml` | `configs/sample.yaml` |
| --- | --- | --- | --- |
| `variant` | None | `hi-small` | `hi-small-sample` |
| `data.name` | `HI-Small` | `HI-Small` | `HI-Small` |
| `data.raw_dir` | `data` | `data` | `tests/fixtures/amlworld_hi_small_sample` |
| `split.window_start` | None | `2022-09-01` | `2022-09-01` |
| `split.train_end` | None | `2022-09-07` | `2022-09-07` |
| `split.val_end` | None | `2022-09-09` | `2022-09-09` |
| `split.test_end` | None | `2022-09-11` | `2022-09-11` |
| `split.crossfit_blocks` | None | `[[1, 2], [3, 4], [5, 6]]` | `[[1, 2], [3, 4], [5, 6]]` |
| `xgb.n_estimators` | `400` | `400` | `80` |
| `xgb.early_stopping_rounds` | `30` | `30` | `10` |
| `xgb.num_threads` | `8` | `8` | `2` |
| `gnn.hidden` | `32` | `64` | `16` |
| `gnn.neighbors` | `[5, 3]` | `[10, 5]` | `[5, 3]` |
| `gnn.batch_size` | `2048` | `2048` | `512` |
| `gnn.score_batch_size` | `8192` | `8192` | `2048` |
| `gnn.epochs` | `1` | `2` | `2` |
| `gnn.neg_sample_rate` | `0.02` | `0.05` | `0.2` |
| `gnn.num_threads` | `8` | `8` | `1` |
| `tune.n_trials` | `20` | `20` | `3` |
| `tune.timeout_minutes` | `10.0` | `10.0` | `2` |

The keys that are not in this table keep their default value in both shipped configs.

## Environment settings

Credentials and service addresses are not config keys. The `KaggleSettings` and `MlflowSettings` classes read these variables from the environment or from the `.env` file in the repository root.

| Variable | Used by | Meaning |
| --- | --- | --- |
| `KAGGLE_USERNAME` | `download` | Kaggle username |
| `KAGGLE_KEY` | `download` | Kaggle API key, held as a secret value |
| `MLFLOW_TRACKING_URI` | Tracking | MLflow store URI. When unset, the store is `artifacts/mlflow.db`. |

Copy the names from `.env.example` into `.env`. The `.env` file is listed in `.gitignore`.
