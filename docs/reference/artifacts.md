# Artifact reference

Every stage writes its outputs under `<paths.artifacts_dir>/<variant>/<stage>/`. With the shipped configs, the path is `artifacts/hi-small/<stage>/` for HI-Small and `artifacts/hi-small-sample/<stage>/` for the sample. The MLflow store sits at the artifacts root.

## Layout

```text
artifacts/
  mlflow.db                  MLflow SQLite store, when MLFLOW_TRACKING_URI is unset
  mlartifacts/               MLflow artifact root for the experiment
  <variant>/
    prepare/                 transactions.parquet, accounts.parquet, manifest.json
    features/                features.parquet, manifest.json
    gnn/                     gnn_scores.parquet, gnn_fit_<n>.pt, manifest.json
    train-xgb/               booster.json, feature_columns.json, metadata.json,
                             metrics_val.json, manifest.json
    tune/                    study.db, best.yaml
    report/                  report.json, history.jsonl
```

The `tune` folder appears after a tune call, and the `report` folder appears after a report call.

## Manifests

Each of the four pipeline stages writes `manifest.json` after its outputs. The `tune` and `report` stages write no manifest. The manifest has no timestamps, so a rerun with the same inputs writes the same bytes.

| Key | Type | Content |
| --- | --- | --- |
| `stage` | string | Stage name, such as `features` |
| `version` | string | Installed package version |
| `config_hash` | string | SHA-256 of the config sections that the stage reads |
| `upstream` | object | Each upstream stage name mapped to the SHA-256 of its manifest file, or to `null` when that manifest is missing |
| `outputs` | list of strings | Output file names of the stage, sorted |

The `prepare` stage has an empty `upstream` object. The `train-xgb` stage lists `gnn` in `upstream` only when `xgb.use_gnn` is true.

## prepare

`transactions.parquet` holds one row per raw Transaction, sorted by `timestamp` and then `row_id`.

| Column | Type | Content |
| --- | --- | --- |
| `row_id` | UInt32 | Zero-based index of the data row in the raw CSV |
| `timestamp` | Datetime, microseconds | Minute of the Transaction |
| `day` | Int16 | 1-based day number from `split.window_start` |
| `split` | String | `train`, `val`, `test` or `tail` |
| `src` | String | Sender Account key, `<bank>:<account>` |
| `dst` | String | Receiver Account key, `<bank>:<account>` |
| `src_node` | Int32 | Node id of the sender Account |
| `dst_node` | Int32 | Node id of the receiver Account |
| `src_bank` | String | Sender bank code, with leading zeros kept |
| `dst_bank` | String | Receiver bank code, with leading zeros kept |
| `amount_paid` | Float64 | Amount in the paid currency |
| `amount_received` | Float64 | Amount in the received currency |
| `paid_currency` | String | Paid currency name |
| `received_currency` | String | Received currency name |
| `payment_format` | String | Payment format name |
| `amount_usd` | Float64 | `amount_paid` times the fixed USD rate of the paid currency |
| `is_laundering` | Int8 | Label of the Transaction. Stages read it only through the labelled-row loader in `sage_xgb/split.py`. |

`accounts.parquet` holds one row per Account, in order of first appearance.

| Column | Type | Content |
| --- | --- | --- |
| `account` | String | Account key, `<bank>:<account>` |
| `node_id` | Int32 | Node id used by `transactions.parquet` |
| `bank` | String | Bank code |
| `number` | String | Account number |

## features

`features.parquet` holds one row per Transaction, sorted by `row_id`. It has no label and no split column. All feature columns are Float32, and nulls are kept.

| Group | Columns | Count |
| --- | --- | --- |
| Key | `row_id` (UInt32) | 1 |
| Row features | `hour`, `weekday`, `log_amount_usd`, `amount_ratio`, `same_bank`, `is_cross_currency`, `is_self_transfer`, `paid_currency_code`, `payment_format_code` | 9 |
| Window features | `<side>_<role>_<stat>_<window>` | 36 |
| Minutes since last | `<side>_<role>_minutes_since_last` | 4 |

The window features use these parts.

- `side` is `src` for the sender Account of the Transaction, or `dst` for the receiver.
- `role` is `out` for events where the Account sent money, or `in` for events where it received money.
- `stat` is `count`, `usd_sum` or `usd_max`.
- `window` is `1h`, `24h` or `7d`.

A Transaction's `src_out_count_1h` is the number of earlier Transactions that the sender sent in the window `[t - 1h, t)`. The [features page](../explanation/features.md) gives the rule.

The `features.exclude` config key removes names from this file.

## gnn

`gnn_scores.parquet` holds one row for each Transaction that a cross-fit fit scored. Each row appears once, with the fit that scored it. With the shipped blocks, the rows of the first block have no row here.

| Column | Type | Content |
| --- | --- | --- |
| `row_id` | UInt32 | Key of the Transaction in `features.parquet` |
| `fit_id` | Int8 | Fit that scored the row, numbered from 1 |
| `gnn_score` | Float32 | Raw logit of the edge head. It is not a probability. |
| `gnn_emb_0` to `gnn_emb_<n-1>` | Float32 | Edge head embedding, with `n` equal to `gnn.embedding_dim` |

`gnn_fit_<n>.pt` holds the PyTorch state dict of the GNN for fit `n`. Loading it needs the `gnn.encoder`, `gnn.hidden`, `gnn.layers` and `gnn.embedding_dim` values that built the model.

## train-xgb

`booster.json` is the trained XGBoost model, written by `Booster.save_model`. The `load_bundle` function reads it together with the two JSON files next to it.

`feature_columns.json` is a JSON list of the column names in model order. It holds the feature columns, then `gnn_score`, then the `gnn_emb_*` columns when `xgb.use_gnn` is true.

`metadata.json` holds these keys.

| Key | Type | Content |
| --- | --- | --- |
| `config_hash` | string | Hash of the `xgb`, `features`, `split` and `seed` sections |
| `training_days` | list of integers | Days whose training rows fit the model |
| `n_train_rows` | integer | Training rows |
| `n_val_rows` | integer | Validation rows |
| `n_train_positives` | integer | Illicit rows among the training rows |
| `use_gnn` | boolean | Whether the GNN columns are features |
| `vocabulary_hash` | string | Short hash of the fixed rates and payment formats |
| `threshold` | float | Validation F1 threshold. A score at or above it is positive. |
| `best_iteration` | integer | Zero-based boosting round that early stopping selected |
| `trained_on` | string | Text such as `days 3-6` |

`metrics_val.json` holds the validation metrics that the stage computed with the stored model. Its keys are `pr_auc`, `f1`, `precision_at_k`, `threshold`, `positives` and `rows`.

## tune

`study.db` is the Optuna study in SQLite. The study name is the variant name. The study also stores a fingerprint of the data, split, seed, feature, GNN and XGBoost settings and the search space. A call with other settings is refused.

`best.yaml` is the complete config with the best trial's values. It has no `extends` key. The file is written when a call completes at least one trial.

## report

`report.json` holds these keys.

| Key | Type | Content |
| --- | --- | --- |
| `bundle_digest` | string | SHA-256 of `booster.json` at the time of the call |
| `threshold` | float | Threshold from `metadata.json`, applied without change |
| `test` | object | Metrics of the test slice |
| `tail` | object | Metrics of the Tail slice |

Each slice object holds these keys.

| Key | Type | Content |
| --- | --- | --- |
| `rows` | integer | Rows in the slice |
| `positives` | integer | Illicit Transactions in the slice |
| `pr_auc` | float or null | PR-AUC, or null when the slice has no positives |
| `f1` | float or null | F1 at the stored threshold, or null when the slice has no positives |
| `precision_at_k` | float or null | Precision among the top `positives` rows, or null when the slice has no positives |
| `intervals` | object or null | A 95 percent Poisson bootstrap interval for each of `pr_auc`, `f1` and `precision_at_k`. Each interval is a two-item list of the 2.5 and 97.5 percentiles, or null when the slice has no positives. |

`history.jsonl` holds one JSON object for each slice that the report stage reads. The stage writes the line before it scores the slice. Each line has these keys.

| Key | Type | Content |
| --- | --- | --- |
| `bundle_digest` | string | Same digest as the `report.json` written by that call |
| `read_at` | string | UTC time of the read in ISO 8601 format |
| `slice` | string | `test` or `tail` |

The count of lines in this file is the number of times the report stage has read the labels of a slice.

## mlflow

`mlflow.db` is the SQLite store that holds the experiments and runs. `mlartifacts/` holds the files that the runs logged, such as manifests. The [MLflow how-to](../how-to/read-runs-in-mlflow.md) explains the runs.
