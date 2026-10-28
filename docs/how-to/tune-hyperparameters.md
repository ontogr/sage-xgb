# Tune hyperparameters

The tune stage searches the ranges in `tune.space` and keeps the values that give the best validation PR-AUC. It writes the best settings to a complete config file that `run` can use. The stage never reads test or Tail labels.

## Prerequisites

- The `prepare` and `features` stages can run on the Variant. The tune stage runs them first when they are missing or out of date.
- `xgb.use_gnn` is true and is not a key of `tune.space`. Each trial scores a GNN, so the tune stage raises a config error otherwise.

## Set the search limits

Set the trial budget and the time limit in the config. The defaults are 20 trials and 10 minutes.

```yaml
tune:
  n_trials: 20
  timeout_minutes: 10.0
```

`n_trials` is the number of trials that the study aims to reach. Completed and pruned trials count toward it. `timeout_minutes` limits each call and starts when the call starts, so it includes the `prepare` and `features` stages. A fit or training step that has started is not interrupted, so a call can end after the limit by the length of that step. The limit does not apply to the whole study.

## Choose the searched keys

`tune.space` maps a dotted config path to a range. Each entry has a `kind` and the bounds for that kind.

| Kind | Fields | Values drawn |
| --- | --- | --- |
| `int` | `low`, `high` | Whole numbers from `low` to `high` |
| `float` | `low`, `high` | Uniform numbers from `low` to `high` |
| `log_float` | `low`, `high` | Numbers from `low` to `high` on a log scale |
| `categorical` | `choices` | One of the listed values |

This config replaces the default space with three keys.

```yaml
tune:
  space:
    gnn.neg_sample_rate:
      kind: log_float
      low: 0.01
      high: 0.2
    xgb.max_depth:
      kind: int
      low: 3
      high: 9
    xgb.learning_rate:
      kind: log_float
      low: 0.03
      high: 0.3
```

A `tune.space` value in a config file replaces the whole default space. Each entry is checked when the config loads: `low` must not exceed `high`, `log_float` needs `low` above 0, `int` needs whole-number bounds and `categorical` needs at least one choice. Each key must name a field that exists in the config. That check runs when the tune stage starts a trial, so a wrong key fails the stage at that point.

Use the config file for the space. A `--set` override cannot address one entry, because the space keys contain dots.

## Run the search

Run the tune stage with the Variant config.

```bash
make tune
```

The command runs `sage-xgb tune --config configs/hi_small.yaml`. Each trial fits a GNN on the earlier cross-fit blocks, scores the last block and the validation rows, and trains XGBoost on the last block with early stopping on validation. The objective is the validation PR-AUC of that XGBoost model.

## Resume a study

The study lives in `artifacts/hi-small/tune/study.db`, a SQLite file. Run `make tune` again to add trials. Each call runs `tune.n_trials` minus the number of completed and pruned trials, so a study that already has `tune.n_trials` of them gets no new trials. Failed trials and trials left running by a stopped process are replaced. Calls for the same variant and artifacts directory wait for each other. The sampler seed moves forward by the number of recorded trials on each resume, so resumed trials do not repeat the parameters of the earlier trials.

Pruned trials do not count toward the completed total, but they count toward `tune.n_trials`. A study with many pruned trials therefore runs fewer new trials on the next call.

The study stores a fingerprint of the data, split, seed, feature, GNN and XGBoost settings and the search space. A call with other settings fails with a config error. Use a new artifacts directory, or delete `study.db`, to start a new study. Changing only `tune.n_trials` or `tune.timeout_minutes` keeps the study.

## Use the best settings

The stage writes `artifacts/hi-small/tune/best.yaml`. The file holds the full config with the best trial's values. It has no `extends` key, so it loads on its own.

```bash
uv run sage-xgb run --config artifacts/hi-small/tune/best.yaml
```

The file keeps the Variant name and the artifacts directory. The run therefore writes to `artifacts/hi-small/` and reuses any stage whose config sections and upstream manifests did not change. Stages whose settings changed are rerun in place.

## Check the result

The stage returns `best_val_pr_auc`, `n_trials` (every trial recorded in the study) and `n_completed`. The log prints the parameters of the best trial. MLflow logs the same values on the `tune` run, and the [MLflow how-to](read-runs-in-mlflow.md) covers how to find it. To list every trial, load the study with Optuna, for example with `optuna.load_study(study_name="hi-small", storage="sqlite:///artifacts/hi-small/tune/study.db")`.

## Failure cases

- `tune scores the GNN proxy, so xgb.use_gnn must be true`: set `xgb.use_gnn: true` in the config. Remove `xgb.use_gnn` from `tune.space` when it appears there.
- `the study ... was created with different data, split, seed, model or search space settings`: use a new artifacts directory or delete the study file.
- `no tune trial completed`: no trial finished, either because every trial was pruned or because the time limit ended first. A trial is pruned when its sampled values break a config rule, such as a `gnn.layers` entry in `tune.space` that gives a count the length of `gnn.neighbors` does not match. The log has a warning for each pruned trial. Fix the ranges in `tune.space`, then run the stage again.
- `tune.space key ... is not a config field`: the key is misspelled or names a section that does not exist.
