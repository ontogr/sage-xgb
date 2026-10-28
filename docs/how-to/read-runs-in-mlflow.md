# Read runs in MLflow

Each pipeline run logs its settings, metrics and stage manifests to MLflow. Opening the MLflow UI on the store that the run wrote shows every logged run, and the run names follow the stage names.

## Prerequisites

- A completed `run`, `tune` or `report` command with tracking enabled. The `tracking.enabled` key defaults to true.
- A clone with `make sync` done.

## Open the UI

Run the UI from the repository root.

```bash
make mlflow-ui
```

The command runs `mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db`. Open the address that MLflow prints. By default, the UI serves on port 5000.

The `make mlflow-ui` target always opens `artifacts/mlflow.db`. The tracking store can be set in `.env` with `MLFLOW_TRACKING_URI`. When that variable is set, start the UI yourself with the same URI.

## Find the runs

Every config writes to the experiment named by `tracking.experiment`, which is `sage-xgb` by default. The HI-Small and sample runs share that experiment. Filter by the `data.raw_dir` parameter to tell them apart, because the sample config changes that value.

The pipeline logs runs in a tree.

- One parent run named `run` covers a `sage-xgb run` command.
- Each stage inside it logs a nested run named after the stage: `prepare`, `features`, `gnn` or `train-xgb`.
- The `tune` and `report` commands log a run with their own names. The `tune` command also starts a `run` for its shared inputs, nested under its stage run.

## Read what each run logs

Each stage run logs these values.

- Parameters. The `data`, `split`, `features`, `xgb`, `gnn` and `seed` sections of the config, flattened to dotted keys such as `gnn.hidden`. Each value is cut to 250 characters.
- Metrics. `duration_seconds` for the stage, and the scalar values that the stage returns. Metric names keep letters, digits, underscores, hyphens, dots, spaces and slashes. Other characters become underscores.
- Artifacts. The stage's `manifest.json`, under a folder with the stage name.

The `train-xgb` run logs the validation metrics that the stage returns. The `report` run logs the scalar keys of each slice, with names such as `test_pr_auc` and `tail_positives`. A slice with no positives has no metric values, so only its row and positive counts are logged.

## Turn tracking off

Set `tracking.enabled: false` in the config, or pass `--set tracking.enabled=false` on the command line. With tracking off, the stages write their files and make no MLflow calls.

## Telemetry

The tracking module sets `MLFLOW_DISABLE_TELEMETRY` to `true` when the variable is unset, so MLflow sends no anonymous usage data. To allow telemetry, set `MLFLOW_DISABLE_TELEMETRY=false` in the environment before you run a command. MLflow still turns telemetry off when `DO_NOT_TRACK` is `true` or when it runs in a CI environment.
