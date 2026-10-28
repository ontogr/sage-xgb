# Train on the sample

Running every pipeline stage on the committed sample writes the stage outputs under `artifacts/hi-small-sample/`. It needs no Kaggle account.

## Prerequisites

- Linux on x86_64.
- uv 0.12.23. Check the version with `uv --version`.
- A clone of the repository.

## Install the environment

1. Clone the repository and change into it.

   ```bash
   git clone https://github.com/ontogr/sage-xgb.git
   cd sage-xgb
   ```

2. Install the locked dependencies.

   ```bash
   make sync
   ```

   The command reads `uv.lock`, so every run installs the same versions. Run the rest of the tutorial from the repository root. The sample config uses paths relative to that folder.

## Run the pipeline

Run the full pipeline with the sample config.

```bash
make train-sample
```

The command runs `sage-xgb run --config configs/sample.yaml`. It runs the stages `prepare`, `features`, `gnn` and `train-xgb` in that order. Each stage writes a `manifest.json` next to its outputs.

## Check the outputs

The run writes these files.

```text
artifacts/
  mlflow.db
  mlartifacts/
  hi-small-sample/
    prepare/     transactions.parquet, accounts.parquet, manifest.json
    features/    features.parquet, manifest.json
    gnn/         gnn_scores.parquet, gnn_fit_1.pt, gnn_fit_2.pt,
                 gnn_fit_3.pt, manifest.json
    train-xgb/   booster.json, feature_columns.json, metadata.json,
                 metrics_val.json, manifest.json
```

The [artifact reference](../reference/artifacts.md) describes the columns of each file.

## Run it again

Run `make train-sample` a second time. Each stage checks its manifest against the config and the upstream manifests. The run logs a skip line for every stage, and it does not rewrite any stage output.

## Score the test and Tail slices

The `run` command does not score the test and Tail slices. Run the report stage with the sample config.

```bash
uv run sage-xgb report --config configs/sample.yaml
```

The stage writes `artifacts/hi-small-sample/report/report.json`. It also appends one line for each slice to `artifacts/hi-small-sample/report/history.jsonl`. Each line records one read of the labels of a slice. Only the report stage reads those labels, as [Test labels are read once](../explanation/split-and-tail.md#test-labels-are-read-once) explains.

## Next steps

- [Train on HI-Small](train-on-hi-small.md) runs the same stages on the full Variant.
- [Read runs in MLflow](../how-to/read-runs-in-mlflow.md) shows the runs that this tutorial logged.
- [Training pipeline](../explanation/training-pipeline.md) explains what each stage reads and writes.

## Troubleshooting

If the run stops with a file-not-found error for `tests/fixtures/amlworld_hi_small_sample`, run the command from the repository root. The `data.raw_dir` value in the config is relative to the current folder.
