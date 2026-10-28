# Training pipeline

A pipeline run turns the raw Transactions file of a Variant into a trained XGBoost model. The run has four stages, and each stage reads the files of the stages before it. A stage writes its outputs and a manifest, and the next run uses the manifests to decide which stages are current.

## The stages

| Stage | Reads | Writes | Purpose |
| --- | --- | --- | --- |
| `prepare` | The raw Transactions CSV | `transactions.parquet`, `accounts.parquet` | Sorts the Transactions, assigns days and splits, and numbers the Accounts |
| `features` | `transactions.parquet` | `features.parquet` | Computes the row features and the trailing-window features |
| `gnn` | `transactions.parquet` | `gnn_scores.parquet`, one `gnn_fit_<n>.pt` per fit | Fits the cross-fit GNNs and writes out-of-fold scores |
| `train-xgb` | `transactions.parquet`, `features.parquet`, and `gnn_scores.parquet` when `xgb.use_gnn` is true | The model bundle | Trains XGBoost on the training days after the first cross-fit block and early-stops on validation |

The `report` stage runs after the training stages. It reads the bundle, the feature table, the labels from the prepared Transactions and, when the bundle uses them, the GNN scores. It first checks that every stage up to `train-xgb` is up to date for the config and raises `StaleArtifactError` otherwise. It scores the test and Tail slices once, writes `report.json` and appends a line to `history.jsonl` for each slice. Only this stage reads test and Tail labels, as [Split and Tail](split-and-tail.md) explains.

The `prepare` stage sorts the raw rows by timestamp, with the raw row id as the tiebreak. Every later stage reads the sorted table, so the stage order is also the order of the data.

Setting `xgb.use_gnn` to `false` does not remove the `gnn` stage from `run`. The `run` command still checks that stage and runs it when its manifest is stale, and `train-xgb` reads no GNN output.

## Skip a stage that is current

A `run` command checks each stage in order. The check runs the stage again unless the manifest on disk matches the current state. The loop in `sage_xgb/pipeline.py` reads as follows.

```python
for index, stage in enumerate(stages[: last_index + 1]):
    if force or index >= from_index or not artifacts.is_fresh(config, stage):
        started = time.perf_counter()
        execute_stage(config, stage)
        timings.append((stage, "ran", time.perf_counter() - started))
    else:
        logger.info("skipping %s, up to date", stage)
        timings.append((stage, "skipped", 0.0))
```

The `is_fresh` function decides whether a stage is current.

```python
def is_fresh(config: Config, stage: str) -> bool:
    """Return true if the stage manifest matches the current config and inputs."""
    manifest = read_manifest(config, stage)
    if manifest is None:
        return False
    expected = _expected(config, stage)
    if any(manifest.get(key) != value for key, value in expected.items()):
        return False
    directory = stage_dir(config, stage)
    return all((directory / name).exists() for name in manifest["outputs"])
```

A stage is current when all of the following hold.

- Its manifest exists.
- The config sections that the stage reads hash to the value in the manifest.
- The digest of each upstream manifest matches the digest in the manifest.
- The stage name and the installed package version match the manifest.
- Every output file named in the manifest exists.

The config hash covers only the sections that the stage reads. Changing `gnn.epochs` therefore reruns the `gnn` stage and the `train-xgb` stage, and leaves `prepare` and `features` alone. The chain of upstream digests carries a change forward. When the `prepare` manifest changes, every later stage is stale.

## Force a rerun

The `--force` option reruns every stage. The `--from-stage` option reruns the named stage and every stage after it. A failure in one stage leaves the manifests of the earlier stages in place, so the next run does not repeat their work.

The single-stage commands, such as `sage-xgb gnn`, run their stage without checking any manifest. The [CLI reference](../reference/cli.md) lists which subcommands check the manifests.

## Keep the outputs reproducible

Manifests hold no timestamps, and the Parquet files are written with a fixed compression. A rerun with the same inputs therefore writes the same manifest bytes. The end-to-end tests check that two runs of the `prepare` stage write identical Parquet files.

## Track the runs

When tracking is on, `run` opens one MLflow run named `run`. Each stage opens a nested run named after itself, and logs its settings, its duration, the values it returned and its manifest. The [MLflow how-to](../how-to/read-runs-in-mlflow.md) explains how to read them.

## Next steps

- [Split and Tail](split-and-tail.md) explains the day boundaries.
- [Leakage rules](leakage-rules.md) lists the rules that every stage follows.
- [GNN to XGBoost hand-off](gnn-to-xgboost-handoff.md) explains the scores that `train-xgb` reads.
