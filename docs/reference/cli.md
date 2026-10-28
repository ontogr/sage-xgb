# CLI reference

The `sage-xgb` command has one subcommand for each pipeline stage. The `run` subcommand runs the four training stages in order. The `tune` and `report` subcommands run outside `run`.

```text
sage-xgb [-h] [--version] {download,prepare,features,gnn,train-xgb,tune,report,run} ...
```

## Options

Every subcommand takes these options.

| Option | Required | Meaning |
| --- | --- | --- |
| `--config PATH` | Yes | YAML config file to load |
| `--set KEY=VALUE` | No, repeatable | Override one config value after the file and its `extends` parent load |

The value in `--set` parses as YAML. `5`, `0.1`, `true` and `[5, 3]` keep their types. A key with dots, such as `gnn.epochs`, names a nested config field.

The top-level option `sage-xgb --version` prints the installed package version and exits.

## Subcommands

| Subcommand | What it runs | Reads | Writes |
| --- | --- | --- | --- |
| `download` | Copies the raw files for the Variant from Kaggle | Kaggle | Files in `data.raw_dir` |
| `prepare` | Sorts the Transactions and writes the split table | The raw Transactions CSV | `prepare/` outputs |
| `features` | Builds the trailing-window feature table | `prepare/` outputs | `features/` outputs |
| `gnn` | Fits the cross-fit GNNs and writes the scores | `prepare/` outputs | `gnn/` outputs |
| `train-xgb` | Trains XGBoost and writes the model bundle | `prepare/`, `features/` and `gnn/` outputs | `train-xgb/` outputs |
| `run` | Runs `prepare`, `features`, `gnn` and `train-xgb` in order, with skipping | The raw Transactions CSV | Every training stage output |
| `tune` | Searches `tune.space` on validation PR-AUC | `prepare/` and `features/` outputs, after they run | `tune/study.db` and `tune/best.yaml` |
| `report` | Scores the test and Tail slices once | The `train-xgb/` bundle, `prepare/` and `features/` outputs, and `gnn/` outputs when the bundle uses them | `report/report.json` and `report/history.jsonl` |

The [artifact reference](artifacts.md) lists the files of each directory.

## Options for `run`

| Option | Meaning |
| --- | --- |
| `--force` | Reruns every stage, even when its manifest matches |
| `--from-stage {prepare,features,gnn,train-xgb}` | Reruns the named stage and every stage after it |

A stage that is not named by these options is skipped when its manifest matches the config and the upstream manifests, and its outputs exist. [Skip a stage that is current](../explanation/training-pipeline.md#skip-a-stage-that-is-current) lists the conditions.

## Single-stage subcommands

The subcommands `download`, `prepare`, `features`, `gnn`, `train-xgb`, `tune` and `report` run their stage directly. They do not check the manifests of earlier stages, and they do not skip a stage whose outputs are current. Run `prepare` before `features`, and run `features` and `gnn` before `train-xgb`. Otherwise the stage fails when it opens a missing file.

The `tune` subcommand is the exception that runs the earlier stages first. It calls the `run` logic up to `features`, so those stages skip when they are current.

## Exit status

| Status | When |
| --- | --- |
| `0` | The command finished |
| `2` | The config failed to load or a stage rejected it, or argparse rejected the command line. A config error writes `config error:` to standard error |
| `1` | Any other error. The traceback is written to standard error |

A config error covers a missing file, a nested `extends`, a malformed `--set` value, an unknown key and a value that fails validation. The `tune` stage raises a config error for `xgb.use_gnn: false`, for `xgb.use_gnn` as a searched key, for a `tune.space` key that is not a config field and for a study created with other settings. Those cases also exit with status `2`.

## Examples

Run the pipeline with one value changed for this run.

```bash
uv run sage-xgb run --config configs/hi_small.yaml --set gnn.epochs=3
```

Rerun the GNN stage and the stages after it.

```bash
uv run sage-xgb run --config configs/hi_small.yaml --from-stage gnn
```

Score the test and Tail slices after the training stage has finished.

```bash
uv run sage-xgb report --config configs/hi_small.yaml
```

## Make targets

The Makefile wraps the common commands.

| Target | Runs |
| --- | --- |
| `make download` | `sage-xgb download --config configs/hi_small.yaml` |
| `make train-sample` | `sage-xgb run --config configs/sample.yaml` |
| `make train` | `sage-xgb run --config configs/hi_small.yaml` |
| `make tune` | `sage-xgb tune --config configs/hi_small.yaml` |
| `make report` | `sage-xgb report --config configs/hi_small.yaml` |
