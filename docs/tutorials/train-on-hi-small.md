# Train on HI-Small

Downloading the HI-Small Variant from Kaggle and training the model on it writes the outputs under `artifacts/hi-small/`. The download needs a Kaggle account with an API key.

## Prerequisites

- Linux on x86_64.
- uv 0.12.23.
- A Kaggle account and its API credentials, which are a username and a key.
- Enough free memory and disk space for the full Variant. The HI-Small config sets `xgb.num_threads` and `gnn.num_threads` to 8. Lower them on a machine with fewer cores, and see [Scale the GNN](../how-to/scale-the-gnn.md) for the keys that change the model size.

## Install the environment

1. Clone the repository and change into it.

   ```bash
   git clone https://github.com/ontogr/sage-xgb.git
   cd sage-xgb
   ```

2. Install the locked dependencies. The command also installs the `kaggle` extra, which provides `kagglehub`.

   ```bash
   make sync
   ```

## Set the Kaggle credentials

Create a `.env` file in the repository root with the two Kaggle variables. Copy the names from `.env.example`.

```text
KAGGLE_USERNAME=your-kaggle-username
KAGGLE_KEY=your-kaggle-key
```

The `.env` file is listed in `.gitignore`. Do not commit it. If the download fails with an authentication error, check that both variables are set, or that kagglehub can read its own `kaggle.json` file.

## Download the raw files

Run the download.

```bash
make download
```

The command copies three files into `data/`. Each file name starts with the `data.name` value, which is `HI-Small` in the shipped config.

- `HI-Small_Trans.csv` holds the Transactions. The pipeline reads this file.
- `HI-Small_accounts.csv` holds the account metadata. The pipeline does not read it.
- `HI-Small_Patterns.txt` lists the labelled laundering patterns. The pipeline does not read it.

## Train the model

Run the full pipeline.

```bash
make train
```

The command runs `sage-xgb run --config configs/hi_small.yaml`. It writes the stage outputs under `artifacts/hi-small/`. If a stage finished in an earlier run and its inputs have not changed, the run skips it.

## Score the test and Tail slices

Run the report stage. It scores the test and Tail slices once, with the threshold from the training stage.

```bash
make report
```

The stage writes `artifacts/hi-small/report/report.json` and appends one line for each slice to `artifacts/hi-small/report/history.jsonl`.

## Check the outputs

The stage directories hold the same files as the sample run. Check them with `ls artifacts/hi-small`. The [artifact reference](../reference/artifacts.md) lists the columns and keys of each file.

## Read the runs

Start the MLflow UI on the store that the run wrote.

```bash
make mlflow-ui
```

Open the address that MLflow prints. The [MLflow how-to](../how-to/read-runs-in-mlflow.md) explains the run names and the logged values.

## Clean up

The `make clean` target deletes the whole `artifacts/` directory. That includes the MLflow store, the trained models and the tune study. Run it only when you no longer need those files. The `data/` directory is not touched.

## Next steps

- [Tune hyperparameters](../how-to/tune-hyperparameters.md) searches the configured ranges on validation data.
- [Add a Variant](../how-to/add-a-variant.md) explains what another AMLworld Variant needs.
- [Split and Tail](../explanation/split-and-tail.md) explains the day boundaries in the config.

## Troubleshooting

If the run fails while it reads `data/HI-Small_Trans.csv`, check that the download finished and that the file is in `data/`. The config sets `data.raw_dir` to `data`, relative to the repository root.
