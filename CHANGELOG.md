# Changelog

Changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html) once it reaches 1.0.0. Before 1.0.0, a minor release may change the public interface.

## [Unreleased]

## [0.0.1] - 2026-10-28

### Added

- Day-aligned split of each AMLworld Variant into train, validation and test splits, with the Tail reported as its own slice.
- Feature table of row features and trailing-window features, with each window covering the minutes before a Transaction and not its own minute.
- Supervised GraphSAGE GNN fitted over forward-chained cross-fit day blocks, with daily snapshot graphs and out-of-fold scores for XGBoost.
- XGBoost training on the feature table and the GNN scores, with early stopping on the validation split and a stored decision threshold.
- Report stage that scores the test and Tail slices once, and records each read of the test labels.
- Hyperparameter search with Optuna on validation PR-AUC, with a resumable study and a written best config.
- Command line tool `sage-xgb` with one subcommand for each stage, and a `run` command that skips stages whose outputs are current.
- Run config in YAML, one file for each Variant, validated when it loads, with command line overrides.
- MLflow tracking of pipeline runs and stage runs in a local SQLite store.
- Committed sample of HI-Small for end-to-end tests and sample runs.
- Dockerfile with a `train` target.
- Documentation of the tutorials, how-to guides, reference and the training design.
- `gnn.encoder` config key that selects the GraphSAGE layer type: `edge_sage` (default) or PyTorch Geometric's `SAGEConv` (`sage`).
