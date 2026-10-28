"""End-to-end tests for the train-xgb stage on the committed sample.

Most tests read the bundle that the session `pipeline_config` fixture wrote.
The no-GNN test runs prepare and features into its own artifacts directory,
then trains a bundle from them.
"""

import json

import polars as pl
import pytest

from sage_xgb import artifacts, pipeline, train_xgb
from sage_xgb.config import Config
from sage_xgb.data import prepare
from sage_xgb.features import table
from sage_xgb.gnn import crossfit


def _read_json(config: Config, name: str) -> dict | list:
    path = artifacts.stage_path(config, "train-xgb", name)
    return json.loads(path.read_text(encoding="utf-8"))


def _validation_rows(config: Config) -> pl.DataFrame:
    transactions = prepare.read_transactions(config)
    return transactions.filter(pl.col("split") == "val").select("row_id")


class TestTrainXgb:
    """Bundle contents and training rows of the train-xgb stage."""

    def test_bundle_predicts_one_probability_per_validation_row(
        self, pipeline_config: Config
    ) -> None:
        """Return one probability in [0, 1] for each validation row."""
        bundle = train_xgb.load_bundle(
            artifacts.stage_dir(pipeline_config, "train-xgb")
        )
        frame = train_xgb.join_inputs(
            _validation_rows(pipeline_config),
            table.read_features(pipeline_config),
            crossfit.read_scores(pipeline_config),
        )

        probabilities = bundle.predict(frame)

        assert probabilities.shape == (frame.height,)
        assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()

    def test_trains_only_on_rows_of_days_three_to_six(
        self, pipeline_config: Config
    ) -> None:
        """Count the training rows of days 3 to 6 and nothing from days 1 and 2."""
        metadata = _read_json(pipeline_config, "metadata.json")
        train_rows = prepare.read_transactions(pipeline_config).filter(
            (pl.col("split") == "train") & pl.col("day").is_in([3, 4, 5, 6])
        )

        assert metadata["training_days"] == [3, 4, 5, 6]
        assert metadata["n_train_rows"] == train_rows.height

    def test_feature_columns_are_table_columns_then_gnn_columns(
        self, pipeline_config: Config
    ) -> None:
        """List the table features first, then the GNN score and embeddings."""
        table_columns = [
            column
            for column in table.read_features(pipeline_config).columns
            if column != "row_id"
        ]
        gnn_columns = [
            crossfit.GNN_SCORE_COLUMN,
            *crossfit.gnn_embedding_columns(pipeline_config.gnn.embedding_dim),
        ]

        assert _read_json(pipeline_config, "feature_columns.json") == [
            *table_columns,
            *gnn_columns,
        ]

    def test_predict_raises_key_error_when_a_feature_column_is_missing(
        self, pipeline_config: Config
    ) -> None:
        """Raise instead of filling in a feature column that is absent."""
        bundle = train_xgb.load_bundle(
            artifacts.stage_dir(pipeline_config, "train-xgb")
        )
        features = table.read_features(pipeline_config).drop("hour")
        frame = train_xgb.join_inputs(
            _validation_rows(pipeline_config),
            features,
            crossfit.read_scores(pipeline_config),
        )

        with pytest.raises(KeyError, match="hour"):
            bundle.predict(frame)

    def test_stores_validation_threshold_in_metadata_and_metrics(
        self, pipeline_config: Config
    ) -> None:
        """Record one validation threshold in the bundle and its metadata."""
        bundle = train_xgb.load_bundle(
            artifacts.stage_dir(pipeline_config, "train-xgb")
        )
        metadata = _read_json(pipeline_config, "metadata.json")
        metrics_val = _read_json(pipeline_config, "metrics_val.json")

        assert metadata["threshold"] == metrics_val["threshold"]
        assert bundle.threshold == metrics_val["threshold"]

    def test_bundle_without_gnn_has_no_gnn_columns_and_loads(
        self, config_factory, tmp_path
    ) -> None:
        """Train without GNN scores and load a bundle with no GNN columns."""
        config = config_factory(tmp_path / "artifacts", "xgb.use_gnn=false")
        pipeline.run(config, through="features")

        train_xgb.run(config)

        feature_columns = _read_json(config, "feature_columns.json")
        assert not any(name.startswith("gnn_") for name in feature_columns)
        bundle = train_xgb.load_bundle(artifacts.stage_dir(config, "train-xgb"))
        assert bundle.feature_columns == feature_columns
