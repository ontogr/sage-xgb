"""End-to-end checks of the cross-fit GNN stage on the committed sample."""

import shutil

import numpy as np
import polars as pl
import pytest

from sage_xgb import artifacts, split
from sage_xgb.data import prepare
from sage_xgb.gnn import crossfit
from sage_xgb.gnn.graph import build_edge_table, snapshot_bounds


def _copy_prepared(source_config, make_config, tmp_path, name, *overrides):
    config = make_config(tmp_path / name, *overrides)
    shutil.copytree(
        artifacts.stage_dir(source_config, "prepare"),
        artifacts.stage_dir(config, "prepare"),
    )
    return config


def _flip_labels(config, from_day):
    path = artifacts.stage_path(config, "prepare", "transactions.parquet")
    frame = pl.read_parquet(path)
    flipped = frame.with_columns(
        pl.when(pl.col("day") >= from_day)
        .then(1 - pl.col("is_laundering"))
        .otherwise(pl.col("is_laundering"))
        .cast(pl.Int8)
        .alias("is_laundering")
    )
    flipped.write_parquet(path)


class TestGnn:
    """Cross-fit GNN stage on the sample."""

    def test_scores_every_row_after_the_first_block_once(self, gnn_config):
        """Every row after block 1 has exactly one score."""
        scores = crossfit.read_scores(gnn_config)
        rows = prepare.read_transactions(gnn_config)
        first_block = split.crossfit_blocks(gnn_config.split)[0]
        expected = rows.filter(~pl.col("day").is_in(first_block))
        assert scores.height == expected.height
        assert scores["row_id"].n_unique() == scores.height
        assert set(scores["row_id"]) == set(expected["row_id"])
        assert scores.null_count().sum_horizontal().item() == 0

    def test_schema_has_score_and_embedding_columns(self, gnn_config):
        """The score file holds the documented columns only."""
        scores = crossfit.read_scores(gnn_config)
        columns = crossfit.gnn_embedding_columns(gnn_config.gnn.embedding_dim)
        assert set(scores.columns) == {"row_id", "fit_id", "gnn_score", *columns}

    def test_out_of_fold_guard_rejects_a_leaking_score(self, gnn_config):
        """The guard raises when a fit scores a day it trained on."""
        rows = prepare.read_transactions(gnn_config)
        plan = crossfit.fold_plan(gnn_config.split)
        day_one = rows.filter(pl.col("day") == 1).head(3)
        leaked = pl.DataFrame(
            {"row_id": day_one["row_id"], "fit_id": [3] * day_one.height},
            schema={"row_id": pl.UInt32, "fit_id": pl.Int8},
        )
        with pytest.raises(ValueError, match="trained on their day"):
            crossfit.assert_out_of_fold(leaked, rows.select("row_id", "day"), plan)

    @pytest.mark.parametrize("lag", [0, 1])
    def test_snapshots_hold_no_edge_from_the_scored_day_or_later(
        self, prepared_config, lag
    ):
        """No snapshot edge comes from the scored day or later."""
        config = prepared_config.model_copy(
            update={
                "gnn": prepared_config.gnn.model_copy(update={"snapshot_lag_days": lag})
            }
        )
        rows = split.unlabelled_rows(prepare.read_transactions(config))
        table = build_edge_table(rows)
        for day in range(1, 11):
            _, hi = snapshot_bounds(table, day, config.gnn)
            assert hi == 0 or table.day[:hi].max() < day - lag

    def test_future_labels_and_edges_do_not_change_earlier_scores(
        self, prepared_config, config_factory, tmp_path
    ):
        """Fit 1 scores of days 3 and 4 ignore labels of day 3 and later."""
        base = _copy_prepared(prepared_config, config_factory, tmp_path, "base")
        flipped = _copy_prepared(prepared_config, config_factory, tmp_path, "flipped")
        _flip_labels(flipped, from_day=3)
        first = crossfit.score_fits(base, fits=[1])
        second = crossfit.score_fits(flipped, fits=[1])
        embeddings = crossfit.gnn_embedding_columns(base.gnn.embedding_dim)
        assert first.height == second.height
        for column in ("gnn_score", *embeddings):
            diff = np.abs(first[column].to_numpy() - second[column].to_numpy())
            assert diff.max() < 1e-5

    def test_plain_sage_encoder_scores_every_row_of_the_next_block(
        self, prepared_config, config_factory, tmp_path
    ):
        """The PyG SAGEConv encoder scores block 2 with no missing values."""
        config = _copy_prepared(
            prepared_config, config_factory, tmp_path, "sage", "gnn.encoder=sage"
        )
        scores = crossfit.score_fits(config, fits=[1])
        rows = prepare.read_transactions(config)
        block = split.crossfit_blocks(config.split)[1]
        assert scores.height == rows.filter(pl.col("day").is_in(block)).height
        assert scores.null_count().sum_horizontal().item() == 0
