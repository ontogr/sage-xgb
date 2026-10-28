"""Forward-chained cross-fitting of the GNN and the `gnn` pipeline stage.

The training days are cut into day blocks. Fit k trains on blocks 1..k and
scores block k+1, so every score XGBoost trains on comes from a GNN that never
saw that row's day or label. The last fit trains on every block and scores the
validation, test and Tail rows.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl
import torch

from sage_xgb import artifacts, split
from sage_xgb.config import Config, SplitConfig
from sage_xgb.data import prepare
from sage_xgb.gnn.graph import EdgeTable, build_edge_table
from sage_xgb.gnn.model import GnnModel
from sage_xgb.gnn.train import TrainRows, fit_model, score_day


logger = logging.getLogger(__name__)

GNN_SCORE_COLUMN = "gnn_score"
_SCORES_FILE = "gnn_scores.parquet"
_TABLE_COLUMNS = (
    "row_id",
    "timestamp",
    "day",
    "split",
    "src_node",
    "dst_node",
    "src_bank",
    "dst_bank",
    "amount_paid",
    "amount_received",
    "paid_currency",
    "received_currency",
    "payment_format",
    "amount_usd",
)


def gnn_embedding_columns(embedding_dim: int) -> list[str]:
    """Return the names of the GNN embedding columns."""
    return [f"gnn_emb_{index}" for index in range(embedding_dim)]


@dataclass(frozen=True)
class Fit:
    """One GNN fit: the days it trains on and the rows it scores."""

    fit_id: int
    train_days: tuple[int, ...]
    score_days: tuple[int, ...]
    score_splits: tuple[str, ...]


def fold_plan(split_config: SplitConfig, also_score: Sequence[str] = ()) -> list[Fit]:
    """Return the forward-chained fits, 1-based by `fit_id`.

    Args:
        split_config: Split with the cross-fit day blocks.
        also_score: Extra splits that every fit scores, for example `("val",)`.
    """
    blocks = split.crossfit_blocks(split_config)
    fits = []
    for index in range(1, len(blocks)):
        train_days = tuple(day for block in blocks[:index] for day in block)
        fits.append(Fit(index, train_days, tuple(blocks[index]), tuple(also_score)))
    final_splits = ("val", "test", "tail")
    extra = tuple(name for name in also_score if name not in final_splits)
    fits.append(
        Fit(
            len(blocks),
            tuple(split.training_days(split_config)),
            (),
            final_splits + extra,
        )
    )
    return fits


def assert_out_of_fold(
    scores: pl.DataFrame, days: pl.DataFrame, plan: Sequence[Fit]
) -> None:
    """Raise if any row was scored by a fit that trained on its day or later.

    Args:
        scores: Frame with `row_id` and `fit_id`.
        days: Frame with `row_id` and `day` for every scored row.
        plan: The fits that produced the scores.

    Raises:
        ValueError: A row's day is not after every training day of its fit.
    """
    last_train_day = {fit.fit_id: max(fit.train_days) for fit in plan}
    joined = scores.select("row_id", "fit_id").join(days, on="row_id", how="left")
    limit = pl.col("fit_id").replace_strict(last_train_day, return_dtype=pl.Int16)
    leaked = joined.filter(pl.col("day") <= limit)
    if leaked.height:
        msg = f"{leaked.height} rows were scored by a fit that trained on their day"
        raise ValueError(msg)


def _train_rows(train: pl.DataFrame, table: EdgeTable, fit: Fit) -> TrainRows:
    rows = train.filter(pl.col("day").is_in(list(fit.train_days)))
    return TrainRows(
        position=table.positions(rows["row_id"].to_numpy()),
        day=rows["day"].to_numpy(),
        label=rows["is_laundering"].to_numpy().astype(np.float32),
    )


def _score_fit(
    model: GnnModel,
    table: EdgeTable,
    meta: pl.DataFrame,
    fit: Fit,
    config: Config,
) -> pl.DataFrame:
    wanted = meta.filter(
        pl.col("day").is_in(list(fit.score_days))
        | pl.col("split").is_in(list(fit.score_splits))
    )
    columns = gnn_embedding_columns(config.gnn.embedding_dim)
    frames = []
    for day in sorted(wanted["day"].unique().to_list()):
        row_ids = wanted.filter(pl.col("day") == day)["row_id"].to_numpy()
        logits, embeddings = score_day(
            model,
            table,
            table.positions(row_ids),
            int(day),
            config.gnn,
            seed=config.seed * 1000 + 500 + fit.fit_id,
        )
        frame = pl.DataFrame(embeddings, schema=columns)
        frames.append(
            frame.with_columns(
                pl.Series("row_id", row_ids).cast(pl.UInt32),
                pl.lit(fit.fit_id).cast(pl.Int8).alias("fit_id"),
                pl.Series(GNN_SCORE_COLUMN, logits),
            )
        )
    if not frames:
        return pl.DataFrame(
            schema={
                "row_id": pl.UInt32,
                "fit_id": pl.Int8,
                GNN_SCORE_COLUMN: pl.Float32,
            }
        )
    return pl.concat(frames)


def _run_fits(
    config: Config, fits: Sequence[int] | None, also_score: Sequence[str]
) -> tuple[pl.DataFrame, dict[int, GnnModel]]:
    torch.set_num_threads(config.gnn.num_threads)
    frame = prepare.scan_transactions(config)
    unlabelled = split.unlabelled_rows(frame).select(_TABLE_COLUMNS).collect()
    train = split.labelled_rows(
        frame.select("row_id", "day", "split", "is_laundering"), ("train",)
    ).collect()
    table = build_edge_table(unlabelled)
    meta = unlabelled.select("row_id", "day", "split")
    plan = [
        fit
        for fit in fold_plan(config.split, also_score)
        if fits is None or fit.fit_id in fits
    ]
    parts, models = [], {}
    for fit in plan:
        started = time.perf_counter()
        model = fit_model(
            table,
            _train_rows(train, table, fit),
            config.gnn,
            seed=config.seed * 1000 + fit.fit_id,
        )
        parts.append(_score_fit(model, table, meta, fit, config))
        models[fit.fit_id] = model
        logger.info(
            "gnn fit %d done in %.1f s", fit.fit_id, time.perf_counter() - started
        )
    scores = pl.concat(parts).sort("row_id")
    if scores["row_id"].n_unique() != scores.height:
        msg = "a row was scored by more than one fit; pass also_score for one fit"
        raise ValueError(msg)
    assert_out_of_fold(scores, meta.select("row_id", "day"), plan)
    return scores, models


def score_fits(
    config: Config, *, fits: Sequence[int] | None = None, also_score: Sequence[str] = ()
) -> pl.DataFrame:
    """Fit the GNNs and return out-of-fold scores without writing files.

    Args:
        config: Run config.
        fits: Fit ids to run, or None for all fits.
        also_score: Extra splits that the chosen fits score, for example
            `("val",)`.

    Returns:
        Frame with `row_id`, `fit_id`, `gnn_score` and `gnn_emb_*`, sorted by
        `row_id`.
    """
    scores, _ = _run_fits(config, fits, also_score)
    return scores


def read_scores(config: Config) -> pl.DataFrame:
    """Read `gnn_scores.parquet` from the `gnn` stage directory."""
    return pl.read_parquet(artifacts.stage_path(config, "gnn", _SCORES_FILE))


def run(config: Config) -> dict[str, float]:
    """Run the `gnn` stage: fit every cross-fit GNN and write scores and weights."""
    scores, models = _run_fits(config, None, ())
    directory = artifacts.ensure_stage_dir(config, "gnn")
    scores.write_parquet(directory / _SCORES_FILE, compression="zstd")
    outputs = [_SCORES_FILE]
    for fit_id, model in models.items():
        name = f"gnn_fit_{fit_id}.pt"
        torch.save(model.state_dict(), directory / name)
        outputs.append(name)
    artifacts.write_manifest(config, "gnn", outputs)
    return {"fits": float(len(models)), "scored_rows": float(scores.height)}
