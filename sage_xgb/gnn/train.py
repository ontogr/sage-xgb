"""Fit one GNN and score Transactions with it.

A fit trains on labelled seed Transactions, one day at a time. The seeds of day
d are messaged through Snapshot(d), so no edge from day d or later reaches them.
Scoring uses the same snapshots and the same encoder, so training and scoring
share one definition of an Account embedding.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader

from sage_xgb.config import GnnConfig
from sage_xgb.gnn.graph import EdgeTable, build_snapshot
from sage_xgb.gnn.model import GnnModel


logger = logging.getLogger(__name__)

_HEAD_CHUNK = 200_000


@dataclass(frozen=True)
class TrainRows:
    """Labelled training Transactions as positions in the edge table."""

    position: np.ndarray
    day: np.ndarray
    label: np.ndarray


def _sample_day(
    rows: TrainRows, day: int, neg_sample_rate: float, rng: np.random.Generator
) -> np.ndarray:
    """Return indices into `rows` for one day: every positive, some negatives."""
    in_day = np.flatnonzero(rows.day == day)
    labels = rows.label[in_day]
    negatives = in_day[labels == 0]
    kept = negatives[rng.random(len(negatives)) < neg_sample_rate]
    return np.concatenate([in_day[labels == 1], kept])


def _train_day(
    model: GnnModel,
    optimizer: torch.optim.Optimizer,
    table: EdgeTable,
    *,
    rows: TrainRows,
    chosen: np.ndarray,
    day: int,
    config: GnnConfig,
) -> tuple[float, int]:
    """Run one pass over the seeds of one day. Returns summed loss and seed count."""
    snapshot = build_snapshot(table, day, config)
    if snapshot is None or len(chosen) == 0:
        return 0.0, 0
    positions = rows.position[chosen]
    seed_index = torch.stack([table.src[positions], table.dst[positions]])
    seed_attr = table.attr[positions]
    loader = LinkNeighborLoader(
        snapshot,
        num_neighbors=config.neighbors,
        edge_label_index=seed_index,
        edge_label=torch.from_numpy(rows.label[chosen]).float(),
        batch_size=config.batch_size,
        shuffle=True,
        neg_sampling=None,
        num_workers=config.num_workers,
    )
    total, count = 0.0, 0
    for batch in loader:
        h = model.encode_nodes(batch.x, batch.edge_index, batch.edge_attr)
        local = batch.edge_label_index
        logit, _ = model.score_edges(
            h[local[0]], h[local[1]], seed_attr[batch.input_id]
        )
        loss = functional.binary_cross_entropy_with_logits(logit, batch.edge_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(logit)
        count += len(logit)
    del loader, snapshot
    gc.collect()
    return total, count


def fit_model(
    table: EdgeTable, rows: TrainRows, config: GnnConfig, *, seed: int
) -> GnnModel:
    """Train a GNN on `rows` for a fixed number of epochs, with no early stopping.

    Every positive seed is used each epoch. Negatives are sampled at
    `config.neg_sample_rate`, so the logits are not calibrated probabilities.
    Early stopping on validation would use labels newer than the days the early
    cross-fit folds score, so the epoch count is fixed instead.

    Args:
        table: Edge table of all Transactions, without labels.
        rows: Labelled Transactions of the fit's training days only.
        config: GNN settings.
        seed: Seeds torch and the negative sampler.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = GnnModel(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    model.train()
    days = np.unique(rows.day)
    for epoch in range(config.epochs):
        total, count = 0.0, 0
        for day in rng.permutation(days):
            chosen = _sample_day(rows, int(day), config.neg_sample_rate, rng)
            day_total, day_count = _train_day(
                model,
                optimizer,
                table,
                rows=rows,
                chosen=chosen,
                day=int(day),
                config=config,
            )
            total, count = total + day_total, count + day_count
        logger.info(
            "epoch %d: %d seeds, mean loss %.4f",
            epoch + 1,
            count,
            total / max(count, 1),
        )
    return model


@torch.no_grad()
def score_day(
    model: GnnModel,
    table: EdgeTable,
    positions: np.ndarray,
    day: int,
    config: GnnConfig,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Score Transactions of one day through Snapshot(day).

    Embeds each Account that appears on the day once, then runs the edge head
    on every Transaction. An empty snapshot gives every Account the same cold
    embedding.

    Returns:
        Logits of shape (n,) and embeddings of shape (n, embedding_dim), in the
        order of `positions`.
    """
    torch.manual_seed(seed)
    model.eval()
    src, dst = table.src[positions], table.dst[positions]
    nodes = torch.unique(torch.cat([src, dst]))
    snapshot = build_snapshot(table, day, config)
    states = torch.zeros(table.num_nodes, config.hidden)
    if snapshot is None:
        cold = model.encode_nodes(
            torch.zeros(1, model.node_in.in_features),
            torch.empty(2, 0, dtype=torch.long),
            torch.empty(0, table.attr.shape[1]),
        )
        states[nodes] = cold[0]
    else:
        loader = NeighborLoader(
            snapshot,
            num_neighbors=config.neighbors,
            input_nodes=nodes,
            batch_size=config.score_batch_size,
            shuffle=False,
        )
        for batch in loader:
            h = model.encode_nodes(batch.x, batch.edge_index, batch.edge_attr)
            states[batch.n_id[: batch.batch_size]] = h[: batch.batch_size]
    logits, embeddings = [], []
    for start in range(0, len(positions), _HEAD_CHUNK):
        part = slice(start, start + _HEAD_CHUNK)
        logit, embedding = model.score_edges(
            states[src[part]], states[dst[part]], table.attr[positions[part]]
        )
        logits.append(logit)
        embeddings.append(embedding)
    del snapshot
    gc.collect()
    return torch.cat(logits).numpy(), torch.cat(embeddings).numpy()
