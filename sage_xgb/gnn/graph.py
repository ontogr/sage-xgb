"""Daily snapshot graphs built from the time-sorted transaction table.

Snapshot(d) holds the Transactions of the `history_days` days that end
`snapshot_lag_days` days before day d starts. No edge from day d, or from any
later day, can reach a message of day d, because those edges are not in the
graph. Each Transaction is stored as a forward edge and a reverse edge so the
encoder sees both the money an Account sent and the money it received.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
import torch
from torch_geometric.data import Data

from sage_xgb.config import GnnConfig
from sage_xgb.data import vocab


# Edge attribute columns: seven continuous values, the direction flag, then the
# integer codes of the paid currency, received currency and payment format.
CONTINUOUS_COLUMNS = 7
DIRECTION_COLUMN = 7
CATEGORY_START = 8
EDGE_ATTR_DIM = 11

# Node features: count, USD sum and USD max of sent and received Transactions,
# over the last day of the snapshot and over the whole snapshot window.
NODE_FEATURE_DIM = 12

_LOG_SCALE = 10.0
_RATIO_CLIP = 5.0

_REQUIRED_COLUMNS = (
    "row_id",
    "timestamp",
    "day",
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


@dataclass(frozen=True)
class EdgeTable:
    """Time-sorted Transaction arrays shared by every snapshot.

    Row i of every array is the same Transaction. `src`, `dst` and the edge
    attributes are tensors so snapshots are slices, not rebuilds.
    """

    row_id: np.ndarray
    day: np.ndarray
    src: torch.Tensor
    dst: torch.Tensor
    usd: np.ndarray
    attr: torch.Tensor
    num_nodes: int
    position_of_row: np.ndarray

    def positions(self, row_ids: np.ndarray) -> np.ndarray:
        """Return the table position of each `row_id`."""
        return self.position_of_row[row_ids]


def build_edge_table(transactions: pl.DataFrame) -> EdgeTable:
    """Build the edge table from prepared Transactions without labels.

    Args:
        transactions: Prepared rows sorted by (`timestamp`, `row_id`). The label
            column is not read and may be absent.

    Raises:
        ValueError: A required column is missing, or the days are not sorted.
    """
    missing = [name for name in _REQUIRED_COLUMNS if name not in transactions.columns]
    if missing:
        msg = f"transactions lack columns: {missing}"
        raise ValueError(msg)
    day = transactions["day"].to_numpy()
    if np.any(np.diff(day) < 0):
        msg = "transactions must be sorted by timestamp"
        raise ValueError(msg)
    currency_codes = {name: code for code, name in enumerate(vocab.CURRENCIES)}
    format_codes = {name: code for code, name in enumerate(vocab.PAYMENT_FORMATS)}
    hour = pl.col("timestamp").dt.hour() + pl.col("timestamp").dt.minute() / 60.0
    angle = hour * (2.0 * np.pi / 24.0)
    ratio = (pl.col("amount_received") / pl.col("amount_paid")).log()
    columns = transactions.select(
        (pl.col("amount_usd").log1p() / _LOG_SCALE).alias("log_usd"),
        (ratio.fill_nan(0.0).fill_null(0.0).clip(-_RATIO_CLIP, _RATIO_CLIP))
        .truediv(_RATIO_CLIP)
        .alias("log_ratio"),
        (pl.col("src_bank") == pl.col("dst_bank")).cast(pl.Float32).alias("same_bank"),
        (pl.col("paid_currency") != pl.col("received_currency"))
        .cast(pl.Float32)
        .alias("cross_currency"),
        (pl.col("src_node") == pl.col("dst_node")).cast(pl.Float32).alias("self"),
        angle.sin().alias("hour_sin"),
        angle.cos().alias("hour_cos"),
        pl.lit(0.0).alias("direction"),
        pl.col("paid_currency").replace_strict(currency_codes).alias("paid_code"),
        pl.col("received_currency")
        .replace_strict(currency_codes)
        .alias("received_code"),
        pl.col("payment_format").replace_strict(format_codes).alias("format_code"),
    )
    attr = torch.from_numpy(
        columns.cast(pl.Float32).to_numpy(order="c").astype(np.float32, copy=False)
    )
    row_id = transactions["row_id"].to_numpy()
    src_ids = transactions["src_node"].to_numpy()
    dst_ids = transactions["dst_node"].to_numpy()
    position_of_row = np.full(int(row_id.max()) + 1, -1, dtype=np.int64)
    position_of_row[row_id] = np.arange(len(row_id))
    return EdgeTable(
        row_id=row_id,
        day=day,
        src=torch.from_numpy(transactions["src_node"].to_numpy().astype(np.int64)),
        dst=torch.from_numpy(transactions["dst_node"].to_numpy().astype(np.int64)),
        usd=transactions["amount_usd"].to_numpy(),
        attr=attr,
        num_nodes=int(max(src_ids.max(), dst_ids.max())) + 1,
        position_of_row=position_of_row,
    )


def snapshot_bounds(table: EdgeTable, day: int, config: GnnConfig) -> tuple[int, int]:
    """Return the table slice [lo, hi) of the Transactions in Snapshot(day)."""
    horizon = day - config.snapshot_lag_days
    lo = int(np.searchsorted(table.day, horizon - config.history_days, side="left"))
    hi = int(np.searchsorted(table.day, horizon, side="left"))
    return lo, hi


def _node_stats(
    table: EdgeTable, lo: int, hi: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return count, sum and max of sent then received USD per node over a slice."""
    src = table.src[lo:hi].numpy()
    dst = table.dst[lo:hi].numpy()
    usd = table.usd[lo:hi]
    n = table.num_nodes
    out_max = np.zeros(n)
    in_max = np.zeros(n)
    np.maximum.at(out_max, src, usd)
    np.maximum.at(in_max, dst, usd)
    return (
        np.bincount(src, minlength=n),
        np.bincount(src, weights=usd, minlength=n),
        out_max,
        np.bincount(dst, minlength=n),
        np.bincount(dst, weights=usd, minlength=n),
        in_max,
    )


def node_features(table: EdgeTable, day: int, config: GnnConfig) -> torch.Tensor:
    """Return the fixed-transform node features of Snapshot(day).

    The first six columns cover the last day before the snapshot time and the
    last six cover the whole snapshot window.
    """
    lo, hi = snapshot_bounds(table, day, config)
    horizon = day - config.snapshot_lag_days
    recent_lo = int(np.searchsorted(table.day, horizon - 1, side="left"))
    blocks = [*_node_stats(table, recent_lo, hi), *_node_stats(table, lo, hi)]
    stacked = np.stack(blocks, axis=1)
    return torch.from_numpy(np.log1p(stacked) / _LOG_SCALE).float()


def build_snapshot(table: EdgeTable, day: int, config: GnnConfig) -> Data | None:
    """Return Snapshot(day) as a bidirected PyG graph, or None if it is empty."""
    lo, hi = snapshot_bounds(table, day, config)
    if hi == lo:
        return None
    src, dst = table.src[lo:hi], table.dst[lo:hi]
    forward = table.attr[lo:hi]
    reverse = forward.clone()
    reverse[:, DIRECTION_COLUMN] = 1.0
    return Data(
        x=node_features(table, day, config),
        edge_index=torch.stack([torch.cat([src, dst]), torch.cat([dst, src])]),
        edge_attr=torch.cat([forward, reverse]),
        num_nodes=table.num_nodes,
    )
