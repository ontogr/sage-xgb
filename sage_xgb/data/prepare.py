"""Sort the raw Transactions and write the prepared tables.

The stage writes `transactions.parquet`, ordered by timestamp with the raw row
id as the tiebreak, and `accounts.parquet`, which gives each Account a dense
node id in order of first appearance.
"""

from __future__ import annotations

import polars as pl

from sage_xgb import artifacts, split
from sage_xgb.config import Config
from sage_xgb.data.load import check_vocabulary, scan_raw
from sage_xgb.data.vocab import USD_PER_UNIT


_STAGE = "prepare"
_TRANSACTIONS_FILE = "transactions.parquet"
_ACCOUNTS_FILE = "accounts.parquet"
_TRANSACTION_COLUMNS: tuple[str, ...] = (
    "row_id",
    "timestamp",
    "day",
    "split",
    "src",
    "dst",
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
    "is_laundering",
)


def _build_accounts(transactions: pl.DataFrame) -> pl.DataFrame:
    """Return one row per Account, numbered in order of first appearance.

    The rows are walked in the order given, and within a row the sender comes
    before the receiver. The output columns are `account`, `node_id` (Int32),
    `bank` and `number`.
    """
    endpoints = transactions.select(
        account=pl.concat_list("src", "dst").explode(),
        bank=pl.concat_list("src_bank", "dst_bank").explode(),
        number=pl.concat_list("src_account", "dst_account").explode(),
    )
    return (
        endpoints.unique(subset="account", keep="first", maintain_order=True)
        .with_row_index("node_id")
        .select(
            "account",
            pl.col("node_id").cast(pl.Int32),
            "bank",
            "number",
        )
    )


def scan_transactions(config: Config) -> pl.LazyFrame:
    """Return the prepared transactions as a lazy scan.

    The prepare stage must have written its output for `config`.
    """
    return pl.scan_parquet(artifacts.stage_path(config, _STAGE, _TRANSACTIONS_FILE))


def read_transactions(config: Config) -> pl.DataFrame:
    """Return the prepared transactions, sorted by timestamp and row id."""
    return scan_transactions(config).collect()


def read_accounts(config: Config) -> pl.DataFrame:
    """Return the Account table, one row per node id in node id order."""
    return pl.read_parquet(artifacts.stage_path(config, _STAGE, _ACCOUNTS_FILE))


def run(config: Config) -> dict[str, float]:
    """Write the prepared transactions and accounts for `config`.

    Args:
        config: The run config. Reads the raw files named by `data` and the
            day boundaries in `split`.

    Returns:
        Row counts: `rows`, `accounts`, and `rows_<split>` for each split name.

    Raises:
        VocabularyError: A currency or payment format is outside the vocabulary.
        ValueError: A Transaction falls before `split.window_start`.
    """
    raw = scan_raw(config.data.raw_dir, config.data.name)
    check_vocabulary(raw)
    transactions = (
        split.add_day_and_split(raw.sort(["timestamp", "row_id"]), config.split)
        .with_columns(
            src=pl.concat_str(["src_bank", "src_account"], separator=":"),
            dst=pl.concat_str(["dst_bank", "dst_account"], separator=":"),
            amount_usd=pl.col("amount_paid")
            * pl.col("paid_currency").replace_strict(
                USD_PER_UNIT, return_dtype=pl.Float64
            ),
        )
        .collect()
    )
    early_rows = (transactions["day"] < 1).sum()
    if early_rows:
        msg = (
            f"{early_rows} transactions fall before "
            f"split.window_start ({config.split.window_start})"
        )
        raise ValueError(msg)

    accounts = _build_accounts(transactions)
    transactions = transactions.with_columns(
        src_node=pl.col("src").replace_strict(
            accounts["account"], accounts["node_id"], return_dtype=pl.Int32
        ),
        dst_node=pl.col("dst").replace_strict(
            accounts["account"], accounts["node_id"], return_dtype=pl.Int32
        ),
    ).select(_TRANSACTION_COLUMNS)

    directory = artifacts.ensure_stage_dir(config, _STAGE)
    transactions.write_parquet(directory / _TRANSACTIONS_FILE, compression="zstd")
    accounts.write_parquet(directory / _ACCOUNTS_FILE, compression="zstd")
    artifacts.write_manifest(config, _STAGE, [_TRANSACTIONS_FILE, _ACCOUNTS_FILE])

    counts = {"rows": float(transactions.height), "accounts": float(accounts.height)}
    for name in split.SPLITS:
        counts[f"rows_{name}"] = float((transactions["split"] == name).sum())
    return counts
