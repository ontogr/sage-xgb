"""End-to-end tests for the prepare stage on the committed sample.

Expected values come from the sample's Trans CSV read with the standard `csv`
module, not from the stage code. The `prepared_config` session fixture runs the
stage once. Tests that need a fresh run or a changed input use their own
artifacts directory from `config_factory`.
"""

import csv
import shutil
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from sage_xgb import artifacts, pipeline
from sage_xgb.config import Config
from sage_xgb.data import prepare
from sage_xgb.data.vocab import USD_PER_UNIT, VocabularyError


SAMPLE_DIR = Path(__file__).parents[1] / "fixtures" / "amlworld_hi_small_sample"
TRANS_FILE = "HI-Small_Trans.csv"
TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M"


def _csv_data_rows() -> list[list[str]]:
    with (SAMPLE_DIR / TRANS_FILE).open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader)
        return list(reader)


def _set_first_payment_currency(raw_dir: Path, currency: str) -> None:
    """Replace the payment currency of the first data row in a copied CSV."""
    path = raw_dir / TRANS_FILE
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    column = rows[0].index("Payment Currency")
    rows[1][column] = currency
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle, lineterminator="\n").writerows(rows)


class TestPrepare:
    """Prepare stage output for the committed sample."""

    def test_keeps_every_csv_data_row_once(self, prepared_config: Config) -> None:
        """Each CSV data row appears once and row_id covers 0 to n-1."""
        row_count = len(_csv_data_rows())
        transactions = prepare.read_transactions(prepared_config)
        assert transactions.height == row_count
        assert sorted(transactions["row_id"].to_list()) == list(range(row_count))

    def test_row_id_is_the_zero_based_csv_row_index(
        self, prepared_config: Config
    ) -> None:
        """Each row keeps the timestamp, bank and label of its CSV row."""
        csv_rows = _csv_data_rows()
        transactions = prepare.read_transactions(prepared_config).sort("row_id")
        assert transactions["timestamp"].to_list() == [
            datetime.strptime(row[0], TIMESTAMP_FORMAT) for row in csv_rows
        ]
        assert transactions["src_bank"].to_list() == [row[1] for row in csv_rows]
        assert transactions["is_laundering"].to_list() == [
            int(row[10]) for row in csv_rows
        ]

    def test_rows_are_sorted_by_timestamp_then_row_id(
        self, prepared_config: Config
    ) -> None:
        """Rows are in ascending (timestamp, row_id) order."""
        transactions = prepare.read_transactions(prepared_config)
        keys = list(
            zip(
                transactions["timestamp"].to_list(),
                transactions["row_id"].to_list(),
                strict=True,
            )
        )
        assert keys == sorted(keys)

    def test_every_split_has_rows_and_positives(self, prepared_config: Config) -> None:
        """Train, val, test and tail each hold rows and illicit rows."""
        transactions = prepare.read_transactions(prepared_config)
        summary = transactions.group_by("split").agg(
            rows=pl.len(), positives=pl.col("is_laundering").sum()
        )
        by_split = {row["split"]: row for row in summary.iter_rows(named=True)}
        assert set(by_split) == {"train", "val", "test", "tail"}
        for name, row in by_split.items():
            assert row["rows"] > 0, name
            assert row["positives"] > 0, name

    def test_splits_follow_day_order(self, prepared_config: Config) -> None:
        """Each split starts on a later day than the split before it."""
        transactions = prepare.read_transactions(prepared_config)
        bounds = transactions.group_by("split").agg(
            first=pl.col("day").min(), last=pl.col("day").max()
        )
        by_split = {row["split"]: row for row in bounds.iter_rows(named=True)}
        assert by_split["train"]["last"] < by_split["val"]["first"]
        assert by_split["val"]["first"] < by_split["test"]["first"]
        assert by_split["test"]["first"] < by_split["tail"]["first"]

    def test_accounts_are_distinct_bank_and_number_pairs(
        self, prepared_config: Config
    ) -> None:
        """An Account is a (bank, number) pair, not a bare account number."""
        csv_rows = _csv_data_rows()
        pairs = {(row[1], row[2]) for row in csv_rows} | {
            (row[3], row[4]) for row in csv_rows
        }
        numbers = {row[2] for row in csv_rows} | {row[4] for row in csv_rows}
        accounts = prepare.read_accounts(prepared_config)
        assert accounts.height == len(pairs)
        account_pairs = set(
            zip(accounts["bank"].to_list(), accounts["number"].to_list(), strict=True)
        )
        assert account_pairs == pairs
        assert len(pairs) > len(numbers)

    def test_node_ids_follow_first_appearance_in_sorted_order(
        self, prepared_config: Config
    ) -> None:
        """Node ids count up in the order Accounts first appear, sender first."""
        transactions = prepare.read_transactions(prepared_config)
        accounts = prepare.read_accounts(prepared_config)
        endpoints = [
            key
            for pair in zip(
                transactions["src"].to_list(),
                transactions["dst"].to_list(),
                strict=True,
            )
            for key in pair
        ]
        first_appearance = list(dict.fromkeys(endpoints))
        assert accounts["account"].to_list() == first_appearance
        assert accounts["node_id"].to_list() == list(range(len(first_appearance)))
        node_of = dict(
            zip(
                accounts["account"].to_list(),
                accounts["node_id"].to_list(),
                strict=True,
            )
        )
        assert transactions["src_node"].to_list() == [
            node_of[key] for key in transactions["src"].to_list()
        ]
        assert transactions["dst_node"].to_list() == [
            node_of[key] for key in transactions["dst"].to_list()
        ]

    def test_amount_usd_is_paid_amount_times_the_fixed_rate(
        self, prepared_config: Config
    ) -> None:
        """amount_usd converts the paid amount with the paid currency rate."""
        transactions = prepare.read_transactions(prepared_config)
        expected = [
            amount * USD_PER_UNIT[currency]
            for amount, currency in zip(
                transactions["amount_paid"].to_list(),
                transactions["paid_currency"].to_list(),
                strict=True,
            )
        ]
        assert transactions["amount_usd"].to_list() == pytest.approx(expected)

    def test_bank_ids_keep_leading_zeros(self, prepared_config: Config) -> None:
        """Bank ids are read as strings, so a leading zero survives."""
        transactions = prepare.read_transactions(prepared_config)
        assert any(bank.startswith("0") for bank in transactions["src_bank"].to_list())

    def test_two_runs_write_identical_parquet_files(
        self, config_factory: Callable[..., Config], tmp_path: Path
    ) -> None:
        """Two runs in different directories write byte-identical files."""
        first = config_factory(tmp_path / "first")
        second = config_factory(tmp_path / "second")
        pipeline.execute_stage(first, "prepare")
        pipeline.execute_stage(second, "prepare")
        for name in ("transactions.parquet", "accounts.parquet"):
            first_digest = artifacts.file_digest(
                artifacts.stage_path(first, "prepare", name)
            )
            second_digest = artifacts.file_digest(
                artifacts.stage_path(second, "prepare", name)
            )
            assert first_digest == second_digest, name

    def test_writes_a_manifest_that_is_fresh(self, prepared_config: Config) -> None:
        """The manifest exists and matches the current config and inputs."""
        assert artifacts.manifest_path(prepared_config, "prepare").is_file()
        assert artifacts.is_fresh(prepared_config, "prepare")

    def test_second_run_skips_prepare(self, prepared_config: Config) -> None:
        """Running the pipeline again leaves the transactions file untouched."""
        path = artifacts.stage_path(prepared_config, "prepare", "transactions.parquet")
        modified_before = path.stat().st_mtime_ns
        pipeline.run(prepared_config, through="prepare")
        assert path.stat().st_mtime_ns == modified_before

    def test_unknown_currency_fails_and_names_the_value(
        self, config_factory: Callable[..., Config], tmp_path: Path
    ) -> None:
        """A currency outside the vocabulary stops the stage and is named."""
        raw_dir = tmp_path / "raw"
        shutil.copytree(SAMPLE_DIR, raw_dir)
        _set_first_payment_currency(raw_dir, "Zorkmid")
        config = config_factory(tmp_path / "artifacts", raw_dir=raw_dir)
        with pytest.raises(VocabularyError, match="Zorkmid"):
            pipeline.execute_stage(config, "prepare")
