"""Checks the committed HI-Small sample against the guarantees of its builder.

The sample is written by scripts/build_sample.py from the raw files in data/.
These tests read only the committed files under
tests/fixtures/amlworld_hi_small_sample, so they run without the raw data. Day
numbers and the Tail come from the split definition in configs/hi_small.yaml.
Run `make sample` to regenerate the fixture after a change to the builder.
"""

from collections import defaultdict
from pathlib import Path

import polars as pl
import pytest

from sage_xgb.config import load_config
from sage_xgb.split import add_day_and_split


SAMPLE_DIR = Path(__file__).parents[1] / "fixtures" / "amlworld_hi_small_sample"
HI_SMALL_CONFIG = Path(__file__).parents[2] / "configs" / "hi_small.yaml"
TRANS_COLUMNS = (
    "timestamp",
    "from_bank",
    "from_account",
    "to_bank",
    "to_account",
    "amount_received",
    "receiving_currency",
    "amount_paid",
    "payment_currency",
    "payment_format",
    "is_laundering",
)
DAY_BLOCKS = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10))
MIN_ILLICIT_PER_BLOCK = 10
MIN_ILLICIT_TAIL = 3
GAP_MINUTES = (60, 1440, 10080)


@pytest.fixture(scope="module")
def transactions() -> pl.DataFrame:
    """Return the committed transactions with day, split and Account keys."""
    split = load_config(HI_SMALL_CONFIG).split
    frame = (
        pl.scan_csv(
            SAMPLE_DIR / "HI-Small_Trans.csv",
            has_header=False,
            skip_rows=1,
            new_columns=list(TRANS_COLUMNS),
            infer_schema=False,
        )
        .with_columns(timestamp=pl.col("timestamp").str.to_datetime("%Y/%m/%d %H:%M"))
        .collect()
    )
    return add_day_and_split(frame, split).with_columns(
        illicit=pl.col("is_laundering") == "1",
        sender=pl.concat_str(["from_bank", "from_account"], separator=":"),
        receiver=pl.concat_str(["to_bank", "to_account"], separator=":"),
        minute=pl.col("timestamp").dt.epoch("s") // 60,
    )


def _minutes_by_account(frame: pl.DataFrame, role: str) -> dict[str, set[int]]:
    """Return the set of minutes of the sends or receives of each Account."""
    minutes: dict[str, set[int]] = defaultdict(set)
    for account, minute in frame.select(role, "minute").iter_rows():
        minutes[account].add(minute)
    return minutes


class TestSampleFixture:
    """The committed sample keeps the properties its builder guarantees."""

    @pytest.mark.parametrize(("first_day", "last_day"), DAY_BLOCKS)
    def test_each_day_block_has_at_least_ten_illicit_rows(
        self, transactions, first_day, last_day
    ):
        """Each day block of days 1 to 10 has enough illicit rows to train on."""
        block = transactions.filter(
            pl.col("day").is_between(first_day, last_day) & pl.col("illicit")
        )

        assert block.height >= MIN_ILLICIT_PER_BLOCK

    def test_tail_has_at_least_three_illicit_rows(self, transactions):
        """The Tail, after the test days, has enough illicit rows to report."""
        tail = transactions.filter((pl.col("split") == "tail") & pl.col("illicit"))

        assert tail.height >= MIN_ILLICIT_TAIL

    def test_one_account_has_two_sends_in_the_same_minute(self, transactions):
        """Some Account sends two Transactions in one minute to two peers."""
        same_minute = (
            transactions.group_by("sender", "minute")
            .agg(
                pl.len().alias("sends"),
                pl.col("receiver").n_unique().alias("peers"),
            )
            .filter((pl.col("sends") >= 2) & (pl.col("peers") >= 2))
        )

        assert same_minute.height > 0

    def test_one_account_has_sends_60_1440_and_10080_minutes_after_one_send(
        self, transactions
    ):
        """One Account has sends exactly 60, 1,440 and 10,080 minutes apart."""
        minutes = _minutes_by_account(transactions, "sender")

        assert any(
            all(base + gap in account_minutes for gap in GAP_MINUTES)
            for account_minutes in minutes.values()
            for base in account_minutes
        )

    def test_one_account_has_two_receives_60_minutes_apart(self, transactions):
        """Some Account receives two Transactions exactly 60 minutes apart."""
        minutes = _minutes_by_account(transactions, "receiver")

        assert any(
            base + GAP_MINUTES[0] in account_minutes
            for account_minutes in minutes.values()
            for base in account_minutes
        )

    def test_one_account_number_appears_under_two_banks(self, transactions):
        """Some account number is used under two different bank ids."""
        banks_by_number = (
            pl.concat(
                [
                    transactions.select(
                        number=pl.col("from_account"), bank=pl.col("from_bank")
                    ),
                    transactions.select(
                        number=pl.col("to_account"), bank=pl.col("to_bank")
                    ),
                ]
            )
            .group_by("number")
            .agg(pl.col("bank").n_unique().alias("banks"))
        )

        assert banks_by_number.filter(pl.col("banks") >= 2).height > 0

    def test_every_pattern_transaction_line_appears_in_the_transactions_file(self):
        """Each Patterns transaction line equals a row of the transactions file."""
        trans_lines = set(
            (SAMPLE_DIR / "HI-Small_Trans.csv")
            .read_text(encoding="utf-8")
            .splitlines()[1:]
        )
        pattern_lines = [
            line
            for line in (SAMPLE_DIR / "HI-Small_Patterns.txt")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.count(",") == len(TRANS_COLUMNS) - 1
        ]

        assert pattern_lines
        assert [line for line in pattern_lines if line not in trans_lines] == []
