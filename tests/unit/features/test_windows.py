"""Window features on hand-built Transactions.

Each frame lists Transactions as (minute, src_node, dst_node, amount_usd), with
minutes counted from a fixed base time. The expected values follow the window
rule: a Transaction sees the events of its Account in the interval [t - w, t),
so events in its own minute are never counted and an event exactly w minutes
earlier is counted.
"""

from datetime import datetime, timedelta
from typing import Any

import polars as pl
import pytest

from sage_xgb.features.windows import window_features


BASE_TIME = datetime(2022, 9, 1)
WINDOW_WIDTHS = [("1h", 60), ("24h", 1_440), ("7d", 10_080)]


def _transactions(rows: list[tuple[int, int, int, float]]) -> pl.DataFrame:
    """Build a prepared-style frame from (minute, src, dst, usd) tuples.

    The row id is the position in `rows`, so Transactions in one minute keep
    list order.
    """
    return pl.DataFrame(
        {
            "row_id": list(range(len(rows))),
            "timestamp": [BASE_TIME + timedelta(minutes=minute) for minute, *_ in rows],
            "src_node": [src for _, src, _, _ in rows],
            "dst_node": [dst for _, _, dst, _ in rows],
            "amount_usd": [usd for *_, usd in rows],
        },
        schema={
            "row_id": pl.UInt32,
            "timestamp": pl.Datetime("us"),
            "src_node": pl.Int32,
            "dst_node": pl.Int32,
            "amount_usd": pl.Float64,
        },
    )


def _feature_row(features: pl.DataFrame, row_id: int) -> dict[str, Any]:
    return features.filter(pl.col("row_id") == row_id).row(0, named=True)


class TestWindowFeatures:
    """Trailing-window counts, sums, maxima and gaps for each Account role."""

    @pytest.mark.parametrize(("window", "width"), WINDOW_WIDTHS)
    def test_counts_event_exactly_window_minutes_earlier(self, window, width):
        transactions = _transactions([(0, 1, 2, 3.0), (width, 1, 3, 5.0)])

        features = window_features(transactions)

        later = _feature_row(features, 1)
        assert later[f"src_out_count_{window}"] == 1
        assert later[f"src_out_usd_sum_{window}"] == 3.0
        assert later[f"src_out_usd_max_{window}"] == 3.0

    @pytest.mark.parametrize(("window", "width"), WINDOW_WIDTHS)
    def test_ignores_event_one_minute_earlier_than_window(self, window, width):
        transactions = _transactions([(0, 1, 2, 3.0), (width + 1, 1, 3, 5.0)])

        features = window_features(transactions)

        later = _feature_row(features, 1)
        assert later[f"src_out_count_{window}"] == 0
        assert later[f"src_out_usd_sum_{window}"] == 0.0
        assert later[f"src_out_usd_max_{window}"] is None

    @pytest.mark.parametrize("window", ["1h", "24h", "7d"])
    def test_same_minute_transactions_do_not_count_each_other(self, window):
        transactions = _transactions([(5, 1, 2, 3.0), (5, 1, 3, 8.0), (6, 1, 4, 1.0)])

        features = window_features(transactions)

        for row_id in (0, 1):
            assert _feature_row(features, row_id)[f"src_out_count_{window}"] == 0
        after = _feature_row(features, 2)
        assert after[f"src_out_count_{window}"] == 2
        assert after[f"src_out_usd_sum_{window}"] == 11.0
        assert after[f"src_out_usd_max_{window}"] == 8.0

    @pytest.mark.parametrize("window", ["1h", "24h", "7d"])
    def test_account_without_earlier_events_has_empty_windows(self, window):
        transactions = _transactions([(0, 1, 2, 3.0)])

        features = window_features(transactions)

        first = _feature_row(features, 0)
        assert first[f"src_out_count_{window}"] == 0
        assert first[f"src_out_usd_sum_{window}"] == 0.0
        assert first[f"src_out_usd_max_{window}"] is None
        assert first["src_out_minutes_since_last"] is None
        assert first[f"dst_in_count_{window}"] == 0
        assert first[f"dst_in_usd_max_{window}"] is None
        assert first["dst_in_minutes_since_last"] is None

    @pytest.mark.parametrize("window", ["1h", "24h", "7d"])
    def test_sender_in_stats_count_money_it_received_earlier(self, window):
        transactions = _transactions([(0, 2, 1, 7.0), (5, 1, 3, 4.0)])

        features = window_features(transactions)

        sent_later = _feature_row(features, 1)
        assert sent_later[f"src_in_count_{window}"] == 1
        assert sent_later[f"src_in_usd_sum_{window}"] == 7.0
        assert sent_later[f"src_in_usd_max_{window}"] == 7.0
        assert sent_later[f"src_out_count_{window}"] == 0
        assert sent_later["src_in_minutes_since_last"] == 5.0

    @pytest.mark.parametrize("window", ["1h", "24h", "7d"])
    def test_receiver_out_stats_count_money_it_sent_earlier(self, window):
        transactions = _transactions([(0, 1, 2, 4.0), (5, 3, 1, 9.0)])

        features = window_features(transactions)

        received_later = _feature_row(features, 1)
        assert received_later[f"dst_out_count_{window}"] == 1
        assert received_later[f"dst_out_usd_sum_{window}"] == 4.0
        assert received_later[f"dst_out_usd_max_{window}"] == 4.0
        assert received_later[f"dst_in_count_{window}"] == 0
        assert received_later["dst_out_minutes_since_last"] == 5.0

    def test_minutes_since_last_skips_events_in_the_same_minute(self):
        transactions = _transactions([(0, 1, 2, 1.0), (0, 1, 3, 1.0), (4, 1, 2, 1.0)])

        features = window_features(transactions)

        assert _feature_row(features, 0)["src_out_minutes_since_last"] is None
        assert _feature_row(features, 1)["src_out_minutes_since_last"] is None
        assert _feature_row(features, 2)["src_out_minutes_since_last"] == 4.0

    def test_self_transfer_counts_as_out_and_in_for_later_transactions(self):
        transactions = _transactions([(0, 1, 1, 6.0), (3, 1, 2, 2.0)])

        features = window_features(transactions)

        own_minute = _feature_row(features, 0)
        assert own_minute["src_in_count_1h"] == 0
        assert own_minute["dst_out_count_1h"] == 0
        later = _feature_row(features, 1)
        assert later["src_out_count_1h"] == 1
        assert later["src_in_count_1h"] == 1
        assert later["src_in_usd_sum_1h"] == 6.0
