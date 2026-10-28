"""Feature table checks on the committed sample.

The perturbation tests change the prepared rows and call `build_features` in
process. Each one checks a feature that must not move, so a leak from later
rows, from the same minute, from row order or from the label fails the exact
comparison. Stage tests that need another config use their own artifacts
directory, and the rest read the shared `featured_config`.
"""

from datetime import datetime, time, timedelta
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from sage_xgb import artifacts, split
from sage_xgb.config import Config
from sage_xgb.data import prepare
from sage_xgb.features import table


SIDES = ("src", "dst")
ROLES = ("out", "in")
STATS = ("count", "usd_sum", "usd_max")
WINDOWS = ("1h", "24h", "7d")
WINDOW_MINUTES = [("1h", 60), ("24h", 1_440), ("7d", 10_080)]

ROW_FEATURES = [
    "hour",
    "weekday",
    "log_amount_usd",
    "amount_ratio",
    "same_bank",
    "is_cross_currency",
    "is_self_transfer",
    "paid_currency_code",
    "payment_format_code",
]
WINDOW_FEATURES = [
    f"{side}_{role}_{stat}_{window}"
    for side in SIDES
    for role in ROLES
    for stat in STATS
    for window in WINDOWS
]
MINUTE_FEATURES = [
    f"{side}_{role}_minutes_since_last" for side in SIDES for role in ROLES
]
EXPECTED_COLUMNS = ["row_id", *ROW_FEATURES, *WINDOW_FEATURES, *MINUTE_FEATURES]


def _features_of(rows: pl.DataFrame) -> pl.DataFrame:
    return table.build_features(split.unlabelled_rows(rows))


def _account_events(rows: pl.DataFrame) -> pl.DataFrame:
    """Return one event per Transaction side: the sender is out, the receiver in."""
    sent = rows.select(
        "row_id",
        "timestamp",
        account=pl.col("src_node"),
        role=pl.lit("out"),
        side=pl.lit("src"),
    )
    received = rows.select(
        "row_id",
        "timestamp",
        account=pl.col("dst_node"),
        role=pl.lit("in"),
        side=pl.lit("dst"),
    )
    return pl.concat([sent, received])


def _same_minute_pair(rows: pl.DataFrame) -> tuple[int, int, datetime]:
    """Return two row ids of one sender in one minute, and that minute."""
    grouped = rows.filter(pl.len().over("src_node", "timestamp") >= 2).sort(
        "timestamp", "row_id"
    )
    if grouped.is_empty():
        pytest.fail("the sample has no two Transactions in one minute on one sender")
    first = grouped.row(grouped.height - 1, named=True)
    mates = grouped.filter(
        (pl.col("src_node") == first["src_node"])
        & (pl.col("timestamp") == first["timestamp"])
    )
    return mates["row_id"][0], mates["row_id"][1], first["timestamp"]


def _exact_gap_pair(events: pl.DataFrame, width_minutes: int) -> dict[str, Any]:
    """Return a later event and an earlier event of one Account, exactly apart."""
    later = events.with_columns(
        earlier_timestamp=pl.col("timestamp") - timedelta(minutes=width_minutes)
    )
    earlier = events.select(
        earlier_row_id=pl.col("row_id"),
        account=pl.col("account"),
        earlier_role=pl.col("role"),
        earlier_timestamp=pl.col("timestamp"),
    )
    pairs = later.join(earlier, on=["account", "earlier_timestamp"]).sort("row_id")
    if pairs.is_empty():
        pytest.fail(f"the sample has no two events {width_minutes} minutes apart")
    return pairs.row(0, named=True)


def _day_three_start(config: Config) -> datetime:
    return datetime.combine(config.split.window_start, time()) + timedelta(days=2)


class TestFeatures:
    """Leakage guards, edge cases and the features.parquet contract."""

    @pytest.mark.parametrize("cut_kind", ["day_boundary", "same_minute_pair"])
    def test_truncated_history_leaves_remaining_features_unchanged(
        self, prepared_config, cut_kind
    ):
        rows = prepare.read_transactions(prepared_config)
        if cut_kind == "day_boundary":
            cut = _day_three_start(prepared_config)
        else:
            _, _, cut = _same_minute_pair(rows)
        kept = rows.filter(pl.col("timestamp") < cut)
        assert 0 < kept.height < rows.height

        full = _features_of(rows).filter(
            pl.col("row_id").is_in(kept["row_id"].to_list())
        )

        assert_frame_equal(full, _features_of(kept), check_exact=True)

    def test_same_minute_transaction_does_not_change_its_pair(self, prepared_config):
        rows = prepare.read_transactions(prepared_config)
        first, second, _ = _same_minute_pair(rows)
        without_first = rows.filter(pl.col("row_id") != first)

        full = _features_of(rows).filter(pl.col("row_id") == second)
        perturbed = _features_of(without_first).filter(pl.col("row_id") == second)

        assert_frame_equal(full, perturbed, check_exact=True)

    @pytest.mark.parametrize(("window", "width"), WINDOW_MINUTES)
    def test_window_edge_counts_event_exactly_window_minutes_earlier(
        self, prepared_config, featured_config, window, width
    ):
        rows = prepare.read_transactions(prepared_config)
        events = _account_events(rows)
        pair = _exact_gap_pair(events, width)
        later_time = pair["timestamp"]
        earlier_role = pair["earlier_role"]
        in_window = events.filter(
            (pl.col("account") == pair["account"])
            & (pl.col("role") == earlier_role)
            & (pl.col("timestamp") >= later_time - timedelta(minutes=width))
            & (pl.col("timestamp") < later_time)
        )
        assert pair["earlier_row_id"] in in_window["row_id"].to_list()

        features = table.read_features(featured_config)
        later_row = features.filter(pl.col("row_id") == pair["row_id"]).row(
            0, named=True
        )

        column = f"{pair['side']}_{earlier_role}_count_{window}"
        assert later_row[column] == in_window.height

    def test_row_order_does_not_change_features(self, prepared_config):
        rows = prepare.read_transactions(prepared_config)
        shuffled = rows.sample(fraction=1.0, shuffle=True, seed=11)
        assert shuffled["row_id"].to_list() != rows["row_id"].to_list()

        assert_frame_equal(_features_of(rows), _features_of(shuffled), check_exact=True)

    def test_flipping_every_label_leaves_features_unchanged(self, prepared_config):
        rows = prepare.read_transactions(prepared_config)
        flipped = rows.with_columns(is_laundering=1 - pl.col("is_laundering"))

        assert_frame_equal(
            table.build_features(rows),
            table.build_features(flipped),
            check_exact=True,
        )

    def test_features_file_has_row_id_and_documented_float32_columns(
        self, prepared_config, featured_config
    ):
        frame = table.read_features(featured_config)
        rows = prepare.read_transactions(prepared_config)

        assert frame.columns == EXPECTED_COLUMNS
        assert frame.schema["row_id"] == pl.UInt32
        assert set(frame.dtypes[1:]) == {pl.Float32}
        assert frame.height == rows.height
        assert frame["row_id"].is_sorted()
        manifest = artifacts.read_manifest(featured_config, "features")
        assert manifest is not None
        assert manifest["outputs"] == ["features.parquet"]

    @pytest.mark.parametrize("name", ["no_such_column", "row_id"])
    def test_exclude_rejects_names_that_are_not_feature_columns(
        self, prepared_config, name
    ):
        rows = prepare.read_transactions(prepared_config)

        with pytest.raises(ValueError, match=name):
            table.build_features(split.unlabelled_rows(rows), exclude=(name,))

    def test_features_exclude_config_removes_named_column(
        self, config_factory, tmp_path
    ):
        config = config_factory(tmp_path, "features.exclude=[hour]")
        prepare.run(config)

        table.run(config)

        frame = table.read_features(config)
        assert frame.columns == [name for name in EXPECTED_COLUMNS if name != "hour"]

    def test_features_exclude_config_rejects_unknown_name(
        self, config_factory, tmp_path
    ):
        config = config_factory(tmp_path, "features.exclude=[no_such_column]")
        prepare.run(config)

        with pytest.raises(ValueError, match="no_such_column"):
            table.run(config)

    def test_rerunning_features_stage_writes_same_file_bytes(
        self, prepared_config, featured_config
    ):
        path = artifacts.stage_path(featured_config, "features", "features.parquet")
        first = artifacts.file_digest(path)

        metrics = table.run(featured_config)

        assert artifacts.file_digest(path) == first
        assert metrics["rows"] == prepare.read_transactions(prepared_config).height
