"""Unit tests for the validation of tune search-space entries."""

import pytest
from pydantic import ValidationError

from sage_xgb import cli
from sage_xgb.config import SpaceEntry


class TestSpaceEntry:
    """Bounds and choices are checked when the config loads."""

    @pytest.mark.parametrize(
        "fields",
        [
            {"kind": "float", "low": 2.0, "high": 1.0},
            {"kind": "log_float", "low": 0.0, "high": 1.0},
            {"kind": "log_float", "low": -1.0, "high": 1.0},
            {"kind": "int", "low": 1.5, "high": 4.0},
            {"kind": "int", "low": 1.0, "high": 4.5},
            {"kind": "float", "low": 1.0},
            {"kind": "categorical"},
            {"kind": "categorical", "choices": []},
        ],
    )
    def test_rejects_invalid_entry(self, fields: dict[str, object]) -> None:
        """Reject reversed, non-positive, fractional or missing bounds."""
        with pytest.raises(ValidationError):
            SpaceEntry(**fields)

    @pytest.mark.parametrize(
        "fields",
        [
            {"kind": "int", "low": 3, "high": 9},
            {"kind": "int", "low": 3, "high": 3},
            {"kind": "float", "low": 0.5, "high": 1.0},
            {"kind": "log_float", "low": 0.001, "high": 0.02},
            {"kind": "categorical", "choices": [16, 32]},
        ],
    )
    def test_accepts_valid_entry(self, fields: dict[str, object]) -> None:
        """Accept whole-number int bounds, ordered bounds and non-empty choices."""
        assert SpaceEntry(**fields).kind == fields["kind"]


class TestCliConfigErrors:
    """A stage that rejects the config exits with the config error code."""

    def test_tune_without_the_gnn_reports_a_config_error(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Print one line and return exit code 2 instead of a traceback."""
        code = cli.main(
            ["tune", "--config", "configs/sample.yaml", "--set", "xgb.use_gnn=false"]
        )

        assert code == 2
        assert "config error: tune scores the GNN proxy" in capsys.readouterr().err
