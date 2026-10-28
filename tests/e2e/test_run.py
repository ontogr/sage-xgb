"""End-to-end tests for `pipeline.run`, stage skipping and the MLflow run log.

Most tests read the session fixture `pipeline_config`, which runs every stage on
the committed sample once. The label-flip test reruns the whole pipeline on a
perturbed copy of the sample, so this module is one of the slower end-to-end
modules. Run it with `make test-e2e`.
"""

import shutil
from datetime import date
from pathlib import Path

import pytest

from sage_xgb import artifacts, pipeline, tracking
from sage_xgb.config import Config, ConfigError, load_config


SAMPLE_DIR = Path(__file__).parents[1] / "fixtures" / "amlworld_hi_small_sample"
SAMPLE_CONFIG = Path(__file__).parents[2] / "configs" / "sample.yaml"
TRANS_FILE = "HI-Small_Trans.csv"
LABEL_FREE_OUTPUTS = (
    ("train-xgb", "booster.json"),
    ("train-xgb", "metrics_val.json"),
    ("gnn", "gnn_scores.parquet"),
)


def _manifest_mtimes(config: Config) -> dict[str, int]:
    """Return the modification time of each stage manifest in nanoseconds."""
    return {
        stage: artifacts.manifest_path(config, stage).stat().st_mtime_ns
        for stage in artifacts.PIPELINE_STAGES
    }


def _output_digest(config: Config, stage: str, filename: str) -> str:
    """Return the SHA-256 digest of one stage output."""
    return artifacts.file_digest(artifacts.stage_path(config, stage, filename))


def _flip_labels_from(source: Path, target: Path, first_day: date) -> int:
    """Write a copy of a Trans CSV with the label flipped from `first_day` on.

    Rows are edited in place, so every byte outside the label column stays the
    same. A timestamp starts with `YYYY/MM/DD`, so comparing that prefix with
    the cutoff compares days. Returns the number of flipped rows.
    """
    with source.open(newline="", encoding="utf-8") as handle:
        header, *rows = handle.readlines()
    cutoff = first_day.strftime("%Y/%m/%d")
    flipped = 0
    for index, line in enumerate(rows):
        body = line.rstrip("\r\n")
        if body[:10] < cutoff:
            continue
        prefix, _, label = body.rpartition(",")
        rows[index] = f"{prefix},{1 - int(label)}{line[len(body) :]}"
        flipped += 1
    with target.open("w", newline="", encoding="utf-8") as handle:
        handle.write("".join([header, *rows]))
    return flipped


class TestRun:
    """Stage outputs, stage skipping, reruns, tracking and config overrides."""

    @pytest.mark.parametrize(
        ("stage", "filename"),
        [
            ("prepare", "transactions.parquet"),
            ("prepare", "accounts.parquet"),
            ("features", "features.parquet"),
            ("gnn", "gnn_scores.parquet"),
            ("train-xgb", "booster.json"),
            ("train-xgb", "feature_columns.json"),
            ("train-xgb", "metadata.json"),
            ("train-xgb", "metrics_val.json"),
        ],
    )
    def test_writes_each_stage_output(self, pipeline_config, stage, filename):
        """Each stage writes its output files in its stage directory."""
        assert artifacts.stage_path(pipeline_config, stage, filename).is_file()

    @pytest.mark.parametrize("stage", artifacts.PIPELINE_STAGES)
    def test_writes_manifest_for_each_stage(self, pipeline_config, stage):
        """Each stage writes a manifest next to its outputs."""
        assert artifacts.manifest_path(pipeline_config, stage).is_file()

    def test_second_run_skips_every_stage(self, pipeline_config):
        """A second run leaves every stage manifest untouched."""
        before = _manifest_mtimes(pipeline_config)

        pipeline.run(pipeline_config)

        assert _manifest_mtimes(pipeline_config) == before

    def test_from_stage_rewrites_manifests_from_that_stage_on(self, pipeline_config):
        """A run from `gnn` rewrites the gnn and train-xgb manifests only."""
        before = _manifest_mtimes(pipeline_config)

        pipeline.run(pipeline_config, from_stage="gnn")

        after = _manifest_mtimes(pipeline_config)
        assert after["prepare"] == before["prepare"]
        assert after["features"] == before["features"]
        assert after["gnn"] > before["gnn"]
        assert after["train-xgb"] > before["train-xgb"]

    def test_records_a_run_per_stage_and_one_for_the_pipeline(self, pipeline_config):
        """The tracking store holds the pipeline run and one run per stage."""
        names = tracking.list_run_names(pipeline_config)

        assert {"run", "prepare", "features", "gnn", "train-xgb"} <= set(names)

    def test_flipped_test_and_tail_labels_leave_outputs_unchanged(
        self, pipeline_config, config_factory, tmp_path
    ):
        """Flipping test and Tail labels leaves trained and scored outputs unchanged."""
        raw_dir = tmp_path / "raw"
        shutil.copytree(SAMPLE_DIR, raw_dir)
        flipped = _flip_labels_from(
            SAMPLE_DIR / TRANS_FILE,
            raw_dir / TRANS_FILE,
            pipeline_config.split.val_end,
        )
        assert flipped > 0
        perturbed = config_factory(tmp_path / "artifacts", raw_dir=raw_dir)

        pipeline.run(perturbed)

        perturbed_rows = _output_digest(perturbed, "prepare", "transactions.parquet")
        original_rows = _output_digest(
            pipeline_config, "prepare", "transactions.parquet"
        )
        assert perturbed_rows != original_rows
        for stage, filename in LABEL_FREE_OUTPUTS:
            assert _output_digest(perturbed, stage, filename) == _output_digest(
                pipeline_config, stage, filename
            )

    def test_rejects_unknown_config_key(self):
        """A key that the config models do not define fails at load time."""
        with pytest.raises(ConfigError):
            load_config(SAMPLE_CONFIG, ["bogus.key=1"])

    def test_applies_set_overrides_on_top_of_the_file(self):
        """A `--set` override replaces the value from the config file."""
        config = load_config(SAMPLE_CONFIG, ["gnn.epochs=3"])

        assert config.gnn.epochs == 3
