"""Run configuration: one YAML file per Variant, validated at startup."""

from __future__ import annotations

import copy
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class ConfigError(ValueError):
    """Error for a config file or override that is missing or invalid."""


class _Section(BaseModel):
    """Base class for config sections. Unknown keys fail at startup."""

    model_config = ConfigDict(extra="forbid")


class PathsConfig(_Section):
    """Where artifacts are written."""

    artifacts_dir: Path = Path("artifacts")


class DataConfig(_Section):
    """Raw AMLworld files."""

    name: str = "HI-Small"
    raw_dir: Path = Path("data")
    kaggle_dataset: str = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"


class SplitConfig(_Section):
    """Day boundaries of the split and the cross-fit day blocks.

    `train_end`, `val_end` and `test_end` are exclusive. Rows on or after
    `test_end` are the Tail. `crossfit_blocks` lists 1-based day numbers and
    must cover the training days in order, each block a run of consecutive days.
    """

    window_start: date
    train_end: date
    val_end: date
    test_end: date
    crossfit_blocks: list[list[int]]

    @model_validator(mode="after")
    def _check_boundaries(self) -> SplitConfig:
        if not self.window_start < self.train_end < self.val_end < self.test_end:
            msg = (
                "split dates must satisfy window_start < train_end < val_end < test_end"
            )
            raise ValueError(msg)
        train_days = (self.train_end - self.window_start).days
        flat = [day for block in self.crossfit_blocks for day in block]
        if len(self.crossfit_blocks) < 2 or flat != list(range(1, train_days + 1)):
            msg = (
                "crossfit_blocks must be at least two blocks that list every "
                f"training day 1..{train_days} once, in order"
            )
            raise ValueError(msg)
        return self


class FeaturesConfig(_Section):
    """Feature table options."""

    exclude: list[str] = Field(default_factory=list)


class XgbConfig(_Section):
    """XGBoost stage."""

    use_gnn: bool = True
    n_estimators: int = 400
    early_stopping_rounds: int = 30
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: float = 1.0
    reg_lambda: float = 1.0
    scale_pos_weight: float = 1.0
    num_threads: int = 8


class GnnConfig(_Section):
    """GNN stage. The defaults are sized for the 10-minute `run` budget."""

    encoder: Literal["edge_sage", "sage"] = "edge_sage"
    hidden: int = 32
    layers: int = 2
    embedding_dim: int = 8
    neighbors: list[int] = Field(default_factory=lambda: [5, 3])
    batch_size: int = 2048
    score_batch_size: int = 8192
    lr: float = 0.005
    epochs: int = 1
    neg_sample_rate: float = 0.02
    snapshot_lag_days: int = 0
    history_days: int = 7
    num_threads: int = 8
    num_workers: int = 0

    @model_validator(mode="after")
    def _check_shapes(self) -> GnnConfig:
        if len(self.neighbors) != self.layers:
            msg = "gnn.neighbors needs one entry per layer"
            raise ValueError(msg)
        if not 0.0 < self.neg_sample_rate <= 1.0:
            msg = "gnn.neg_sample_rate must be in (0, 1]"
            raise ValueError(msg)
        return self


class TrackingConfig(_Section):
    """MLflow tracking."""

    enabled: bool = True
    experiment: str = "sage-xgb"


class SpaceEntry(_Section):
    """One searched hyperparameter."""

    kind: Literal["int", "float", "log_float", "categorical"]
    low: float | None = None
    high: float | None = None
    choices: list[Any] | None = None


def _default_space() -> dict[str, SpaceEntry]:
    return {
        "gnn.hidden": SpaceEntry(kind="categorical", choices=[16, 32, 64]),
        "gnn.lr": SpaceEntry(kind="log_float", low=0.001, high=0.02),
        "gnn.neg_sample_rate": SpaceEntry(kind="log_float", low=0.01, high=0.2),
        "xgb.max_depth": SpaceEntry(kind="int", low=3, high=9),
        "xgb.learning_rate": SpaceEntry(kind="log_float", low=0.03, high=0.3),
        "xgb.subsample": SpaceEntry(kind="float", low=0.6, high=1.0),
        "xgb.colsample_bytree": SpaceEntry(kind="float", low=0.5, high=1.0),
        "xgb.min_child_weight": SpaceEntry(kind="log_float", low=1.0, high=20.0),
        "xgb.reg_lambda": SpaceEntry(kind="log_float", low=0.1, high=10.0),
    }


class TuneConfig(_Section):
    """Optuna study limits and the searched keys (dotted config paths)."""

    n_trials: int = 20
    timeout_minutes: float = 10.0
    space: dict[str, SpaceEntry] = Field(default_factory=_default_space)


class Config(_Section):
    """The full run configuration."""

    variant: str
    seed: int = 7
    paths: PathsConfig = Field(default_factory=PathsConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    split: SplitConfig
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    xgb: XgbConfig = Field(default_factory=XgbConfig)
    gnn: GnnConfig = Field(default_factory=GnnConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    tune: TuneConfig = Field(default_factory=TuneConfig)


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge `override` into a copy of `base`. Lists and scalars replace."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        msg = f"config file not found: {path}"
        raise ConfigError(msg) from error
    except yaml.YAMLError as error:
        msg = f"config file is not valid YAML: {path}: {error}"
        raise ConfigError(msg) from error
    if not isinstance(loaded, dict):
        msg = f"config file must hold a mapping: {path}"
        raise ConfigError(msg)
    return loaded


def _apply_override(raw: dict[str, Any], assignment: str) -> None:
    """Apply one `dotted.key=value` override. The value is parsed as YAML."""
    key, separator, text = assignment.partition("=")
    if not separator or not key:
        msg = f"override must look like key=value, got {assignment!r}"
        raise ConfigError(msg)
    parts = key.split(".")
    node = raw
    for part in parts[:-1]:
        child = node.setdefault(part, {})
        if not isinstance(child, dict):
            msg = f"override key {key!r} descends into a non-mapping at {part!r}"
            raise ConfigError(msg)
        node = child
    try:
        node[parts[-1]] = yaml.safe_load(text)
    except yaml.YAMLError as error:
        msg = f"override value is not valid YAML: {assignment!r}"
        raise ConfigError(msg) from error


def load_config(
    path: Path | str, overrides: list[str] | tuple[str, ...] = ()
) -> Config:
    """Load a config file, resolve one level of `extends` and apply overrides.

    Args:
        path: YAML file. A top-level `extends` names a parent file, relative
            to this file. The parent cannot extend another file.
        overrides: `dotted.key=value` strings. Values parse as YAML, so `5`,
            `0.1`, `true` and `[5, 3]` work. They apply last.

    Returns:
        The validated config.

    Raises:
        ConfigError: The file is missing, `extends` is nested, an override is
            malformed, or validation fails (including unknown keys).
    """
    config_path = Path(path)
    raw = _read_yaml(config_path)
    parent_name = raw.pop("extends", None)
    if parent_name is not None:
        parent = _read_yaml(config_path.parent / parent_name)
        if "extends" in parent:
            msg = "extends supports one level only"
            raise ConfigError(msg)
        raw = _merge(parent, raw)
    for assignment in overrides:
        _apply_override(raw, assignment)
    try:
        return Config.model_validate(raw)
    except ValidationError as error:
        raise ConfigError(str(error)) from error


def dump_config(config: Config) -> dict[str, Any]:
    """Return the config as plain YAML-safe data (paths and dates as strings)."""
    return yaml.safe_load(yaml.safe_dump(config.model_dump(mode="json")))
