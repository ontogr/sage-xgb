"""Artifact paths and stage manifests.

Each stage writes its outputs under `<artifacts_dir>/<variant>/<stage>/` and
ends by writing `manifest.json`. A manifest holds a hash of the config sections
the stage reads and the digests of its upstream manifests, so `run` can skip a
stage whose inputs did not change. Manifests hold no timestamps, so a rerun
with the same inputs writes the same bytes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sage_xgb import __version__
from sage_xgb.config import Config


PIPELINE_STAGES: tuple[str, ...] = ("prepare", "features", "gnn", "train-xgb")

_SECTIONS: dict[str, tuple[str, ...]] = {
    "prepare": ("data", "split"),
    "features": ("features",),
    "gnn": ("gnn", "split", "seed"),
    "train-xgb": ("xgb", "features", "split", "seed"),
}

_UPSTREAM: dict[str, tuple[str, ...]] = {
    "prepare": (),
    "features": ("prepare",),
    "gnn": ("prepare",),
    "train-xgb": ("prepare", "features", "gnn"),
}

MANIFEST_NAME = "manifest.json"


def stage_dir(config: Config, stage: str) -> Path:
    """Return the output directory of a stage without creating it."""
    return config.paths.artifacts_dir / config.variant / stage


def ensure_stage_dir(config: Config, stage: str) -> Path:
    """Return the output directory of a stage, creating it if needed."""
    path = stage_dir(config, stage)
    path.mkdir(parents=True, exist_ok=True)
    return path


def stage_path(config: Config, stage: str, name: str) -> Path:
    """Return the path of one output file of a stage."""
    return stage_dir(config, stage) / name


def file_digest(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def config_hash(config: Config, sections: Sequence[str]) -> str:
    """Return a hash of the named top-level config sections."""
    dumped = config.model_dump(mode="json")
    payload = {name: dumped[name] for name in sections}
    text = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


def upstream_stages(config: Config, stage: str) -> tuple[str, ...]:
    """Return the stages whose manifests feed `stage`.

    `train-xgb` ignores the GNN manifest when `xgb.use_gnn` is false.
    """
    upstream = _UPSTREAM[stage]
    if stage == "train-xgb" and not config.xgb.use_gnn:
        return tuple(name for name in upstream if name != "gnn")
    return upstream


def manifest_path(config: Config, stage: str) -> Path:
    """Return the manifest path of a stage."""
    return stage_path(config, stage, MANIFEST_NAME)


def manifest_digest(config: Config, stage: str) -> str | None:
    """Return the digest of a stage manifest, or None if it does not exist."""
    path = manifest_path(config, stage)
    return file_digest(path) if path.exists() else None


def read_manifest(config: Config, stage: str) -> dict[str, Any] | None:
    """Return the parsed manifest of a stage, or None if it does not exist."""
    path = manifest_path(config, stage)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _expected(config: Config, stage: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "version": __version__,
        "config_hash": config_hash(config, _SECTIONS[stage]),
        "upstream": {
            name: manifest_digest(config, name)
            for name in upstream_stages(config, stage)
        },
    }


def write_manifest(config: Config, stage: str, outputs: Sequence[str]) -> Path:
    """Write the manifest of a stage after its outputs exist.

    Args:
        config: The run config.
        stage: Stage name, one of `PIPELINE_STAGES`.
        outputs: File names written to the stage directory.

    Returns:
        The manifest path.
    """
    manifest = {**_expected(config, stage), "outputs": sorted(outputs)}
    path = manifest_path(config, stage)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def is_fresh(config: Config, stage: str) -> bool:
    """Return true if the stage manifest matches the current config and inputs."""
    manifest = read_manifest(config, stage)
    if manifest is None:
        return False
    expected = _expected(config, stage)
    if any(manifest.get(key) != value for key, value in expected.items()):
        return False
    directory = stage_dir(config, stage)
    return all((directory / name).exists() for name in manifest["outputs"])
