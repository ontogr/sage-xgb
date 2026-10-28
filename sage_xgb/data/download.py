"""Download the raw AMLworld files for a Variant from Kaggle.

This is the only module that imports `kagglehub`, which is in the `kaggle`
extra. Each file is fetched on its own, so the rest of the Kaggle dataset is
not downloaded.
"""

from __future__ import annotations

import importlib
import os
import shutil

from sage_xgb.config import Config
from sage_xgb.settings import KaggleSettings


_FILE_SUFFIXES: tuple[str, ...] = ("_Trans.csv", "_accounts.csv", "_Patterns.txt")


def run(config: Config) -> dict[str, float]:
    """Copy the Variant's raw files from Kaggle into `config.data.raw_dir`.

    Credentials come from `KAGGLE_USERNAME` and `KAGGLE_KEY`, or from the repo
    `.env` file. A credential that is not set is left to kagglehub, which can
    also read its own `kaggle.json` file.

    Args:
        config: The run config. Reads `data.name`, `data.kaggle_dataset` and
            `data.raw_dir`.

    Returns:
        An empty dict, because this stage logs no metrics.

    Raises:
        ImportError: `kagglehub` is not installed. Install the `kaggle` extra
            with `uv sync --extra kaggle`.
    """
    try:
        kagglehub = importlib.import_module("kagglehub")
    except ImportError as error:
        msg = "kagglehub is not installed; run `uv sync --extra kaggle`"
        raise ImportError(msg) from error

    settings = KaggleSettings()
    if settings.username is not None:
        os.environ["KAGGLE_USERNAME"] = settings.username
    if settings.key is not None:
        os.environ["KAGGLE_KEY"] = settings.key.get_secret_value()

    raw_dir = config.data.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    for suffix in _FILE_SUFFIXES:
        file_name = f"{config.data.name}{suffix}"
        downloaded = kagglehub.dataset_download(
            config.data.kaggle_dataset, path=file_name
        )
        shutil.copyfile(downloaded, raw_dir / file_name)
    return {}
