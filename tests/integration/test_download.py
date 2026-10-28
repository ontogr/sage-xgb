"""Integration test for the Kaggle download stage.

The test downloads the three HI-Small files from Kaggle into a temporary
directory. It needs the `kaggle` extra and credentials. `KAGGLE_USERNAME` and
`KAGGLE_KEY` are read from the environment or the repo `.env` file, and the
test is skipped when either is missing.
"""

from pathlib import Path

import pytest

from sage_xgb.config import load_config
from sage_xgb.data import download
from sage_xgb.settings import KaggleSettings


HI_SMALL_CONFIG = Path(__file__).parents[2] / "configs" / "hi_small.yaml"
HI_SMALL_FILES = (
    "HI-Small_Trans.csv",
    "HI-Small_accounts.csv",
    "HI-Small_Patterns.txt",
)
_settings = KaggleSettings()
_HAS_CREDENTIALS = _settings.username is not None and _settings.key is not None


class TestDownload:
    """Download of the HI-Small files from Kaggle."""

    @pytest.mark.skipif(
        not _HAS_CREDENTIALS, reason="KAGGLE_USERNAME and KAGGLE_KEY are not set"
    )
    def test_writes_the_hi_small_files_to_raw_dir(self, tmp_path: Path) -> None:
        """The three HI-Small files are copied into data.raw_dir."""
        config = load_config(HI_SMALL_CONFIG, [f"data.raw_dir={tmp_path}"])
        download.run(config)
        for file_name in HI_SMALL_FILES:
            assert (tmp_path / file_name).is_file(), file_name
