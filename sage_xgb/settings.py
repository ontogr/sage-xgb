"""Credentials and service URIs. Run settings live in the run config."""

from __future__ import annotations

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class KaggleSettings(BaseSettings):
    """Kaggle credentials read from `KAGGLE_*` variables or the repo `.env`."""

    model_config = SettingsConfigDict(
        env_prefix="KAGGLE_", env_file=".env", extra="ignore", env_ignore_empty=True
    )

    username: str | None = None
    key: SecretStr | None = None


class MlflowSettings(BaseSettings):
    """MLflow settings read from `MLFLOW_*` variables or the repo `.env`."""

    model_config = SettingsConfigDict(
        env_prefix="MLFLOW_", env_file=".env", extra="ignore", env_ignore_empty=True
    )

    tracking_uri: str | None = None
