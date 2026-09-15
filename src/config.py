"""Application configuration loaded from environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings shared by the pipeline and API."""

    project_name: str = "PulseOps"
    environment: str = "development"
    data_dir: Path = Path("data")
    model_dir: Path = Path("artifacts/models")
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    model_name: str = "pulseops-demand-model"

    model_config = SettingsConfigDict(env_prefix="PULSEOPS_", env_file=".env", extra="ignore")


settings = Settings()
