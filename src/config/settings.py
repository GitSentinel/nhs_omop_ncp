import sys
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Project configuration imports
from config import (
    DUCKDB_PATH,
    AZURE_ENDPOINT,
    AZURE_DEPLOYMENT,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_AGENT,
)


class Settings(BaseSettings):
    # Environment file configuration
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database configuration
    duckdb_path: Path = DUCKDB_PATH
    person_id: int = 1

    # Azure OpenAI configuration
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = AZURE_ENDPOINT
    azure_openai_deployment: str = AZURE_DEPLOYMENT

    # MLflow configuration
    mlflow_tracking_uri: str = MLFLOW_TRACKING_URI
    mlflow_experiment_name: str = MLFLOW_EXPERIMENT_AGENT


# Shared settings instance
settings = Settings()