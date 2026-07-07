from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Data
    duckdb_path: Path = PROJECT_ROOT / "data" / "processed" / "omop_v54.duckdb"

    # Patient isolation (baked in at server startup)
    person_id: int = 1

    # Azure OpenAI
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = "https://openai-omop-dev-01.services.ai.azure.com/openai/v1"
    azure_openai_deployment: str = "gpt-5-nano"

    # MLflow
    mlflow_tracking_uri: str = "sqlite:///mlflow_runs/mlflow.db"
    mlflow_experiment_name: str = "nhs_omop_agent"

settings = Settings()