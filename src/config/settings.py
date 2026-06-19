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

    # Inference backend
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"

    # MLflow
    mlflow_tracking_uri: str = str(PROJECT_ROOT / "mlflow_runs")
    mlflow_experiment_name: str = "nhs_omop_agent"

settings = Settings()