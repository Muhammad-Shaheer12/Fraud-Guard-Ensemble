from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = Field(default="Fraud Detection API", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")

    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")
    postgres_db: str = Field(default="Fraud_Detection_Database", alias="POSTGRES_DB")
    postgres_user: str = Field(default="postgres", alias="POSTGRES_USER")
    postgres_password: str = Field(default="postgres", alias="POSTGRES_PASSWORD")

    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")

    data_dir: Path = Field(default=Path("data"), alias="DATA_DIR")
    artifact_dir: Path = Field(default=Path("artifacts"), alias="ARTIFACT_DIR")
    model_dir: Path = Field(default=Path("artifacts/models"), alias="MODEL_DIR")

    mlflow_tracking_uri: str = Field(default="./artifacts/mlruns", alias="MLFLOW_TRACKING_URI")
    wandb_project: str = Field(default="fraud-detection-dnn", alias="WANDB_PROJECT")
    wandb_mode: str = Field(default="offline", alias="WANDB_MODE")

    @property
    def sqlalchemy_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
