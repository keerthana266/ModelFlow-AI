from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = (
        "postgresql+psycopg2://postgres:postgres@localhost:5432/automl_forge"
    )

    REDIS_BROKER_URL: str = "redis://localhost:6379/0"
    REDIS_BACKEND_URL: str = "redis://localhost:6379/1"

    CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"
    MAX_UPLOAD_MB: int = 50

    MLFLOW_TRACKING_URI: str = "http://127.0.0.1:5000"
    MLFLOW_EXPERIMENT_NAME: str = "AutoML Forge"

    ARTIFACT_STORE_BACKEND: str = "local"
    S3_ENDPOINT_URL: str = "http://127.0.0.1:9000"
    S3_ACCESS_KEY_ID: str = "minioadmin"
    S3_SECRET_ACCESS_KEY: str = "minioadmin"
    S3_BUCKET_NAME: str = "automl-forge"
    S3_REGION_NAME: str = "us-east-1"

    JWT_SECRET_KEY: str = "replace-this-with-a-long-random-secret"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440

    @property
    def cors_origins_list(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.CORS_ORIGINS.split(",")
            if origin.strip()
        ]

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
