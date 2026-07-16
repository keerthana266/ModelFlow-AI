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

    MODEL_ZOO_ALLOWED_MODULES: str = "sklearn"
    MODEL_ZOO_MAX_MODELS_PER_PROJECT: int = 50

    # Per-model Docker deployment runtime
    DOCKER_RUNTIME_IMAGE: str = "automl-forge-runtime:1.0"
    DOCKER_DEPLOYMENT_HOST: str = "127.0.0.1"
    DOCKER_DEPLOYMENT_BIND_HOST: str = "127.0.0.1"
    DOCKER_PORT_START: int = 8101
    DOCKER_PORT_END: int = 8999
    DOCKER_CONTAINER_PORT: int = 8000
    DOCKER_HEALTH_TIMEOUT_SECONDS: int = 60
    DOCKER_PREDICTION_TIMEOUT_SECONDS: int = 30
    DOCKER_MEMORY_LIMIT: str = "1g"
    DOCKER_NANO_CPUS: int = 1_000_000_000
    DOCKER_PIDS_LIMIT: int = 128

    @property
    def cors_origins_list(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.CORS_ORIGINS.split(",")
            if origin.strip()
        ]

    @property
    def model_zoo_allowed_modules_list(self) -> list[str]:
        return [
            module.strip()
            for module in self.MODEL_ZOO_ALLOWED_MODULES.split(",")
            if module.strip()
        ]

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
