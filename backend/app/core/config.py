# app/core/config.py

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # PostgreSQL database URL
    DATABASE_URL: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/automl_forge"

    # Redis URLs for Celery
    REDIS_BROKER_URL: str = "redis://localhost:6379/0"
    REDIS_BACKEND_URL: str = "redis://localhost:6379/1"

    # Frontend URLs allowed to call backend
    CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"

    # Maximum CSV upload size
    MAX_UPLOAD_MB: int = 50

    # Tells Pydantic to read values from .env file
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    @property
    def cors_origins_list(self) -> list[str]:
        """
        Converts:
        "http://localhost:5173,http://127.0.0.1:5173"

        Into:
        [
            "http://localhost:5173",
            "http://127.0.0.1:5173"
        ]
        """
        return [
            origin.strip()
            for origin in self.CORS_ORIGINS.split(",")
            if origin.strip()
        ]


settings = Settings()