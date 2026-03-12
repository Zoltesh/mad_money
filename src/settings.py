"""Settings module for mad-money application."""

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    coinbase_api_key: str | None = Field(default=None, alias="COINBASE_API_KEY")
    coinbase_private_key: str | None = Field(default=None, alias="COINBASE_PRIVATE_KEY")

    @field_validator("coinbase_api_key", "coinbase_private_key", mode="before")
    @classmethod
    def check_not_empty(cls, v: str | None) -> str | None:
        """Ensure credentials are not empty."""
        if v is None or v == "":
            raise ValueError("Setting is required but not found in .env")
        return v


settings = Settings()
