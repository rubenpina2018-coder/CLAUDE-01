"""Service configuration, read from ``FRAUD_API_*`` environment variables (or ``.env``)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FRAUD_API_", env_file=".env", extra="ignore")

    model_path: Path = Field(
        default=Path("outputs/model/model.gguf"), description="Optimized GGUF model to serve."
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    docs_enabled: bool = Field(default=True, description="Expose /docs and /openapi.json.")


@lru_cache
def get_settings() -> Settings:
    return Settings()
