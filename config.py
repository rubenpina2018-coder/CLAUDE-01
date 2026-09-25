"""Configuración compartida del proyecto, leída de variables de entorno.

Los valores por defecto coinciden con los de `docker-compose.yml`, de modo que
el proyecto funciona en local sin definir ninguna variable.
"""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

PARQUET_PATH = Path(os.getenv("PARQUET_PATH", str(DATA_DIR / "transacciones.parquet")))
RECHAZADOS_PATH = Path(os.getenv("RECHAZADOS_PATH", str(DATA_DIR / "rechazados.parquet")))


def _database_url_por_defecto() -> str:
    user = os.getenv("POSTGRES_USER", "etl_user")
    password = os.getenv("POSTGRES_PASSWORD", "etl_password")
    host = os.getenv("POSTGRES_HOST", "127.0.0.1")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "finanzas")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


DATABASE_URL = os.getenv("DATABASE_URL") or _database_url_por_defecto()
