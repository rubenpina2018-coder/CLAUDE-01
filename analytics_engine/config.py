"""Configuración de rutas y de conexión a PostgreSQL.

Los parámetros se leen de variables de entorno (opcionalmente desde un fichero
``.env`` en la raíz del proyecto). Se usa un prefijo propio (``DW_``) para no
interferir con las variables ``PG*`` que el usuario pueda tener definidas.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from sqlalchemy import URL, create_engine
from sqlalchemy.engine import Engine

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SQL_DIR = PROJECT_ROOT / "sql"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
REJECTED_DIR = DATA_DIR / "rejected"


def load_dotenv(path: Path | None = None) -> None:
    """Carga pares ``CLAVE=valor`` de un ``.env`` sin pisar variables ya definidas.

    Implementación mínima para no añadir dependencias: ignora líneas vacías y
    comentarios, y retira comillas envolventes del valor.
    """
    path = path or PROJECT_ROOT / ".env"
    if not path.is_file():
        return
    # utf-8-sig: tolera el BOM que añaden algunos editores de Windows al guardar en UTF-8
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


@dataclass(frozen=True)
class DbSettings:
    host: str = "localhost"
    port: int = 5432
    database: str = "analytics_dw"
    user: str = "etl_user"
    password: str | None = None
    sslmode: str = "prefer"

    @classmethod
    def from_env(cls) -> DbSettings:
        """Parámetros del usuario ETL (propietario del modelo)."""
        load_dotenv()
        return cls(
            host=os.getenv("DW_HOST", cls.host),
            port=int(os.getenv("DW_PORT", str(cls.port))),
            database=os.getenv("DW_DATABASE", cls.database),
            user=os.getenv("DW_USER", cls.user),
            password=os.getenv("DW_PASSWORD") or None,
            sslmode=os.getenv("DW_SSLMODE", cls.sslmode),
        )

    @classmethod
    def admin_from_env(cls) -> DbSettings:
        """Parámetros del superusuario, usados solo por ``setup_database.py``."""
        load_dotenv()
        dw = cls.from_env()
        return cls(
            host=os.getenv("PG_ADMIN_HOST", dw.host),
            port=int(os.getenv("PG_ADMIN_PORT", str(dw.port))),
            database=os.getenv("PG_ADMIN_DATABASE", "postgres"),
            user=os.getenv("PG_ADMIN_USER", "postgres"),
            password=os.getenv("PG_ADMIN_PASSWORD") or None,
            sslmode=dw.sslmode,
        )

    def with_database(self, database: str) -> DbSettings:
        return replace(self, database=database)

    def url(self) -> URL:
        # Con host de tipo socket Unix (/var/run/postgresql) libpq ignora sslmode.
        return URL.create(
            "postgresql+psycopg2",
            username=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
            database=self.database,
            query={"sslmode": self.sslmode, "application_name": "analytics_engine"},
        )

    def create_engine(self, **kwargs) -> Engine:
        return create_engine(self.url(), pool_pre_ping=True, **kwargs)

    def describe(self) -> str:
        """Representación segura (sin contraseña) para logs."""
        return f"postgresql://{self.user}@{self.host}:{self.port}/{self.database}"
