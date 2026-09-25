"""Utilidades de base de datos: ejecución de scripts SQL y carga masiva con COPY."""

from __future__ import annotations

import io
from pathlib import Path

import polars as pl
from sqlalchemy.engine import Engine


def run_sql_script(engine: Engine, path: Path) -> None:
    """Ejecuta un script SQL completo en una única transacción (todo o nada).

    Se usa el cursor DBAPI directamente y sin parámetros: ``sqlalchemy.text``
    interpretaría como parámetros de enlace los ``:identificador`` que aparecen
    dentro de literales (p. ej. ``'HH24:MI'``), y psycopg2 los ``%`` si se le
    pasaran parámetros.
    """
    sql = Path(path).read_text(encoding="utf-8")
    raw = engine.raw_connection()
    try:
        with raw.cursor() as cur:
            cur.execute(sql)
        raw.commit()
    except Exception:
        raw.rollback()
        raise
    finally:
        raw.close()


def copy_dataframe(cursor, df: pl.DataFrame, table: str) -> int:
    """Carga un DataFrame en ``table`` con ``COPY ... FROM STDIN`` (formato CSV).

    Es el método de ingesta más rápido de PostgreSQL: una única operación en
    streaming en lugar de miles de INSERT. Polars serializa a CSV en memoria
    (multihilo); los nulos viajan como campo vacío sin comillas (NULL en COPY)
    y las cadenas vacías como ``""``.
    """
    if df.is_empty():
        return 0
    buffer = io.BytesIO()
    df.write_csv(buffer, include_header=False, null_value="")
    buffer.seek(0)
    columns = ", ".join(df.columns)
    cursor.copy_expert(f"COPY {table} ({columns}) FROM STDIN WITH (FORMAT csv)", buffer)
    return df.height
