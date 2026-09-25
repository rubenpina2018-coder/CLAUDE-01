"""Carga en PostgreSQL: COPY a staging y merge set-based al Star Schema.

Toda la carga ocurre en UNA transacción: o se aplica completa o no se aplica
nada, y los lectores (Power BI, Looker Studio) siguen viendo la versión anterior
hasta el COMMIT gracias a MVCC. Un advisory lock impide que dos ejecuciones
concurrentes del ETL se intercalen.

Estrategia de sincronización (cada hoja es un snapshot completo del negocio):

* Merge set-based por clave natural: ``UPDATE ... FROM`` solo de las filas cuyo
  contenido cambió (``IS DISTINCT FROM``) + ``INSERT ... WHERE NOT EXISTS`` de
  las nuevas. Ambas sentencias se resuelven con hash joins; a diferencia de
  ``INSERT ... ON CONFLICT`` no hay inserción especulativa fila a fila ni se
  bloquean/reescriben filas sin cambios, así que re-ejecutar el pipeline con
  los mismos datos no escribe nada.
* Dimensiones: SCD tipo 1 con claves subrogadas estables entre ejecuciones.
* Hechos: además se eliminan las líneas que ya no existen en la hoja. Un
  "disyuntor" aborta la carga si fuera a borrar más de un porcentaje razonable
  de la tabla (protege frente a exports truncados o vacíos).
* Modo bulk adaptativo: PostgreSQL valida cada FK con una consulta por fila
  insertada (4 FK x N filas), lo que triplica el coste de una carga masiva.
  Si hay muchas filas nuevas, dentro de la misma transacción se retiran las FK
  de ``fact_sales`` y se recrean tras insertar: al recrearlas se validan con una
  única consulta set-based. Las definiciones se leen del catálogo (no pueden
  divergir de ``init.sql``) y, si algo falla, el ROLLBACK las restaura.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path

import polars as pl
from sqlalchemy import text
from sqlalchemy.engine import Engine

from analytics_engine.db import copy_dataframe, run_sql_script
from analytics_engine.transform import CUSTOMER_OUTPUT, PRODUCT_OUTPUT, SALES_OUTPUT, SheetResult
from analytics_engine.validate import Check, ValidationFailed

log = logging.getLogger(__name__)


class MassDeleteError(RuntimeError):
    """La carga eliminaría demasiadas filas de hechos (posible export incompleto)."""


DATE_COLUMNS = [
    "date_key", "full_date", "year", "quarter", "quarter_name", "year_quarter", "month", "month_name",
    "month_short", "year_month", "year_month_key", "year_month_label", "month_start", "month_end",
    "iso_year", "iso_week", "day_of_month", "day_of_year", "day_of_week", "day_name", "day_short",
    "is_weekend", "is_holiday", "holiday_name",
]
PRODUCT_ATTRS = [c for c in PRODUCT_OUTPUT if c not in ("sku", "source_row")]
CUSTOMER_ATTRS = [c for c in CUSTOMER_OUTPUT if c not in ("customer_id", "source_row")]
FACT_ATTRS = [
    "date_key", "customer_key", "product_key", "channel_key", "order_status", "payment_method", "quantity",
    "unit_price", "unit_cost", "discount_pct", "gross_amount", "discount_amount", "net_amount",
    "cost_amount", "profit_amount",
]

# Serializa las cargas: una segunda ejecución espera a que termine la primera.
ADVISORY_LOCK = "SELECT pg_advisory_xact_lock(hashtext('analytics_engine.load_star_schema'))"
# A partir de este nº de líneas nuevas compensa validar las FK en bloque (ver docstring).
BULK_MODE_MIN_NEW_ROWS = 10_000
FACT_FOREIGN_KEYS = """
    SELECT conname, pg_get_constraintdef(oid) FROM pg_constraint
    WHERE conrelid = 'dw.fact_sales'::regclass AND contype = 'f' ORDER BY conname
"""
NEW_FACTS = """
    SELECT count(*) FROM tmp_fact_source s
    WHERE NOT EXISTS (SELECT 1 FROM dw.fact_sales f WHERE f.order_id = s.order_id AND f.order_line = s.order_line)
"""

# Líneas de venta con sus claves subrogadas resueltas (tabla temporal, se borra al COMMIT).
BUILD_FACT_SOURCE = """
    CREATE TEMP TABLE tmp_fact_source ON COMMIT DROP AS
    SELECT s.order_id, s.order_line, s.date_key,
           COALESCE(c.customer_key, -1) AS customer_key,
           p.product_key,
           COALESCE(ch.channel_key, -1)::SMALLINT AS channel_key,
           s.order_status, s.payment_method, s.quantity, s.unit_price, s.unit_cost, s.discount_pct,
           s.gross_amount, s.discount_amount, s.net_amount, s.cost_amount, s.profit_amount
      FROM staging.stg_sales s
      JOIN dw.dim_product p ON p.sku = s.sku
      LEFT JOIN dw.dim_customer c ON c.customer_id = s.customer_id
      LEFT JOIN dw.dim_channel ch ON ch.channel_code = s.channel_code
"""
STALE_FACTS = """
    FROM dw.fact_sales f
    WHERE NOT EXISTS (SELECT 1 FROM staging.stg_sales s
                      WHERE s.order_id = f.order_id AND s.order_line = f.order_line)
"""
# Fecha de primera compra: atributo de ciclo de vida usado por "Nuevos Clientes" (DAX) y las cohortes.
UPDATE_FIRST_ORDER = """
    WITH first_orders AS (
        SELECT f.customer_key, min(d.full_date) AS first_order_date
        FROM dw.fact_sales f JOIN dw.dim_date d ON d.date_key = f.date_key
        WHERE f.customer_key <> -1
        GROUP BY f.customer_key
    )
    UPDATE dw.dim_customer c
       SET first_order_date = fo.first_order_date, updated_at = now()
      FROM dw.dim_customer c2 LEFT JOIN first_orders fo ON fo.customer_key = c2.customer_key
     WHERE c.customer_key = c2.customer_key
       AND c.customer_key <> -1
       AND c.first_order_date IS DISTINCT FROM fo.first_order_date
"""


def merge_statements(target: str, source: str, key: list[str], attrs: list[str],
                     audited: bool) -> tuple[str, str]:
    """(UPDATE de filas cambiadas, INSERT de filas nuevas) para sincronizar ``target`` con ``source``."""
    match = " AND ".join(f"t.{k} = s.{k}" for k in key)
    assignments = [f"{c} = s.{c}" for c in attrs]
    if audited:
        assignments += ["etl_run_id = %(run_id)s", "updated_at = now()"]
    changed = (f"({', '.join('t.' + c for c in attrs)}) IS DISTINCT FROM "
               f"({', '.join('s.' + c for c in attrs)})")
    update = (f"UPDATE {target} AS t SET {', '.join(assignments)} "
              f"FROM {source} AS s WHERE {match} AND {changed}")
    columns = key + attrs
    insert_columns = columns + (["etl_run_id"] if audited else [])
    values = [f"s.{c}" for c in columns] + (["%(run_id)s"] if audited else [])
    insert = (f"INSERT INTO {target} ({', '.join(insert_columns)}) "
              f"SELECT {', '.join(values)} FROM {source} AS s "
              f"WHERE NOT EXISTS (SELECT 1 FROM {target} AS t WHERE {match})")
    return update, insert


MERGES = {
    "dim_date": merge_statements("dw.dim_date", "staging.stg_date", ["date_key"], DATE_COLUMNS[1:], audited=False),
    "dim_product": merge_statements("dw.dim_product", "staging.stg_product", ["sku"], PRODUCT_ATTRS, audited=True),
    "dim_customer": merge_statements("dw.dim_customer", "staging.stg_customer", ["customer_id"], CUSTOMER_ATTRS,
                                     audited=True),
    "fact_sales": merge_statements("dw.fact_sales", "tmp_fact_source", ["order_id", "order_line"], FACT_ATTRS,
                                   audited=True),
}


class _TimedCursor:
    """Envoltorio del cursor que registra (nivel DEBUG) la duración de cada sentencia."""

    def __init__(self, cursor):
        self._cursor = cursor

    def __getattr__(self, name):
        return getattr(self._cursor, name)

    def execute(self, sql, params=None):
        start = time.perf_counter()
        self._cursor.execute(sql, params)
        log.debug("    %.3f s  %s", time.perf_counter() - start, " ".join(sql.split())[:90])

    def copy(self, df: pl.DataFrame, table: str) -> int:
        start = time.perf_counter()
        rows = copy_dataframe(self._cursor, df, table)
        log.debug("    %.3f s  COPY %s (%s filas)", time.perf_counter() - start, table, f"{rows:,}")
        return rows


def start_run(engine: Engine, source: str) -> int:
    with engine.begin() as conn:
        return conn.execute(text("INSERT INTO audit.etl_run (source) VALUES (:source) RETURNING run_id"),
                            {"source": source}).scalar_one()


def finish_run(engine: Engine, run_id: int, status: str, metrics: dict, error: str | None = None) -> None:
    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE audit.etl_run
               SET status = :status, finished_at = now(), metrics = CAST(:metrics AS JSONB),
                   rows_extracted = :extracted, rows_rejected = :rejected, rows_loaded = :loaded,
                   error_message = :error
             WHERE run_id = :run_id"""), {
            "status": status, "metrics": json.dumps(metrics, default=str), "run_id": run_id,
            "extracted": metrics.get("rows_extracted"), "rejected": metrics.get("rows_rejected"),
            # Solo hay filas cargadas si la transacción de carga llegó a confirmarse
            "loaded": metrics.get("rows_loaded") if "load" in metrics else 0, "error": error,
        })


def _rejected_frame(run_id: int, results: list[SheetResult]) -> pl.DataFrame:
    frames = [
        r.rejected.select(
            pl.lit(run_id, pl.Int64).alias("run_id"), pl.lit(r.sheet).alias("source_sheet"),
            "source_row", "reason", "record_key", "record")
        for r in results if r.rejected.height
    ]
    return pl.concat(frames) if frames else pl.DataFrame()


def _merge(cur: _TimedCursor, table: str, params: dict) -> dict:
    update, insert = MERGES[table]
    cur.execute(update, params)
    updated = cur.rowcount
    cur.execute(insert, params)
    return {"inserted": cur.rowcount, "updated": updated}


def load_star_schema(engine: Engine, run_id: int, products: SheetResult, customers: SheetResult,
                     sales: SheetResult, dates: pl.DataFrame, *, max_delete_ratio: float = 0.25,
                     pre_commit_checks: Callable[[Callable[[str], tuple]], list[Check]] | None = None) -> dict:
    """Carga staging y sincroniza el Star Schema en una única transacción. Devuelve estadísticas.

    ``pre_commit_checks`` recibe una función de consulta ligada a la transacción en curso; si
    alguna comprobación falla se lanza :class:`ValidationFailed` y se hace ROLLBACK (nada se publica).
    """
    stats: dict = {}
    params = {"run_id": run_id}
    raw = engine.raw_connection()
    try:
        with raw.cursor() as db_cursor:
            cur = _TimedCursor(db_cursor)
            cur.execute(ADVISORY_LOCK)
            cur.execute("SET LOCAL work_mem = '64MB'")
            cur.execute("TRUNCATE staging.stg_date, staging.stg_product, staging.stg_customer, staging.stg_sales")
            stats["staging_rows"] = {
                "stg_date": cur.copy(dates.select(DATE_COLUMNS), "staging.stg_date"),
                "stg_product": cur.copy(products.clean.select(PRODUCT_OUTPUT), "staging.stg_product"),
                "stg_customer": cur.copy(customers.clean.select(CUSTOMER_OUTPUT), "staging.stg_customer"),
                "stg_sales": cur.copy(sales.clean.select(SALES_OUTPUT), "staging.stg_sales"),
            }

            for table in ("dim_date", "dim_product", "dim_customer"):
                stats[table] = _merge(cur, table, params)

            cur.execute(BUILD_FACT_SOURCE)
            if cur.rowcount != stats["staging_rows"]["stg_sales"]:
                raise RuntimeError(f"{stats['staging_rows']['stg_sales'] - cur.rowcount} líneas de venta "
                                   "referencian SKUs ausentes en dim_product")

            # Disyuntor: un export truncado no debe vaciar la tabla de hechos.
            cur.execute("SELECT count(*) FROM dw.fact_sales")
            existing = cur.fetchone()[0]
            cur.execute("SELECT count(*) " + STALE_FACTS)
            stale = cur.fetchone()[0]
            if existing and stale / existing > max_delete_ratio:
                raise MassDeleteError(
                    f"La carga eliminaría {stale:,} de {existing:,} líneas de venta ({stale / existing:.1%}), "
                    f"por encima del límite del {max_delete_ratio:.0%}. Revisa el export o ejecuta con "
                    f"--max-delete-ratio 1 si el cambio es intencionado.")
            cur.execute("DELETE " + STALE_FACTS)
            deleted = cur.rowcount

            cur.execute(NEW_FACTS)
            bulk_mode = cur.fetchone()[0] >= BULK_MODE_MIN_NEW_ROWS
            foreign_keys = []
            if bulk_mode:
                cur.execute(FACT_FOREIGN_KEYS)
                foreign_keys = cur.fetchall()
                if foreign_keys:
                    cur.execute("ALTER TABLE dw.fact_sales "
                                + ", ".join(f"DROP CONSTRAINT {name}" for name, _ in foreign_keys))
            stats["fact_sales"] = {**_merge(cur, "fact_sales", params), "deleted": deleted, "bulk_mode": bulk_mode}
            if foreign_keys:  # recrear = validar todas las filas con una consulta set-based por FK
                cur.execute("ALTER TABLE dw.fact_sales "
                            + ", ".join(f"ADD CONSTRAINT {name} {definition}" for name, definition in foreign_keys))

            cur.execute(UPDATE_FIRST_ORDER)
            stats["dim_customer"]["first_order_date_updated"] = cur.rowcount

            rejected = _rejected_frame(run_id, [products, customers, sales])
            stats["rejected_rows"] = cur.copy(rejected, "audit.rejected_record") if rejected.height else 0
            if bulk_mode:  # tras cambios masivos, estadísticas frescas para el planificador (y las consultas BI)
                cur.execute("ANALYZE dw.dim_customer, dw.dim_product, dw.fact_sales")

            if pre_commit_checks:  # write-audit-publish: solo se publica si todo cuadra
                def query(sql: str) -> tuple:
                    cur.execute(sql)
                    return cur.fetchone()

                stats["checks"] = pre_commit_checks(query)
                if not all(c.passed for c in stats["checks"]):
                    raise ValidationFailed(stats["checks"])
        raw.commit()
    except Exception:
        raw.rollback()
        raise
    finally:
        raw.close()
    return stats


# ---------------------------------------------------------------------------
# Capa BI
# ---------------------------------------------------------------------------

def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def refresh_bi_layer(engine: Engine, bi_sql: Path, force_deploy: bool = False) -> dict:
    """Despliega sql/bi_views.sql si es necesario; si no, refresca las vistas materializadas.

    El hash del fichero desplegado se guarda en el comentario del esquema ``bi``:
    si el SQL cambia, el siguiente run lo redespliega automáticamente.
    """
    digest = _file_hash(bi_sql)
    with engine.connect() as conn:
        mviews = conn.execute(text(
            "SELECT matviewname FROM pg_matviews WHERE schemaname = 'bi' ORDER BY 1")).scalars().all()
        comment = conn.execute(text("SELECT obj_description('bi'::regnamespace, 'pg_namespace')")).scalar() or ""
    if force_deploy or not mviews or f"sha256:{digest}" not in comment:
        run_sql_script(engine, bi_sql)
        with engine.begin() as conn:
            conn.exec_driver_sql(
                f"COMMENT ON SCHEMA bi IS 'Capa de consumo para herramientas BI (bi_views.sql sha256:{digest})'")
        return {"action": "deployed", "sql_hash": digest}

    raw = engine.raw_connection()
    try:
        raw.autocommit = True  # cada REFRESH en su propia transacción
        with raw.cursor() as cur:
            for mv in mviews:
                # CONCURRENTLY: los dashboards pueden seguir consultando durante el refresco
                cur.execute(f"REFRESH MATERIALIZED VIEW CONCURRENTLY bi.{mv}")
    finally:
        raw.close()
    return {"action": "refreshed", "materialized_views": list(mviews)}
