"""Hito 1 — Inicializa el esquema de la base de datos.

Crea de forma idempotente la tabla `transacciones_financieras`, su índice por
usuario y la vista materializada de resumen que consume la API.

Uso:
    python init_db.py            # crea lo que falte (seguro de re-ejecutar)
    python init_db.py --reset    # elimina y recrea el esquema (destructivo)
"""

from __future__ import annotations

import argparse
import asyncio
import logging

import asyncpg

import config
import db_schema

log = logging.getLogger("init_db")


async def conectar_con_reintentos(
    dsn: str, intentos: int = 30, espera_s: float = 1.0
) -> asyncpg.Connection:
    """Conecta esperando a que PostgreSQL acepte conexiones (arranque del contenedor)."""
    for intento in range(1, intentos + 1):
        try:
            return await asyncpg.connect(dsn, timeout=5)
        except (TimeoutError, OSError, asyncpg.PostgresError) as exc:
            if intento == intentos:
                raise
            log.warning("PostgreSQL aún no disponible (%d/%d): %s", intento, intentos, exc)
            await asyncio.sleep(espera_s)
    raise AssertionError("inalcanzable")


async def describir_tabla(conn: asyncpg.Connection, tabla: str) -> None:
    columnas = await conn.fetch(
        """
        SELECT column_name, data_type, numeric_precision, numeric_scale, is_nullable
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = $1
        ORDER BY ordinal_position
        """,
        tabla,
    )
    log.info("Tabla %s:", tabla)
    for c in columnas:
        tipo = c["data_type"]
        if c["data_type"] == "numeric":
            tipo = f"numeric({c['numeric_precision']},{c['numeric_scale']})"
        nulo = "NULL" if c["is_nullable"] == "YES" else "NOT NULL"
        log.info("  %-17s %-26s %s", c["column_name"], tipo, nulo)

    for idx in await conn.fetch(
        "SELECT indexdef FROM pg_indexes WHERE tablename = $1 ORDER BY indexname", tabla
    ):
        log.info("  índice: %s", idx["indexdef"])


async def main(reset: bool) -> None:
    conn = await conectar_con_reintentos(config.DATABASE_URL)
    try:
        log.info("Conectado a %s", await conn.fetchval("SELECT version()"))
        async with conn.transaction():
            if reset:
                log.warning("--reset: eliminando esquema existente")
                await conn.execute(db_schema.eliminar_esquema_sql())
            await conn.execute(db_schema.crear_tabla_sql(db_schema.TABLA, con_pk=True))
            await conn.execute(db_schema.indice_usuario_sql(db_schema.TABLA))
            await conn.execute(
                db_schema.crear_vista_resumen_sql(db_schema.VISTA_RESUMEN, db_schema.TABLA)
            )
        await describir_tabla(conn, db_schema.TABLA)
        filas = await conn.fetchval(f"SELECT count(*) FROM {db_schema.TABLA}")
        log.info("Esquema listo. Filas actuales en %s: %s", db_schema.TABLA, f"{filas:,}")
    finally:
        await conn.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reset", action="store_true", help="elimina y recrea el esquema")
    asyncio.run(main(parser.parse_args().reset))
