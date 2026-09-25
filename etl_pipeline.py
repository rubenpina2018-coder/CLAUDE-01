"""Hito 3 — Pipeline ETL concurrente: Parquet -> Polars -> PostgreSQL.

Fases:
  1. EXTRACT   Lectura del Parquet con Polars (decodificación multihilo).
  2. TRANSFORM Normalización, limpieza de nulos e inválidos, deduplicación y
               cálculo de `monto_neto` (impuesto según el tipo, aritmética exacta
               en centavos con redondeo HALF-UP). Todo vectorizado en Polars.
  3. LOAD      Patrón write-audit-publish:
               a) COPY en paralelo (N conexiones asyncpg) a una tabla de carga
                  sin índices, serializando cada lote a CSV en hilos (Polars
                  libera el GIL, así serialización y envío se solapan);
               b) índices construidos en paralelo y VACUUM ANALYZE;
               c) auditoría: los totales en PostgreSQL deben coincidir
                  exactamente con los calculados por Polars;
               d) swap atómico con la tabla productiva: la API nunca ve datos a
                  medio cargar y una carga fallida no toca los datos vigentes.

Uso:
    python etl_pipeline.py
    python etl_pipeline.py --conexiones 8 --filas-por-lote 25000
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path

import asyncpg
import polars as pl

import config
import db_schema
from db_schema import COLUMNAS, TABLA, VISTA_RESUMEN

log = logging.getLogger("etl")

MONEDA = pl.Decimal(14, 2)
TABLA_CARGA = f"{TABLA}_carga"
VISTA_CARGA = f"{VISTA_RESUMEN}_carga"

# Impuesto aplicado sobre el monto bruto según el tipo de transacción, en puntos
# básicos (1 pb = 0,01 %):  monto_neto = monto - redondeo_half_up(monto * tasa).
TASAS_IMPUESTO_PB: dict[str, int] = {
    "COMPRA": 1_600,  # 16 % IVA
    "PAGO_SERVICIO": 500,  # 5 % tasa sobre servicios
    "RETIRO": 150,  # 1,5 % comisión/retención por retiro
    "TRANSFERENCIA": 40,  # 0,4 % gravamen a movimientos financieros (4x1000)
    "DEPOSITO": 0,  # exento
}
assert set(TASAS_IMPUESTO_PB) == set(db_schema.TIPOS_TRANSACCION)

# Un estado ausente se asume PENDIENTE: criterio conservador, nunca se da por
# completada una transacción sin confirmación del sistema origen.
ESTADO_POR_DEFECTO = "PENDIENTE"


# --------------------------------------------------------------------------- #
# TRANSFORM (funciones puras, testeables sin base de datos)
# --------------------------------------------------------------------------- #
def _motivo_rechazo() -> pl.Expr:
    """Primer motivo por el que una fila no es cargable (null si es válida).

    Se deduplica por clave de negocio antes de validar: solo la primera
    aparición de cada `id` compite por ser cargada.
    """
    col = pl.col
    return (
        pl.when(col("id").is_null()).then(pl.lit("id_nulo"))
        .when(~col("id").is_first_distinct()).then(pl.lit("duplicado"))
        .when(col("usuario_id").is_null()).then(pl.lit("usuario_nulo"))
        .when(col("fecha").is_null()).then(pl.lit("fecha_nula"))
        .when(col("monto").is_null()).then(pl.lit("monto_nulo"))
        .when(col("monto") <= 0).then(pl.lit("monto_no_positivo"))
        .when(col("tipo_transaccion").is_null()).then(pl.lit("tipo_nulo"))
        .when(~col("tipo_transaccion").is_in(db_schema.TIPOS_TRANSACCION))
        .then(pl.lit("tipo_desconocido"))
        .when(~col("estado").is_in(db_schema.ESTADOS)).then(pl.lit("estado_desconocido"))
        .otherwise(None)
    )


def monto_neto_expr() -> pl.Expr:
    """monto_neto exacto: aritmética entera en centavos con redondeo HALF-UP.

    Se evita a propósito el float (errores de representación) y el Decimal de
    Polars, cuya multiplicación redondea half-to-even (0,125 -> 0,12).
    """
    cien = pl.lit(100, dtype=MONEDA)
    centavos = (pl.col("monto") * cien).cast(pl.Int64)
    tasa_pb = pl.col("tipo_transaccion").replace_strict(TASAS_IMPUESTO_PB, return_dtype=pl.Int64)
    impuesto = (centavos * tasa_pb + 5_000) // 10_000  # HALF-UP (montos > 0)
    return ((centavos - impuesto).cast(MONEDA) / cien).cast(MONEDA)


def transformar(raw: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Devuelve (válidas listas para cargar, rechazadas con su motivo)."""
    evaluadas = (
        raw.lazy()
        .with_columns(
            pl.col("tipo_transaccion").str.strip_chars().str.to_uppercase(),
            pl.col("estado").str.strip_chars().str.to_uppercase().fill_null(ESTADO_POR_DEFECTO),
        )
        .with_columns(_motivo_rechazo().alias("motivo_rechazo"))
        .collect()
    )
    validas = (
        evaluadas.lazy()
        .filter(pl.col("motivo_rechazo").is_null())
        .with_columns(monto_neto_expr().alias("monto_neto"))
        # Orden físico por (usuario, fecha): el historial de un usuario queda en
        # páginas contiguas del heap y la API lo lee con muy pocos accesos.
        .sort("usuario_id", "fecha")
        .select(COLUMNAS)
        .collect()
    )
    rechazadas = evaluadas.filter(pl.col("motivo_rechazo").is_not_null())
    return validas, rechazadas


# --------------------------------------------------------------------------- #
# LOAD
# --------------------------------------------------------------------------- #
def _a_csv(lote: pl.DataFrame) -> memoryview:
    buffer = io.BytesIO()
    lote.write_csv(buffer, include_header=False)
    # memoryview y no bytes: asyncpg prueba primero os.fspath(source), que
    # acepta bytes como *ruta de fichero*. getbuffer() además evita una copia.
    return buffer.getbuffer()


async def copiar_en_paralelo(
    pool: asyncpg.Pool, df: pl.DataFrame, conexiones: int, filas_por_lote: int
) -> dict[str, int | float]:
    """Productor/consumidor: N trabajadores, cada uno con su conexión y su COPY.

    La serialización CSV corre en hilos (`asyncio.to_thread`) y Polars libera el
    GIL, así que mientras un trabajador serializa, los demás están enviando.
    """
    cola: asyncio.Queue[pl.DataFrame] = asyncio.Queue()
    for lote in df.iter_slices(filas_por_lote):
        cola.put_nowait(lote)
    n_lotes = cola.qsize()
    bytes_enviados = 0

    async def trabajador(n: int) -> None:
        nonlocal bytes_enviados
        filas = lotes = 0
        t0 = time.perf_counter()
        async with pool.acquire() as conn:
            while not cola.empty():
                lote = cola.get_nowait()
                payload = await asyncio.to_thread(_a_csv, lote)
                await conn.copy_to_table(
                    TABLA_CARGA, source=payload, columns=list(COLUMNAS), format="csv"
                )
                filas += lote.height
                lotes += 1
                bytes_enviados += len(payload)
        log.info(
            "  trabajador %d: %2d lotes, %8s filas en %.2fs",
            n, lotes, f"{filas:,}", time.perf_counter() - t0,
        )

    # TaskGroup cancela el resto de trabajadores si uno falla.
    async with asyncio.TaskGroup() as tg:
        for n in range(conexiones):
            tg.create_task(trabajador(n))
    return {"lotes": n_lotes, "mb_csv": bytes_enviados / 1e6}


async def auditar(conn: asyncpg.Connection, validas: pl.DataFrame) -> dict[str, str]:
    """Write-audit-publish: compara totales exactos Polars vs. PostgreSQL."""
    esperado = validas.select(
        pl.len().alias("filas"),
        pl.col("monto").sum().alias("monto"),
        pl.col("monto_neto").sum().alias("monto_neto"),
    ).row(0, named=True)
    real = await conn.fetchrow(
        f"""SELECT total_transacciones AS filas, monto_total AS monto,
                   monto_neto_total AS monto_neto
            FROM {VISTA_CARGA} WHERE tipo_transaccion IS NULL"""
    )
    for clave in ("filas", "monto", "monto_neto"):
        if Decimal(real[clave]) != Decimal(esperado[clave]):
            raise RuntimeError(
                f"Auditoría fallida en '{clave}': Polars={esperado[clave]} "
                f"PostgreSQL={real[clave]}. No se publica la carga."
            )
    return {k: str(v) for k, v in esperado.items()}


async def publicar(conn: asyncpg.Connection) -> None:
    """Swap atómico: en una sola transacción la tabla de carga pasa a productiva."""
    async with conn.transaction():
        # Si la API retiene locks, se aborta limpio en vez de encolar lecturas.
        await conn.execute("SET LOCAL lock_timeout = '10s'")
        await conn.execute(f"DROP MATERIALIZED VIEW IF EXISTS {VISTA_RESUMEN}")
        await conn.execute(f"DROP TABLE IF EXISTS {TABLA}")
        await conn.execute(f"ALTER TABLE {TABLA_CARGA} RENAME TO {TABLA}")
        for sufijo in db_schema.SUFIJOS_INDICES:
            await conn.execute(f"ALTER INDEX {TABLA_CARGA}_{sufijo} RENAME TO {TABLA}_{sufijo}")
        await conn.execute(f"ALTER MATERIALIZED VIEW {VISTA_CARGA} RENAME TO {VISTA_RESUMEN}")


@contextmanager
def fase(nombre: str, tiempos: dict[str, float]) -> Iterator[None]:
    t0 = time.perf_counter()
    yield
    tiempos[nombre] = time.perf_counter() - t0
    log.info("%-40s %7.3f s", nombre, tiempos[nombre])


async def cargar(
    validas: pl.DataFrame, dsn: str, conexiones: int, filas_por_lote: int, tiempos: dict
) -> dict:
    pool = await asyncpg.create_pool(
        dsn, min_size=conexiones, max_size=conexiones, command_timeout=600
    )
    try:
        with fase("load: preparar tabla de carga", tiempos):
            await pool.execute(
                f"DROP MATERIALIZED VIEW IF EXISTS {VISTA_CARGA};"
                f"DROP TABLE IF EXISTS {TABLA_CARGA};"
                + db_schema.crear_tabla_sql(TABLA_CARGA, con_pk=False)
            )

        with fase(f"load: COPY paralelo ({conexiones} conexiones)", tiempos):
            copia = await copiar_en_paralelo(pool, validas, conexiones, filas_por_lote)

        with fase("load: índices en paralelo + PK", tiempos):
            async with asyncio.TaskGroup() as tg:
                tg.create_task(pool.execute(db_schema.indice_pk_sql(TABLA_CARGA)))
                tg.create_task(pool.execute(db_schema.indice_usuario_sql(TABLA_CARGA)))
            await pool.execute(db_schema.promover_pk_sql(TABLA_CARGA))

        with fase("load: VACUUM ANALYZE", tiempos):
            await pool.execute(f"VACUUM (ANALYZE) {TABLA_CARGA}")

        with fase("load: vista de resumen + auditoría", tiempos):
            await pool.execute(db_schema.crear_vista_resumen_sql(VISTA_CARGA, TABLA_CARGA))
            async with pool.acquire() as conn:
                auditoria = await auditar(conn, validas)

        with fase("load: swap atómico (publicación)", tiempos):
            async with pool.acquire() as conn:
                await publicar(conn)
    finally:
        await pool.close()
    return {**copia, "auditoria": auditoria}


# --------------------------------------------------------------------------- #
# Orquestación
# --------------------------------------------------------------------------- #
def ejecutar(parquet: Path, dsn: str, conexiones: int, filas_por_lote: int) -> dict:
    tiempos: dict[str, float] = {}
    t_total = time.perf_counter()

    with fase("extract: lectura Parquet", tiempos):
        raw = pl.read_parquet(parquet)

    with fase("transform: limpieza + dedupe + monto_neto", tiempos):
        validas, rechazadas = transformar(raw)

    config.RECHAZADOS_PATH.parent.mkdir(parents=True, exist_ok=True)
    rechazadas.write_parquet(config.RECHAZADOS_PATH, compression="zstd")

    # Salvaguarda: una carga completa vacía (origen vacío o corrupto) dejaría a
    # la API sin datos. Se aborta antes de tocar la base de datos.
    if validas.is_empty():
        raise RuntimeError(
            f"Ninguna fila válida de {raw.height:,} leídas: no se publica una tabla vacía."
        )

    carga = asyncio.run(cargar(validas, dsn, conexiones, filas_por_lote, tiempos))
    total = time.perf_counter() - t_total
    t_copy = next(v for k, v in tiempos.items() if k.startswith("load: COPY"))

    motivos = rechazadas.group_by("motivo_rechazo").len().sort("len", descending=True)
    metricas = {
        "filas_leidas": raw.height,
        "filas_cargadas": validas.height,
        "filas_rechazadas": rechazadas.height,
        "rechazos_por_motivo": dict(motivos.iter_rows()),
        "conexiones": conexiones,
        "filas_por_lote": filas_por_lote,
        "lotes": carga["lotes"],
        "mb_csv_enviados": round(carga["mb_csv"], 1),
        "copy_filas_por_segundo": round(validas.height / t_copy),
        "tiempos_s": {k: round(v, 3) for k, v in tiempos.items()},
        "total_s": round(total, 3),
        "total_filas_por_segundo": round(raw.height / total),
        "auditoria": carga["auditoria"],
    }
    imprimir_resumen(metricas)
    return metricas


def imprimir_resumen(m: dict) -> None:
    linea = "=" * 62
    print(f"\n{linea}\n RESUMEN DEL PIPELINE ETL\n{linea}")
    print(f" Filas leídas del Parquet     {m['filas_leidas']:>12,}")
    pct = 100 * m["filas_rechazadas"] / m["filas_leidas"]
    print(f" Filas rechazadas             {m['filas_rechazadas']:>12,}  ({pct:.2f} %)")
    for motivo, n in m["rechazos_por_motivo"].items():
        print(f"   - {motivo:<24} {n:>12,}")
    print(f" Filas cargadas en PostgreSQL {m['filas_cargadas']:>12,}")
    print(f" Auditoría (Polars == PG)     OK  monto={m['auditoria']['monto']}"
          f"  neto={m['auditoria']['monto_neto']}")
    print(linea)
    for nombre, segundos in m["tiempos_s"].items():
        print(f" {nombre:<42} {segundos:>8.3f} s")
    print(linea)
    print(f" COPY: {m['lotes']} lotes, {m['mb_csv_enviados']} MB CSV, "
          f"{m['copy_filas_por_segundo']:,} filas/s")
    print(f" TIEMPO TOTAL ETL {m['total_s']:>35.3f} s  ({m['total_filas_por_segundo']:,} filas/s)")
    print(linea)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parquet", type=Path, default=config.PARQUET_PATH)
    parser.add_argument("--dsn", default=config.DATABASE_URL)
    parser.add_argument("--conexiones", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--filas-por-lote", type=int, default=50_000)
    parser.add_argument(
        "--metricas", type=Path, default=config.RESULTS_DIR / "etl_metrics.json",
        help="ruta del JSON con las métricas de la ejecución",
    )
    args = parser.parse_args()

    log.info("Polars %s con %d hilos", pl.__version__, pl.thread_pool_size())
    metricas = ejecutar(args.parquet, args.dsn, args.conexiones, args.filas_por_lote)
    args.metricas.parent.mkdir(parents=True, exist_ok=True)
    args.metricas.write_text(json.dumps(metricas, indent=2, ensure_ascii=False))
    log.info("Métricas guardadas en %s", args.metricas)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
