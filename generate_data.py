"""Hito 2 — Genera un dataset sintético masivo de transacciones financieras.

Produce N filas (1.000.000 por defecto) con distribuciones realistas y una
fracción controlada de "datos sucios" para que el ETL tenga trabajo real:
nulos, montos negativos, tipos con formato inconsistente y registros duplicados.

Todo es vectorizado: NumPy genera los números aleatorios (reproducibles por
semilla) y Polars ensambla, ensucia y escribe el Parquet usando todos los núcleos.

Uso:
    python generate_data.py
    python generate_data.py --filas 5000000 --usuarios 50000 --semilla 7
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

import config
import db_schema

log = logging.getLogger("generate_data")

MONEDA = pl.Decimal(14, 2)

TIPOS = db_schema.TIPOS_TRANSACCION
PROB_TIPOS = (0.40, 0.15, 0.15, 0.20, 0.10)
# Monto log-normal por tipo: (mu, sigma) del logaritmo; mediana = e^mu.
MONTO_LOGNORMAL = {
    "COMPRA": (3.6, 1.0),  # mediana ~37
    "DEPOSITO": (5.5, 1.1),  # mediana ~245
    "RETIRO": (4.6, 0.8),  # mediana ~100
    "TRANSFERENCIA": (5.2, 1.2),  # mediana ~181
    "PAGO_SERVICIO": (4.0, 0.7),  # mediana ~55
}
MONTO_MAX_CENTAVOS = 5_000_000  # 50.000,00

ESTADOS = db_schema.ESTADOS
PROB_ESTADOS = (0.92, 0.04, 0.03, 0.01)

FECHA_INICIO = datetime(2025, 1, 1, tzinfo=UTC)
FECHA_FIN = datetime(2026, 1, 1, tzinfo=UTC)

# Fracción de filas afectadas por cada tipo de defecto (independientes entre sí).
SUCIEDAD = {
    "monto_nulo": 0.005,
    "monto_negativo": 0.001,
    "tipo_nulo": 0.003,
    "tipo_mal_formato": 0.010,
    "estado_nulo": 0.010,
    "usuario_nulo": 0.002,
    "fecha_nula": 0.001,
    "duplicados": 0.001,
}


def generar(filas: int, usuarios: int, semilla: int) -> pl.DataFrame:
    rng = np.random.default_rng(semilla)

    # Fechas uniformes en el año; ordenadas para que el id crezca con el tiempo,
    # como ocurre con los identificadores secuenciales de un sistema real.
    inicio_us = int(FECHA_INICIO.timestamp() * 1_000_000)
    rango_us = int((FECHA_FIN - FECHA_INICIO).total_seconds() * 1_000_000)
    fechas_us = np.sort(inicio_us + rng.integers(0, rango_us, size=filas, dtype=np.int64))

    # Actividad heterogénea: unos pocos usuarios concentran muchas transacciones.
    pesos = rng.lognormal(mean=0.0, sigma=0.6, size=usuarios)
    usuario_ids = rng.choice(
        np.arange(1, usuarios + 1, dtype=np.int32), size=filas, p=pesos / pesos.sum()
    )

    tipo_idx = rng.choice(len(TIPOS), size=filas, p=PROB_TIPOS)
    mu = np.array([MONTO_LOGNORMAL[t][0] for t in TIPOS])[tipo_idx]
    sigma = np.array([MONTO_LOGNORMAL[t][1] for t in TIPOS])[tipo_idx]
    montos_centavos = np.clip(
        np.rint(rng.lognormal(mu, sigma) * 100), 1, MONTO_MAX_CENTAVOS
    ).astype(np.int64)
    estado_idx = rng.choice(len(ESTADOS), size=filas, p=PROB_ESTADOS)

    df = pl.DataFrame(
        {
            "id": pl.int_range(1, filas + 1, dtype=pl.Int64, eager=True),
            "fecha": pl.Series(fechas_us).cast(pl.Datetime("us")).dt.replace_time_zone("UTC"),
            "usuario_id": usuario_ids,
            "monto_centavos": montos_centavos,
            "tipo_transaccion": pl.Series(TIPOS)[tipo_idx],
            "estado": pl.Series(ESTADOS)[estado_idx],
        }
    )
    return ensuciar(df, rng)


def ensuciar(df: pl.DataFrame, rng: np.random.Generator) -> pl.DataFrame:
    """Inyecta defectos típicos de un sistema origen real."""
    n = df.height

    def mascara(clave: str) -> pl.Series:
        return pl.Series(rng.random(n) < SUCIEDAD[clave])

    def anular(col: str, clave: str) -> pl.Expr:
        return pl.when(mascara(clave)).then(None).otherwise(pl.col(col)).alias(col)

    # Tipos con mayúsculas/espacios inconsistentes: " compra ", "Deposito", ...
    variante = pl.Series(rng.integers(0, 3, size=n))
    tipo = pl.col("tipo_transaccion")
    tipo_sucio = (
        pl.when(variante == 0)
        .then(tipo.str.to_lowercase())
        .when(variante == 1)
        .then(tipo.str.to_titlecase())
        .otherwise(pl.lit("  ") + tipo + pl.lit(" "))
    )
    moneda_100 = pl.lit(100, dtype=MONEDA)
    signo = pl.when(mascara("monto_negativo")).then(-1).otherwise(1)

    df = df.with_columns(
        (pl.col("monto_centavos") * signo).alias("monto_centavos"),
        pl.when(mascara("tipo_mal_formato")).then(tipo_sucio).otherwise(tipo).alias(
            "tipo_transaccion"
        ),
    ).with_columns(
        # El origen publica el monto como DECIMAL(14,2): nunca float para dinero.
        (pl.col("monto_centavos").cast(MONEDA) / moneda_100).cast(MONEDA).alias("monto"),
    ).with_columns(
        anular("monto", "monto_nulo"),
        anular("tipo_transaccion", "tipo_nulo"),
        anular("estado", "estado_nulo"),
        anular("usuario_id", "usuario_nulo"),
        anular("fecha", "fecha_nula"),
    )

    # Duplicados exactos (reenvíos del sistema origen) añadidos al final.
    n_dup = int(n * SUCIEDAD["duplicados"])
    duplicados = df[rng.choice(n, size=n_dup, replace=False)]
    return pl.concat([df, duplicados]).select(
        "id", "fecha", "usuario_id", "monto", "tipo_transaccion", "estado"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--filas", type=int, default=1_000_000)
    parser.add_argument("--usuarios", type=int, default=10_000)
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--salida", type=Path, default=config.PARQUET_PATH)
    args = parser.parse_args()

    t0 = time.perf_counter()
    df = generar(args.filas, args.usuarios, args.semilla)
    t_gen = time.perf_counter() - t0

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    t1 = time.perf_counter()
    df.write_parquet(args.salida, compression="zstd", statistics=True)
    t_escritura = time.perf_counter() - t1

    log.info("Esquema: %s", dict(df.schema))
    log.info("Nulos por columna: %s", df.null_count().row(0, named=True))
    log.info("Ids duplicados: %s", f"{df.height - df['id'].n_unique():,}")
    log.info("Montos <= 0: %s", f"{df.filter(pl.col('monto') <= 0).height:,}")
    log.info("Muestra:\n%s", df.head(5))
    log.info(
        "Generadas %s filas en %.2fs; Parquet escrito en %.2fs -> %s (%.1f MB)",
        f"{df.height:,}",
        t_gen,
        t_escritura,
        args.salida,
        args.salida.stat().st_size / 1e6,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
