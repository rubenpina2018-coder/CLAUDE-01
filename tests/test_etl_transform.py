"""Tests unitarios de las reglas del ETL (sin base de datos)."""

from __future__ import annotations

import random
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal

import polars as pl
import pytest

import config
import db_schema
from etl_pipeline import TASAS_IMPUESTO_PB, ejecutar, monto_neto_expr, transformar

MONEDA = pl.Decimal(14, 2)
F = datetime(2025, 6, 1, 12, 0, tzinfo=UTC)
ESQUEMA_RAW = {
    "id": pl.Int64,
    "fecha": pl.Datetime("us", "UTC"),
    "usuario_id": pl.Int32,
    "monto": MONEDA,
    "tipo_transaccion": pl.String,
    "estado": pl.String,
}


def neto_esperado(monto: Decimal, tipo: str) -> Decimal:
    tasa = Decimal(TASAS_IMPUESTO_PB[tipo]) / Decimal(10_000)
    impuesto = (monto * tasa).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return monto - impuesto


def calcular_neto(montos: list[Decimal], tipos: list[str]) -> list[Decimal]:
    df = pl.DataFrame(
        {"monto": montos, "tipo_transaccion": tipos}, schema_overrides={"monto": MONEDA}
    )
    return df.select(monto_neto_expr())["monto"].to_list()


@pytest.mark.parametrize(
    ("monto", "tipo", "neto"),
    [
        ("100.00", "COMPRA", "84.00"),  # 16 %
        ("100.00", "DEPOSITO", "100.00"),  # exento
        ("100.00", "TRANSFERENCIA", "99.60"),  # 0,4 %
        # Impuesto de exactamente medio centavo: HALF-UP sube (half-to-even no).
        ("0.10", "PAGO_SERVICIO", "0.09"),  # 0,005 -> 0,01
        ("1.25", "TRANSFERENCIA", "1.24"),  # 0,005 -> 0,01
        ("1.00", "RETIRO", "0.98"),  # 0,015 -> 0,02
        ("0.01", "COMPRA", "0.01"),  # 0,0016 -> 0,00
    ],
)
def test_monto_neto_casos_conocidos(monto: str, tipo: str, neto: str) -> None:
    assert calcular_neto([Decimal(monto)], [tipo]) == [Decimal(neto)]


def test_monto_neto_coincide_con_decimal_half_up() -> None:
    rng = random.Random(7)
    tipos = [rng.choice(db_schema.TIPOS_TRANSACCION) for _ in range(20_000)]
    montos = [Decimal(rng.randint(1, 5_000_000)) / 100 for _ in tipos]
    esperados = [neto_esperado(m, t) for m, t in zip(montos, tipos, strict=True)]
    assert calcular_neto(montos, tipos) == esperados


def test_transformar_limpia_normaliza_y_deduplica() -> None:
    filas = [
        # id, fecha, usuario, monto, tipo, estado
        (1, F, 10, Decimal("50.00"), " compra ", "Completada"),  # se normaliza
        (2, F, 10, Decimal("20.00"), "DEPOSITO", None),  # estado -> PENDIENTE
        (1, F, 10, Decimal("50.00"), "COMPRA", "COMPLETADA"),  # duplicado del id 1
        (3, F, None, Decimal("10.00"), "RETIRO", "COMPLETADA"),
        (4, None, 11, Decimal("10.00"), "RETIRO", "COMPLETADA"),
        (5, F, 11, None, "RETIRO", "COMPLETADA"),
        (6, F, 11, Decimal("-5.00"), "RETIRO", "COMPLETADA"),
        (7, F, 11, Decimal("5.00"), None, "COMPLETADA"),
        (8, F, 11, Decimal("5.00"), "CRIPTO", "COMPLETADA"),
        (9, F, 11, Decimal("5.00"), "RETIRO", "EN_REVISION"),
    ]
    raw = pl.DataFrame(filas, schema=ESQUEMA_RAW, orient="row")

    validas, rechazadas = transformar(raw)

    assert validas.columns == list(db_schema.COLUMNAS)
    assert validas["id"].to_list() == [1, 2]
    assert validas["tipo_transaccion"].to_list() == ["COMPRA", "DEPOSITO"]
    assert validas["estado"].to_list() == ["COMPLETADA", "PENDIENTE"]
    assert validas["monto_neto"].to_list() == [Decimal("42.00"), Decimal("20.00")]
    assert validas.schema["monto"] == MONEDA and validas.schema["monto_neto"] == MONEDA

    motivos = dict(zip(rechazadas["id"], rechazadas["motivo_rechazo"], strict=True))
    assert motivos == {
        1: "duplicado",
        3: "usuario_nulo",
        4: "fecha_nula",
        5: "monto_nulo",
        6: "monto_no_positivo",
        7: "tipo_nulo",
        8: "tipo_desconocido",
        9: "estado_desconocido",
    }
    assert validas.height + rechazadas.height == raw.height


def test_transformar_ordena_por_usuario_y_fecha() -> None:
    fechas = [datetime(2025, 1, d, tzinfo=UTC) for d in (3, 1, 2, 1)]
    raw = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "fecha": fechas,
            "usuario_id": [2, 2, 1, 1],
            "monto": [Decimal("1.00")] * 4,
            "tipo_transaccion": ["DEPOSITO"] * 4,
            "estado": ["COMPLETADA"] * 4,
        },
        schema=ESQUEMA_RAW,
    )
    validas, _ = transformar(raw)
    assert validas.select("usuario_id", "id").rows() == [(1, 4), (1, 3), (2, 2), (2, 1)]


def test_etl_no_publica_si_no_hay_filas_validas(tmp_path, monkeypatch) -> None:
    raw = pl.DataFrame(
        [(1, F, 10, None, "COMPRA", "COMPLETADA"), (2, F, None, Decimal("5.00"), "RETIRO", None)],
        schema=ESQUEMA_RAW,
        orient="row",
    )
    parquet = tmp_path / "solo_invalidas.parquet"
    raw.write_parquet(parquet)
    monkeypatch.setattr(config, "RECHAZADOS_PATH", tmp_path / "rechazados.parquet")

    # Aborta antes de conectar: el DSN inválido demuestra que la BD no se toca.
    with pytest.raises(RuntimeError, match="no se publica una tabla vacía"):
        ejecutar(parquet, "postgresql://nadie@127.0.0.1:1/ninguna", 1, 10)
    assert pl.read_parquet(tmp_path / "rechazados.parquet").height == 2
