"""Tests de integración de la API contra PostgreSQL con datos cargados por el ETL.

Se omiten automáticamente si la base de datos no está disponible. La API no
valida sus respuestas en caliente (por rendimiento); estos tests verifican que
cumplen el contrato publicado en OpenAPI (modelos Pydantic de main.py).
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import asyncpg
import pytest
from fastapi.testclient import TestClient

import config
from main import HistorialUsuario, Resumen, app


def _usuario_con_datos() -> tuple[int, int] | None:
    async def consultar() -> tuple[int, int] | None:
        conn = await asyncpg.connect(config.DATABASE_URL, timeout=2)
        try:
            fila = await conn.fetchrow(
                "SELECT usuario_id, count(*) AS n FROM transacciones_financieras "
                "GROUP BY usuario_id ORDER BY count(*) DESC LIMIT 1"
            )
            return (fila["usuario_id"], fila["n"]) if fila else None
        finally:
            await conn.close()

    try:
        return asyncio.run(consultar())
    except (TimeoutError, OSError, asyncpg.PostgresError):
        return None


USUARIO = _usuario_con_datos()
pytestmark = pytest.mark.skipif(
    USUARIO is None, reason="PostgreSQL no disponible o sin datos (ejecutar el ETL)"
)


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    with TestClient(app) as c:  # ejecuta el lifespan: crea y cierra el pool
        yield c


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_resumen_cumple_contrato_y_cuadra(client: TestClient) -> None:
    r = client.get("/api/v1/resumen")
    assert r.status_code == 200
    resumen = Resumen.model_validate(r.json())
    assert resumen.total_transacciones > 0
    assert {t.tipo_transaccion for t in resumen.por_tipo} <= set(
        ("COMPRA", "DEPOSITO", "RETIRO", "TRANSFERENCIA", "PAGO_SERVICIO")
    )
    assert sum(t.total_transacciones for t in resumen.por_tipo) == resumen.total_transacciones
    assert sum(t.monto_total for t in resumen.por_tipo) == pytest.approx(resumen.monto_total)
    assert resumen.monto_neto_total <= resumen.monto_total


def test_historial_cumple_contrato_y_orden(client: TestClient) -> None:
    usuario_id, total = USUARIO
    r = client.get(f"/api/v1/transacciones/{usuario_id}", params={"limit": 1000})
    assert r.status_code == 200
    historial = HistorialUsuario.model_validate(r.json())
    assert historial.usuario_id == usuario_id
    assert historial.cantidad == len(historial.transacciones) == min(total, 1000)
    claves = [(t.fecha, t.id) for t in historial.transacciones]
    assert claves == sorted(claves, reverse=True)  # más reciente primero
    assert all(0 <= t.monto_neto <= t.monto for t in historial.transacciones)


def test_historial_paginacion(client: TestClient) -> None:
    usuario_id, _ = USUARIO
    completo = client.get(f"/api/v1/transacciones/{usuario_id}", params={"limit": 10}).json()
    pagina = client.get(
        f"/api/v1/transacciones/{usuario_id}", params={"limit": 5, "offset": 5}
    ).json()
    assert pagina["transacciones"] == completo["transacciones"][5:10]


def test_historial_usuario_inexistente_devuelve_404(client: TestClient) -> None:
    r = client.get("/api/v1/transacciones/2000000000")
    assert r.status_code == 404
    assert "no tiene transacciones" in r.json()["detail"]


@pytest.mark.parametrize(
    "url",
    [
        "/api/v1/transacciones/0",
        "/api/v1/transacciones/abc",
        "/api/v1/transacciones/99999999999",  # fuera de rango INTEGER
        "/api/v1/transacciones/1?limit=0",
        "/api/v1/transacciones/1?limit=1001",
        "/api/v1/transacciones/1?offset=-1",
        "/api/v1/transacciones/1?offset=99999999999999999999",  # fuera de int64
    ],
)
def test_parametros_invalidos_devuelven_422(client: TestClient, url: str) -> None:
    assert client.get(url).status_code == 422
