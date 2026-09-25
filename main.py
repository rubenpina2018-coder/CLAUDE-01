"""Hito 4 — API asíncrona de alta velocidad (FastAPI + asyncpg).

Endpoints:
  GET /api/v1/resumen                     Volumen total, monto global y desglose por tipo.
  GET /api/v1/transacciones/{usuario_id}  Historial de un usuario, más reciente primero.
  GET /health                             Readiness: verifica que la base de datos responde.

Decisiones de rendimiento:
  * Un pool asyncpg por proceso, creado en el `lifespan`, abierto completo al
    arrancar y con cada conexión precalentada. asyncpg usa el protocolo binario
    de PostgreSQL y cachea cada sentencia preparada por conexión.
  * /resumen lee la vista materializada que el ETL recalcula en cada carga:
    6 filas precalculadas en lugar de agregar ~1M de filas por petición.
  * /transacciones usa el índice (usuario_id, fecha DESC, id DESC) sobre una tabla
    cargada en orden físico por usuario: unas 5 páginas de 8 KB por consulta.
  * El pool no ejecuta la consulta de reset al liberar cada conexión (ahorra un
    viaje a la base de datos por petición; ver `_reset_sin_consulta`).
  * El historial se serializa a JSON dentro de PostgreSQL y se incrusta con
    orjson.Fragment; el resumen se serializa con orjson. Los modelos Pydantic
    documentan el contrato en OpenAPI y los tests lo verifican, sin pagar una
    validación redundante en cada petición.

Ejecución:
    uvicorn main:app --host 127.0.0.1 --port 8000 --workers 4 --no-access-log
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated, Any

import asyncpg
import orjson
from fastapi import FastAPI, HTTPException, Path, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import config
from db_schema import TABLA, VISTA_RESUMEN

# Logger general de uvicorn: los mensajes salen con el formato y nivel del servidor.
log = logging.getLogger("uvicorn.error")

POOL_MAX = int(os.getenv("DB_POOL_MAX", "10"))
# Pool pre-abierto por defecto: abrir conexiones (autenticación SCRAM incluida)
# en mitad de un pico de tráfico disparaba el p99.9 de 100 ms a 330 ms.
POOL_MIN = int(os.getenv("DB_POOL_MIN", str(POOL_MAX)))
INT4_MAX = 2_147_483_647  # usuario_id es INTEGER en PostgreSQL

SQL_RESUMEN = f"""
SELECT tipo_transaccion, total_transacciones,
       monto_total::float8 AS monto_total, monto_neto_total::float8 AS monto_neto_total,
       actualizado_en
FROM {VISTA_RESUMEN}
ORDER BY tipo_transaccion NULLS FIRST
"""

# PostgreSQL construye el array JSON del historial (row_to_json, en C) y Python
# lo incrusta tal cual con orjson.Fragment: sin crear ~600 objetos Python por
# petición ni re-serializarlos. Los NUMERIC salen como números JSON exactos.
# (string_agg en vez de json_agg para obtener JSON compacto, sin saltos de línea).
SQL_HISTORIAL = f"""
SELECT count(*) AS cantidad,
       '[' || coalesce(string_agg(row_to_json(t)::text, ',' ORDER BY t.fecha DESC, t.id DESC), '')
           || ']' AS transacciones
FROM (
    SELECT id, fecha, monto, monto_neto, tipo_transaccion, estado
    FROM {TABLA}
    WHERE usuario_id = $1
    ORDER BY fecha DESC, id DESC
    LIMIT $2 OFFSET $3
) AS t
"""


# --------------------------------------------------------------------------- #
# Contrato de respuesta (documentación OpenAPI; verificado en tests/test_api.py)
# --------------------------------------------------------------------------- #
class ResumenPorTipo(BaseModel):
    tipo_transaccion: str
    total_transacciones: int
    monto_total: float
    monto_neto_total: float


class Resumen(BaseModel):
    total_transacciones: int
    monto_total: float
    monto_neto_total: float
    actualizado_en: datetime
    por_tipo: list[ResumenPorTipo]


class Transaccion(BaseModel):
    id: int
    fecha: datetime
    monto: float
    monto_neto: float
    tipo_transaccion: str
    estado: str


class HistorialUsuario(BaseModel):
    usuario_id: int
    cantidad: int
    limit: int
    offset: int
    transacciones: list[Transaccion]


class Error(BaseModel):
    detail: str


class RespuestaJSON(Response):
    """Serialización con orjson (Rust): datetime nativo y fragmentos JSON ya
    serializados (orjson.Fragment) incrustados sin volver a procesarlos."""

    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        return orjson.dumps(content)


# --------------------------------------------------------------------------- #
# Aplicación
# --------------------------------------------------------------------------- #
async def _reset_sin_consulta(conn: asyncpg.Connection) -> None:
    """Evita el reset por defecto de asyncpg al devolver la conexión al pool.

    El reset por defecto envía `RESET ALL; UNLISTEN *; CLOSE ALL; ...`: un viaje
    extra a la base de datos en cada petición. Esta API solo ejecuta SELECTs sin
    modificar el estado de la sesión, así que no hay nada que restaurar (asyncpg
    sigue haciendo ROLLBACK si alguna transacción quedara abierta).
    """


async def _preparar_conexion(conn: asyncpg.Connection) -> None:
    """Calienta cada conexión nueva: prepara y planifica las consultas de la API.

    Se ejecutan tal como lo harán los endpoints para que queden en la caché de
    sentencias de asyncpg antes de recibir tráfico; sin esto, la primera ráfaga
    de peticiones pagaba la preparación en cada una de las conexiones del pool.
    """
    try:
        await conn.fetch(SQL_RESUMEN)
        await conn.fetchrow(SQL_HISTORIAL, 0, 1, 0)
    except asyncpg.UndefinedTableError:
        log.warning("Esquema aún no creado: se omite el calentamiento de la conexión")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.pool = await asyncpg.create_pool(
        config.DATABASE_URL,
        min_size=POOL_MIN,
        max_size=POOL_MAX,
        init=_preparar_conexion,
        reset=_reset_sin_consulta,
        command_timeout=5,  # fallar rápido antes que acumular peticiones colgadas
        server_settings={"application_name": "api-transacciones", "jit": "off"},
    )
    log.info("Pool asyncpg listo (min=%d, max=%d)", POOL_MIN, POOL_MAX)
    try:
        yield
    finally:
        await app.state.pool.close()


app = FastAPI(
    title="API de Transacciones Financieras",
    version="1.0.0",
    summary="Consultas de baja latencia sobre los datos cargados por el pipeline ETL.",
    lifespan=lifespan,
)


async def bd_no_disponible(request: Request, exc: Exception) -> JSONResponse:
    log.error("Base de datos no disponible: %r", exc)
    return JSONResponse(status_code=503, content={"detail": "Base de datos no disponible"})


for _exc in (
    OSError,
    TimeoutError,
    asyncpg.PostgresConnectionError,
    asyncpg.CannotConnectNowError,
    asyncpg.InterfaceError,
    asyncpg.UndefinedTableError,  # esquema no inicializado: ejecutar init_db.py
):
    app.add_exception_handler(_exc, bd_no_disponible)


@app.get("/health", tags=["operación"])
async def health(request: Request) -> dict[str, str]:
    await request.app.state.pool.fetchval("SELECT 1")
    return {"status": "ok"}


@app.get(
    "/api/v1/resumen",
    response_model=Resumen,
    responses={503: {"model": Error}},
    tags=["transacciones"],
)
async def resumen(request: Request) -> Response:
    """Volumen total de transacciones y monto global, con desglose por tipo."""
    filas = await request.app.state.pool.fetch(SQL_RESUMEN)
    # GROUP BY ROLLUP garantiza la fila del total global (tipo NULL), que el
    # ORDER BY ... NULLS FIRST deja en primera posición.
    if not filas or filas[0]["tipo_transaccion"] is not None:
        raise HTTPException(status_code=503, detail="Resumen no disponible")
    total = filas[0]
    return RespuestaJSON(
        {
            "total_transacciones": total["total_transacciones"],
            "monto_total": total["monto_total"],
            "monto_neto_total": total["monto_neto_total"],
            "actualizado_en": total["actualizado_en"],
            "por_tipo": [
                {
                    "tipo_transaccion": f["tipo_transaccion"],
                    "total_transacciones": f["total_transacciones"],
                    "monto_total": f["monto_total"],
                    "monto_neto_total": f["monto_neto_total"],
                }
                for f in filas[1:]
            ],
        }
    )


@app.get(
    "/api/v1/transacciones/{usuario_id}",
    response_model=HistorialUsuario,
    responses={404: {"model": Error}, 503: {"model": Error}},
    tags=["transacciones"],
)
async def historial_usuario(
    request: Request,
    usuario_id: Annotated[int, Path(ge=1, le=INT4_MAX)],
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    # Cota superior: sin ella, un offset fuera de int64 llegaba a asyncpg como 500.
    offset: Annotated[int, Query(ge=0, le=INT4_MAX)] = 0,
) -> Response:
    """Historial de un usuario ordenado de la transacción más reciente a la más antigua."""
    fila = await request.app.state.pool.fetchrow(SQL_HISTORIAL, usuario_id, limit, offset)
    if fila["cantidad"] == 0 and offset == 0:
        raise HTTPException(
            status_code=404, detail=f"El usuario {usuario_id} no tiene transacciones"
        )
    return RespuestaJSON(
        {
            "usuario_id": usuario_id,
            "cantidad": fila["cantidad"],
            "limit": limit,
            "offset": offset,
            "transacciones": orjson.Fragment(fila["transacciones"]),
        }
    )
