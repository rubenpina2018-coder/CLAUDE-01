"""DDL compartido: tabla de transacciones, índices y vista materializada de resumen.

Es la única fuente de verdad del esquema:
  * `init_db.py` lo usa para crear el esquema inicial.
  * `etl_pipeline.py` lo reutiliza para construir una tabla de carga idéntica
    (sin índices durante el COPY), crear sus índices al final y hacer el swap.
"""

from __future__ import annotations

from collections.abc import Iterable

TABLA = "transacciones_financieras"
VISTA_RESUMEN = "mv_resumen_transacciones"

# Dominios válidos: los CHECK de la tabla y las reglas del ETL usan los mismos.
TIPOS_TRANSACCION: tuple[str, ...] = (
    "COMPRA",
    "DEPOSITO",
    "RETIRO",
    "TRANSFERENCIA",
    "PAGO_SERVICIO",
)
ESTADOS: tuple[str, ...] = ("COMPLETADA", "PENDIENTE", "FALLIDA", "REVERTIDA")

# Orden físico de las columnas (el CSV que el ETL envía por COPY sigue este orden).
COLUMNAS: tuple[str, ...] = (
    "id",
    "fecha",
    "usuario_id",
    "monto",
    "monto_neto",
    "tipo_transaccion",
    "estado",
)

# Los índices se nombran f"{tabla}_{sufijo}", así la tabla de carga y la
# definitiva nunca colisionan y el swap solo tiene que renombrarlos.
SUFIJO_PK = "pkey"
SUFIJO_IDX_USUARIO = "usuario_fecha_idx"
SUFIJOS_INDICES: tuple[str, ...] = (SUFIJO_PK, SUFIJO_IDX_USUARIO)


def _lista_sql(valores: Iterable[str]) -> str:
    return ", ".join(f"'{v}'" for v in valores)


def crear_tabla_sql(tabla: str = TABLA, *, con_pk: bool = True) -> str:
    """CREATE TABLE. Con `con_pk=False` se omite la PK para acelerar la carga masiva."""
    pk = f",\n    CONSTRAINT {tabla}_{SUFIJO_PK} PRIMARY KEY (id)" if con_pk else ""
    return f"""
CREATE TABLE IF NOT EXISTS {tabla} (
    id               BIGINT        NOT NULL,
    fecha            TIMESTAMPTZ   NOT NULL,
    usuario_id       INTEGER       NOT NULL,
    monto            NUMERIC(14,2) NOT NULL,
    monto_neto       NUMERIC(14,2) NOT NULL,
    tipo_transaccion TEXT          NOT NULL,
    estado           TEXT          NOT NULL,
    CONSTRAINT chk_monto_positivo CHECK (monto > 0),
    CONSTRAINT chk_monto_neto     CHECK (monto_neto >= 0 AND monto_neto <= monto),
    CONSTRAINT chk_tipo_valido    CHECK (tipo_transaccion IN ({_lista_sql(TIPOS_TRANSACCION)})),
    CONSTRAINT chk_estado_valido  CHECK (estado IN ({_lista_sql(ESTADOS)})){pk}
)"""


def indice_pk_sql(tabla: str) -> str:
    """Índice único que después se promueve a PK (ver `promover_pk_sql`)."""
    return f"CREATE UNIQUE INDEX IF NOT EXISTS {tabla}_{SUFIJO_PK} ON {tabla} (id)"


def indice_usuario_sql(tabla: str) -> str:
    """Sirve el historial de un usuario ya ordenado por fecha (sin paso de sort)."""
    return (
        f"CREATE INDEX IF NOT EXISTS {tabla}_{SUFIJO_IDX_USUARIO} "
        f"ON {tabla} (usuario_id, fecha DESC, id DESC)"
    )


def promover_pk_sql(tabla: str) -> str:
    """Convierte el índice único ya construido en la PK (operación solo de catálogo)."""
    return (
        f"ALTER TABLE {tabla} ADD CONSTRAINT {tabla}_{SUFIJO_PK} "
        f"PRIMARY KEY USING INDEX {tabla}_{SUFIJO_PK}"
    )


def crear_vista_resumen_sql(vista: str = VISTA_RESUMEN, tabla: str = TABLA) -> str:
    """Agregados precalculados para `GET /api/v1/resumen`.

    `GROUP BY ROLLUP` produce una fila por tipo de transacción más la fila del
    total global (la que tiene `tipo_transaccion` NULL), así la API responde con
    una lectura de ~6 filas en lugar de agregar 1M de filas en cada petición.
    """
    return f"""
CREATE MATERIALIZED VIEW IF NOT EXISTS {vista} AS
SELECT
    tipo_transaccion,
    count(*)::bigint                            AS total_transacciones,
    coalesce(sum(monto), 0)::numeric(20,2)      AS monto_total,
    coalesce(sum(monto_neto), 0)::numeric(20,2) AS monto_neto_total,
    now()                                       AS actualizado_en
FROM {tabla}
GROUP BY ROLLUP (tipo_transaccion)"""


def eliminar_esquema_sql(tabla: str = TABLA, vista: str = VISTA_RESUMEN) -> str:
    return f"DROP MATERIALIZED VIEW IF EXISTS {vista}; DROP TABLE IF EXISTS {tabla};"
