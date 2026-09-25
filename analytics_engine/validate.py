"""Validaciones de la carga (patrón write-audit-publish).

* :func:`model_checks` se ejecuta DENTRO de la transacción de carga, antes del
  COMMIT, sobre la misma conexión: si alguna comprobación falla se hace
  ROLLBACK y los consumidores nunca ven datos incoherentes.
* :func:`bi_checks` verifica la capa BI tras publicarla (refresco de vistas).

Un fallo marca la ejecución como ``failed`` en ``audit.etl_run`` y el proceso
termina con código de salida distinto de cero, de modo que un orquestador
(cron, Airflow, n8n...) pueda alertar.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal

from sqlalchemy import text
from sqlalchemy.engine import Engine

from analytics_engine.transform import SheetResult

Query = Callable[[str], tuple]  # ejecuta una consulta y devuelve su única fila


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


class ValidationFailed(RuntimeError):
    def __init__(self, checks: list[Check]):
        self.checks = checks
        failed = ", ".join(c.name for c in checks if not c.passed)
        super().__init__(f"Validaciones fallidas antes de publicar: {failed}")


def _cents(value: Decimal | None) -> int:
    return int((value or Decimal(0)) * 100)


def model_checks(query: Query, products: SheetResult, customers: SheetResult, sales: SheetResult,
                 manifest: dict | None = None) -> list[Check]:
    checks: list[Check] = []
    s = sales.clean

    # 1. Volumen: la tabla de hechos refleja exactamente el snapshot válido de la hoja
    (fact_rows,) = query("SELECT count(*) FROM dw.fact_sales")
    checks.append(Check("filas_hechos", fact_rows == s.height,
                        f"dw.fact_sales={fact_rows:,} | filas válidas en origen={s.height:,}"))

    # 2. Importes: reconciliación exacta al céntimo (aritmética entera en ambos lados)
    db = query("""SELECT sum(quantity), sum(gross_amount), sum(discount_amount), sum(net_amount),
                         sum(cost_amount), sum(profit_amount) FROM dw.fact_sales""")
    expected = [int(s["quantity"].sum())] + [int(s[f"{n}_cents"].sum())
                                              for n in ("gross", "discount", "net", "cost", "profit")]
    actual = [int(db[0] or 0)] + [_cents(v) for v in db[1:]]
    checks.append(Check("importes_al_centimo", actual == expected,
                        f"neto BD={actual[3] / 100:,.2f} | neto origen={expected[3] / 100:,.2f} | "
                        f"unidades {actual[0]:,}/{expected[0]:,}"))

    # 3. Integridad referencial (explícita: en modo bulk las FK se recrean en la misma transacción)
    (orphans,) = query("""
        SELECT count(*) FROM dw.fact_sales f
        LEFT JOIN dw.dim_date d ON d.date_key = f.date_key
        LEFT JOIN dw.dim_customer c ON c.customer_key = f.customer_key
        LEFT JOIN dw.dim_product p ON p.product_key = f.product_key
        LEFT JOIN dw.dim_channel ch ON ch.channel_key = f.channel_key
        WHERE d.date_key IS NULL OR c.customer_key IS NULL OR p.product_key IS NULL OR ch.channel_key IS NULL""")
    checks.append(Check("integridad_referencial", orphans == 0, f"hechos huérfanos={orphans}"))

    # 4. Calendario: años completos, sin huecos y cubriendo todas las ventas
    n_days, d_min, d_max = query("SELECT count(*), min(full_date), max(full_date) FROM dw.dim_date")
    f_min, f_max = query("""SELECT min(d.full_date), max(d.full_date)
                            FROM dw.fact_sales f JOIN dw.dim_date d USING (date_key)""")
    contiguous = d_min is not None and n_days == (d_max - d_min + timedelta(days=1)).days
    full_years = d_min is not None and (d_min.month, d_min.day, d_max.month, d_max.day) == (1, 1, 12, 31)
    covers = f_min is None or (d_min <= f_min and f_max <= d_max)
    checks.append(Check("calendario_continuo", contiguous and full_years and covers,
                        f"{d_min} -> {d_max} ({n_days:,} días); ventas {f_min} -> {f_max}"))

    # 5. Dimensiones: todo lo publicado en las hojas está en el modelo
    (missing_customers,) = query("""SELECT count(*) FROM staging.stg_customer s
                                    WHERE NOT EXISTS (SELECT 1 FROM dw.dim_customer c
                                                      WHERE c.customer_id = s.customer_id)""")
    (missing_products,) = query("""SELECT count(*) FROM staging.stg_product s
                                   WHERE NOT EXISTS (SELECT 1 FROM dw.dim_product p WHERE p.sku = s.sku)""")
    checks.append(Check("dimensiones_completas", missing_customers == 0 and missing_products == 0,
                        f"clientes={customers.clean.height:,} productos={products.clean.height:,} "
                        f"(faltantes: {missing_customers}/{missing_products})"))

    # 6. Atributo derivado first_order_date coherente con los hechos
    (bad_first,) = query("""
        SELECT count(*) FROM dw.dim_customer c
        LEFT JOIN (SELECT f.customer_key, min(d.full_date) AS first_date
                   FROM dw.fact_sales f JOIN dw.dim_date d USING (date_key)
                   GROUP BY f.customer_key) x USING (customer_key)
        WHERE c.customer_key <> -1 AND c.first_order_date IS DISTINCT FROM x.first_date""")
    checks.append(Check("primera_compra_coherente", bad_first == 0, f"clientes incoherentes={bad_first}"))

    # 7. Ventas sin cliente identificable (invitados o ID inexistente) -> miembro -1
    (unknown,) = query("SELECT count(*) FROM dw.fact_sales WHERE customer_key = -1")
    ratio = unknown / fact_rows if fact_rows else 0.0
    checks.append(Check("clientes_no_identificados", ratio <= 0.05,
                        f"{unknown:,} líneas ({ratio:.2%}) asignadas a 'Cliente no identificado' (umbral 5 %)"))

    # 8. Autocomprobación frente al manifiesto del generador (solo con datos sintéticos)
    if manifest:
        exp = manifest.get("expected_etl", {})
        expected_rejects = {k: v for k, v in exp.get("rejected", {}).items() if v}
        diffs = []
        if exp.get("fact_rows") != s.height:
            diffs.append(f"filas {s.height} != {exp.get('fact_rows')}")
        if exp.get("net_amount_cents") != int(s["net_cents"].sum()):
            diffs.append("importe neto distinto")
        if sales.rejected_by_reason != expected_rejects:
            diffs.append(f"rechazos {sales.rejected_by_reason} != {expected_rejects}")
        if exp.get("unknown_customer_rows") is not None and exp["unknown_customer_rows"] != unknown:
            diffs.append(f"clientes no identificados {unknown} != {exp['unknown_customer_rows']}")
        checks.append(Check("manifiesto_generador", not diffs,
                            "resultado idéntico al esperado por generate_data.py" if not diffs else "; ".join(diffs)))
    return checks


def bi_checks(engine: Engine, sales: SheetResult) -> list[Check]:
    """Coherencia de la capa BI publicada y permisos mínimos del rol bi_reader."""
    checks: list[Check] = []
    with engine.connect() as conn:
        bi_rows, bi_net = conn.execute(text("SELECT count(*), sum(net_amount) FROM bi.mv_sales_flat")).one()
        expected_net = int(sales.clean["net_cents"].sum())
        checks.append(Check("capa_bi_sincronizada", bi_rows == sales.clean.height and _cents(bi_net) == expected_net,
                            f"bi.mv_sales_flat={bi_rows:,} filas, neto={_cents(bi_net) / 100:,.2f}"))
        if conn.execute(text("SELECT count(*) FROM pg_roles WHERE rolname = 'bi_reader'")).scalar() == 1:
            can_bi, can_dw = conn.execute(text("""
                SELECT has_table_privilege('bi_reader', 'bi.mv_sales_flat', 'SELECT'),
                       has_table_privilege('bi_reader', 'dw.fact_sales', 'SELECT')""")).one()
            checks.append(Check("permisos_bi_reader", can_bi and not can_dw,
                                f"lee bi={can_bi} | lee dw={can_dw} (esperado: True | False)"))
    return checks
