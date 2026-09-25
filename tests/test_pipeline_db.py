"""Tests de integración de extremo a extremo contra PostgreSQL.

Usan una base de datos dedicada (``<DW_DATABASE>_test``, creada con
``python setup_database.py --with-test-db``) que se reinicia al empezar.
Si no está disponible, los tests se marcan como omitidos (skip).

Los tests de este módulo se ejecutan en orden y comparten el estado de la BD:
carga inicial -> re-ejecución idempotente -> cambios incrementales -> protecciones.
"""

import json
import os
import shutil
from pathlib import Path

import polars as pl
import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

import etl_pipeline
from analytics_engine.config import SQL_DIR, DbSettings

pytestmark = pytest.mark.db


@pytest.fixture(scope="module")
def db():
    settings = DbSettings.from_env()
    if not settings.password:
        pytest.skip("DW_PASSWORD no configurada (copia .env.example a .env)")
    test_settings = settings.with_database(os.getenv("DW_TEST_DATABASE", f"{settings.database}_test"))
    engine = test_settings.create_engine()
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except OperationalError as exc:
        engine.dispose()
        pytest.skip(f"BD de test no disponible ({test_settings.describe()}): {exc.orig}")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("DW_DATABASE", test_settings.database)  # el pipeline leerá esta BD
        yield engine
    engine.dispose()


def run_pipeline(source: Path, tmp_path: Path, *extra: str) -> int:
    return etl_pipeline.main(["--source-dir", str(source), "--rejected-dir", str(tmp_path / "rejected"), *extra])


def scalar(engine, sql: str, **params):
    with engine.connect() as conn:
        return conn.execute(text(sql), params).scalar()


def last_run(engine) -> dict:
    with engine.connect() as conn:
        row = conn.execute(text("SELECT status, metrics, rows_loaded FROM audit.etl_run "
                                "ORDER BY run_id DESC LIMIT 1")).mappings().one()
    return dict(row)


def copy_dataset(src: Path, dst: Path, keep_manifest: bool = False) -> Path:
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("productos.csv", "clientes.csv", "ventas.csv"):
        shutil.copy(src / name, dst / name)
    if keep_manifest:
        shutil.copy(src / "_manifest.json", dst / "_manifest.json")
    return dst


def test_full_load_from_scratch(db, small_dataset, manifest, tmp_path):
    assert run_pipeline(small_dataset, tmp_path, "--reset") == 0
    expected = manifest["expected_etl"]
    assert scalar(db, "SELECT count(*) FROM dw.fact_sales") == expected["fact_rows"]
    assert scalar(db, "SELECT round(sum(net_amount) * 100) FROM dw.fact_sales") == expected["net_amount_cents"]
    assert scalar(db, "SELECT count(*) FROM dw.dim_customer WHERE customer_key <> -1") == expected["dim_customer_rows"]
    assert scalar(db, "SELECT count(*) FROM dw.dim_product") == expected["dim_product_rows"]
    assert scalar(db, "SELECT count(*) FROM dw.fact_sales WHERE customer_key = -1") == expected["unknown_customer_rows"]
    assert scalar(db, "SELECT count(*) FROM bi.mv_sales_flat") == expected["fact_rows"]
    run = last_run(db)
    assert run["status"] == "success" and run["rows_loaded"] == expected["fact_rows"]
    assert run["metrics"]["load"]["fact_sales"]["bulk_mode"] is (expected["fact_rows"] >= 10_000)
    assert all(check["passed"] for check in run["metrics"]["checks"])
    # Las FK se recrean tras el modo bulk (o nunca se retiraron)
    assert scalar(db, "SELECT count(*) FROM pg_constraint WHERE conrelid = 'dw.fact_sales'::regclass "
                      "AND contype = 'f'") == 4


def test_rerun_with_same_data_changes_nothing(db, small_dataset, tmp_path):
    assert run_pipeline(small_dataset, tmp_path) == 0
    metrics = last_run(db)["metrics"]
    for table in ("dim_date", "dim_product", "dim_customer", "fact_sales"):
        assert metrics["load"][table]["inserted"] == 0, table
        assert metrics["load"][table]["updated"] == 0, table
    assert metrics["load"]["fact_sales"]["deleted"] == 0
    assert metrics["bi"]["action"] == "skipped"


def test_incremental_changes_are_synchronized(db, small_dataset, tmp_path):
    src = copy_dataset(small_dataset, tmp_path / "changed")
    sales = pl.read_csv(src / "ventas.csv", infer_schema=False)
    key = (pl.col("ID Pedido").str.strip_chars().str.to_uppercase() + "/" + pl.col("Línea").str.strip_chars())
    single = set(sales.group_by(key.alias("k")).len().filter(pl.col("len") == 1)["k"].to_list())
    with db.connect() as conn:
        loaded = [r[0] for r in conn.execute(text(
            "SELECT order_id || '/' || order_line FROM dw.fact_sales ORDER BY order_id, order_line LIMIT 300"))]
    candidates = [k for k in loaded if k in single]
    to_delete, to_update = candidates[:5], candidates[5]
    old_qty = scalar(db, "SELECT quantity FROM dw.fact_sales WHERE order_id || '/' || order_line = :k", k=to_update)
    new_qty = old_qty + 5
    sku = scalar(db, "SELECT sku FROM dw.dim_product ORDER BY sku LIMIT 1")
    new_line = {c: None for c in sales.columns} | {
        "ID Pedido": "PED-9999999", "Línea": "1", "Fecha Pedido ": "2026-08-15", "SKU": sku, "Canal": "Online",
        "Método de Pago": "PayPal", "Cantidad": "2", "Precio Unitario": "10,00 €", "Estado": "Completado"}
    sales = (sales.filter(~key.is_in(to_delete))
             .with_columns(pl.when(key == to_update).then(pl.lit(str(new_qty))).otherwise(pl.col("Cantidad"))
                           .alias("Cantidad")))
    pl.concat([sales, pl.DataFrame([new_line], schema=sales.schema)]).write_csv(src / "ventas.csv")

    customers = pl.read_csv(src / "clientes.csv", infer_schema=False)
    once = customers.group_by("ID Cliente").len().filter(pl.col("len") == 1)["ID Cliente"]
    target = next(c for c in once.to_list() if c and c.startswith("CLI-"))
    current = scalar(db, "SELECT segment FROM dw.dim_customer WHERE customer_id = :c", c=target)
    new_segment = "Corporativo" if current != "Corporativo" else "Pyme"
    customers.with_columns(pl.when(pl.col("ID Cliente") == target).then(pl.lit(new_segment))
                           .otherwise(pl.col("Segmento")).alias("Segmento")).write_csv(src / "clientes.csv")

    assert run_pipeline(src, tmp_path) == 0
    load = last_run(db)["metrics"]["load"]
    assert (load["fact_sales"]["inserted"], load["fact_sales"]["updated"], load["fact_sales"]["deleted"]) == (1, 1, 5)
    assert load["fact_sales"]["bulk_mode"] is False  # pocos cambios: FK comprobadas fila a fila
    assert load["dim_customer"]["updated"] == 1
    assert scalar(db, "SELECT quantity FROM dw.fact_sales WHERE order_id || '/' || order_line = :k",
                  k=to_update) == new_qty
    assert scalar(db, "SELECT net_amount FROM dw.fact_sales WHERE order_id = 'PED-9999999'") == 20
    assert scalar(db, "SELECT segment FROM dw.dim_customer WHERE customer_id = :c", c=target) == new_segment
    assert scalar(db, "SELECT count(*) FROM bi.mv_sales_flat WHERE order_id = 'PED-9999999'") == 1
    assert last_run(db)["metrics"]["bi"]["action"] == "refreshed"


def test_truncated_export_is_blocked_by_circuit_breaker(db, small_dataset, tmp_path):
    before = scalar(db, "SELECT count(*) FROM dw.fact_sales")
    src = copy_dataset(small_dataset, tmp_path / "truncated")
    pl.read_csv(src / "ventas.csv", infer_schema=False).head(100).write_csv(src / "ventas.csv")
    assert run_pipeline(src, tmp_path) == 2
    assert scalar(db, "SELECT count(*) FROM dw.fact_sales") == before
    run = last_run(db)
    assert run["status"] == "failed" and run["rows_loaded"] == 0


def test_failed_validation_rolls_back_the_load(db, small_dataset, tmp_path):
    """Write-audit-publish: si una validación falla antes del COMMIT, no se publica nada."""
    assert run_pipeline(small_dataset, tmp_path) == 0  # vuelve al estado original
    src = copy_dataset(small_dataset, tmp_path / "tampered", keep_manifest=True)
    sales = pl.read_csv(src / "ventas.csv", infer_schema=False)
    target = sales.filter(pl.col("ID Pedido").str.starts_with("PED-") & (pl.col("Cantidad") == "1")).row(0, named=True)
    sales.with_columns(
        pl.when((pl.col("ID Pedido") == target["ID Pedido"]) & (pl.col("Línea") == target["Línea"]))
        .then(pl.lit("9")).otherwise(pl.col("Cantidad")).alias("Cantidad")).write_csv(src / "ventas.csv")
    net_before = scalar(db, "SELECT sum(net_amount) FROM dw.fact_sales")
    assert run_pipeline(src, tmp_path) == 1  # el manifiesto ya no cuadra -> ROLLBACK
    assert scalar(db, "SELECT sum(net_amount) FROM dw.fact_sales") == net_before
    run = last_run(db)
    assert run["status"] == "failed" and run["rows_loaded"] == 0
    assert any(c["name"] == "manifiesto_generador" and not c["passed"] for c in run["metrics"]["checks"])


def test_bi_reader_is_read_only_and_limited_to_bi(db):
    bi_password = os.getenv("BI_READER_PASSWORD")
    if not bi_password:
        pytest.skip("BI_READER_PASSWORD no configurada")
    settings = DbSettings.from_env().with_database(os.environ["DW_DATABASE"])
    reader = type(settings)(host=settings.host, port=settings.port, database=settings.database,
                            user="bi_reader", password=bi_password, sslmode=settings.sslmode).create_engine()
    try:
        # Cursor DBAPI sin parámetros: el SQL contiene '%' en literales (psycopg2 los interpretaría)
        raw = reader.raw_connection()
        try:
            with raw.cursor() as cur:
                cur.execute((SQL_DIR / "kpi_validation.sql").read_text(encoding="utf-8"))
                rows = cur.fetchall()
        finally:
            raw.close()
        assert len(rows) == 14 and all(row[3] is not None for row in rows)
        with reader.connect() as conn:
            assert conn.execute(text("SELECT count(*) FROM bi.mv_kpi_monthly")).scalar() > 0
        with reader.connect() as conn, pytest.raises(Exception, match="permission denied"):
            conn.execute(text("SELECT 1 FROM dw.fact_sales LIMIT 1"))
        with reader.connect() as conn, pytest.raises(Exception, match="read-only transaction"):
            conn.execute(text("CREATE TABLE bi.not_allowed (id int)"))
    finally:
        reader.dispose()


def test_audit_trail_keeps_rejected_rows_with_sheet_row_numbers(db):
    with db.connect() as conn:
        row = conn.execute(text("""
            SELECT rr.source_sheet, rr.source_row, rr.reason, rr.record
            FROM audit.rejected_record rr JOIN audit.etl_run r USING (run_id)
            WHERE r.status = 'success' AND rr.reason = 'fecha_invalida'
            ORDER BY rr.rejected_id DESC LIMIT 1""")).mappings().one()
    assert row["source_sheet"] == "ventas" and row["source_row"] >= 2
    assert "ID Pedido" in (row["record"] if isinstance(row["record"], dict) else json.loads(row["record"]))
