#!/usr/bin/env python3
"""Pipeline ETL: exports de Google Sheets (CSV) -> Polars -> PostgreSQL (Star Schema) -> capa BI.

Fases:

1. Esquema         ejecuta ``sql/init.sql`` (idempotente, no destructivo).
2. Extracción      lee cada hoja como texto, igual que llega de Google Sheets.
3. Transformación  limpieza, normalización de tipos y reglas de calidad con Polars.
4. Quality gate    aborta si el % de filas rechazadas supera el umbral.
5. Carga           COPY a staging + merge set-based al Star Schema (una transacción).
6. Capa BI         despliega/refresca ``sql/bi_views.sql`` (Power BI / Looker Studio).
7. Validación      reconciliación origen-destino al céntimo y comprobaciones de integridad.

Cada ejecución queda registrada en ``audit.etl_run`` y las filas descartadas en
``audit.rejected_record`` (y en ``data/rejected/*.csv`` para quien mantiene las hojas).

Uso::

    python etl_pipeline.py                 # ejecución completa
    python etl_pipeline.py --dry-run       # solo extracción + transformación (no necesita BD)
    python etl_pipeline.py --deploy-bi     # fuerza el redespliegue de las vistas BI
    python etl_pipeline.py --reset         # DESTRUCTIVO: borra y recrea los esquemas del modelo

Códigos de salida: 0 = éxito, 1 = error o validación fallida, 2 = error de configuración/datos de entrada.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import polars as pl
import psycopg2
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from analytics_engine.config import RAW_DIR, REJECTED_DIR, SQL_DIR, DbSettings
from analytics_engine.db import run_sql_script
from analytics_engine.load import (
    MassDeleteError,
    data_changed,
    finish_run,
    load_star_schema,
    refresh_bi_layer,
    start_run,
)
from analytics_engine.transform import (
    SheetResult,
    SheetSchemaError,
    build_date_dimension,
    transform_customers,
    transform_products,
    transform_sales,
)
from analytics_engine.validate import Check, ValidationFailed, bi_checks, model_checks

SHEETS = {"productos": "productos.csv", "clientes": "clientes.csv", "ventas": "ventas.csv"}
log = logging.getLogger("etl")


class DataQualityError(RuntimeError):
    """Demasiadas filas rechazadas: se aborta antes de publicar datos."""


@contextmanager
def phase(name: str, timings: dict):
    log.info("> %s", name)
    start = time.perf_counter()
    yield
    timings[name] = round(time.perf_counter() - start, 3)
    log.info("  %s completada en %.2f s", name, timings[name])


def extract(source_dir: Path) -> dict[str, pl.DataFrame]:
    """Lee cada hoja con todas las columnas como texto (sin inferencia de tipos)."""
    frames = {}
    for sheet, filename in SHEETS.items():
        path = source_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"No se encuentra {path}. Genera los datos con: python generate_data.py")
        frames[sheet] = pl.read_csv(path, infer_schema=False, encoding="utf8")
        log.info("  %-10s %9s filas  (%s)", sheet, f"{frames[sheet].height:,}", path.name)
    return frames


def transform(frames: dict[str, pl.DataFrame]) -> tuple[SheetResult, SheetResult, SheetResult, pl.DataFrame]:
    products = transform_products(frames["productos"])
    customers = transform_customers(frames["clientes"])
    sales = transform_sales(frames["ventas"], products)
    if sales.clean.is_empty():
        raise DataQualityError("La hoja de ventas no contiene ninguna fila válida")
    dates = build_date_dimension(sales.clean["order_date"].min(), sales.clean["order_date"].max())
    for r in (products, customers, sales):
        m = r.metrics
        log.info("  %-10s válidas %9s | rechazadas %5s | vacías %3s | duplicados exactos %4s | "
                 "versiones antiguas %4s", r.sheet, f"{r.clean.height:,}", f"{r.rejected.height:,}",
                 m["blank_rows"], m["exact_duplicates"], m["key_duplicates_superseded"])
        if r.rejected.height:
            log.info("  %-10s motivos de rechazo: %s", "", r.rejected_by_reason)
    sm = sales.metrics
    log.info("  ventas     pedidos cancelados excluidos %s | precios imputados %s | canal desconocido %s",
             f"{sm['cancelled_lines_excluded']:,}", sm["prices_imputed"], sm["unknown_channel_rows"])
    log.info("  dim_date   %s días (%s -> %s)", f"{dates.height:,}", dates["full_date"].min(), dates["full_date"].max())
    return products, customers, sales, dates


def quality_gate(results: list[SheetResult], max_reject_rate: float) -> None:
    for r in results:
        m = r.metrics
        candidates = m["rows_read"] - m["blank_rows"] - m["exact_duplicates"] - m["key_duplicates_superseded"]
        rate = r.rejected.height / candidates if candidates else 0.0
        if rate > max_reject_rate:
            raise DataQualityError(
                f"Hoja '{r.sheet}': {rate:.2%} de filas rechazadas supera el umbral del {max_reject_rate:.0%}. "
                f"Revisa data/rejected/rechazos_{r.sheet}.csv")


def write_rejected_files(results: list[SheetResult], directory: Path) -> None:
    """CSV de rechazos por hoja: fila de la hoja, motivo y valores originales (para corregir en Sheets)."""
    directory.mkdir(parents=True, exist_ok=True)
    for r in results:
        path = directory / f"rechazos_{r.sheet}.csv"
        if r.rejected.is_empty():
            path.unlink(missing_ok=True)
            continue
        original = pl.from_dicts([json.loads(x) for x in r.rejected["record"].to_list()], infer_schema_length=None)
        header = r.rejected.select(pl.col("source_row").alias("fila_hoja"), pl.col("reason").alias("motivo"),
                                   pl.col("record_key").alias("clave"))
        header.hstack(original).write_csv(path)


def collect_metrics(products: SheetResult, customers: SheetResult, sales: SheetResult) -> dict:
    results = (products, customers, sales)
    return {
        "rows_extracted": sum(r.metrics["rows_read"] for r in results),
        "rows_rejected": sum(r.rejected.height for r in results),
        "rows_loaded": sales.clean.height,
        "sheets": {r.sheet: {**r.metrics, "rejected": r.rejected_by_reason} for r in results},
    }


def print_summary(run_id: int | None, status: str, metrics: dict, checks: list[Check], elapsed: float) -> None:
    line = "=" * 78
    print(f"\n{line}\nRESUMEN  ejecución #{run_id if run_id else '-'}  |  estado: {status.upper()}  |  {elapsed:.2f} s\n{line}")
    sheets = metrics["sheets"]
    print("Extraídas   : " + " | ".join(f"{k} {v['rows_read']:,}" for k, v in sheets.items()))
    print("Rechazadas  : " + " | ".join(f"{k} {sum(v['rejected'].values()):,}" for k, v in sheets.items())
          + "  -> audit.rejected_record y data/rejected/")
    load = metrics.get("load")
    if load:
        f = load["fact_sales"]
        print(f"Hechos      : {metrics['rows_loaded']:,} líneas en dw.fact_sales "
              f"(+{f['inserted']:,} nuevas, {f['updated']:,} actualizadas, {f['deleted']:,} eliminadas)")
        for dim in ("dim_customer", "dim_product", "dim_date"):
            print(f"{dim:<12}: +{load[dim]['inserted']:,} nuevas, {load[dim]['updated']:,} actualizadas")
    if metrics.get("bi"):
        print(f"Capa BI     : {metrics['bi']['action']}")
    if checks:
        print("Validaciones:")
        for c in checks:
            print(f"  [{'OK ' if c.passed else 'FALLO'}] {c.name:<26} {c.detail}")
    print(line)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-dir", type=Path, default=RAW_DIR, help="carpeta con los CSV exportados")
    p.add_argument("--dry-run", action="store_true", help="solo extracción y transformación (sin BD)")
    p.add_argument("--reset", action="store_true", help="DESTRUCTIVO: DROP de staging/dw/bi/audit antes de cargar")
    p.add_argument("--skip-bi", action="store_true", help="no desplegar/refrescar la capa BI")
    p.add_argument("--deploy-bi", action="store_true", help="forzar el redespliegue de sql/bi_views.sql")
    p.add_argument("--max-reject-rate", type=float, default=0.05,
                   help="máximo de filas rechazadas por hoja antes de abortar (defecto 0.05)")
    p.add_argument("--max-delete-ratio", type=float, default=0.25,
                   help="máximo de líneas de hechos que una carga puede eliminar (defecto 0.25)")
    p.add_argument("--rejected-dir", type=Path, default=REJECTED_DIR)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    started = time.perf_counter()
    timings: dict = {}
    manifest_path = args.source_dir / "_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None

    if args.dry_run:
        try:
            with phase("Extracción", timings):
                frames = extract(args.source_dir)
            with phase("Transformación", timings):
                products, customers, sales, _ = transform(frames)
            write_rejected_files([products, customers, sales], args.rejected_dir)
            quality_gate([products, customers, sales], args.max_reject_rate)
        except (FileNotFoundError, SheetSchemaError, DataQualityError) as exc:
            log.error("%s", exc)
            return 2
        print_summary(None, "dry-run", collect_metrics(products, customers, sales), [], time.perf_counter() - started)
        return 0

    settings = DbSettings.from_env()
    if not settings.password:
        log.error("Falta DW_PASSWORD: copia .env.example a .env y ejecuta setup_database.py")
        return 2
    engine = settings.create_engine()
    try:
        with phase("Esquema (sql/init.sql)", timings):
            if args.reset:
                log.warning("  --reset: eliminando esquemas staging, dw, bi y audit")
                with engine.begin() as conn:
                    conn.exec_driver_sql("DROP SCHEMA IF EXISTS bi, dw, staging, audit CASCADE")
            run_sql_script(engine, SQL_DIR / "init.sql")
    except OperationalError as exc:
        log.error("No se puede conectar a %s: %s", settings.describe(), exc.orig)
        engine.dispose()
        return 2
    except (SQLAlchemyError, psycopg2.Error) as exc:  # p. ej. permisos insuficientes al crear el esquema
        log.error("Error preparando el esquema en %s: %s", settings.describe(), getattr(exc, "orig", exc))
        engine.dispose()
        return 2

    run_id = start_run(engine, source=str(args.source_dir.resolve()))
    log.info("Ejecución #%s registrada en audit.etl_run (%s)", run_id, settings.describe())
    metrics: dict = {}
    checks: list[Check] = []
    try:
        with phase("Extracción", timings):
            frames = extract(args.source_dir)
        with phase("Transformación", timings):
            products, customers, sales, dates = transform(frames)
            metrics = collect_metrics(products, customers, sales)
            write_rejected_files([products, customers, sales], args.rejected_dir)
        quality_gate([products, customers, sales], args.max_reject_rate)
        if manifest:
            log.info("  manifiesto del generador detectado: se verificará el resultado esperado")
        with phase("Carga (COPY + merge + validación previa al COMMIT)", timings):
            metrics["load"] = load_star_schema(
                engine, run_id, products, customers, sales, dates, max_delete_ratio=args.max_delete_ratio,
                pre_commit_checks=lambda query: model_checks(query, products, customers, sales, manifest))
            checks = metrics["load"].pop("checks")
        if not args.skip_bi:
            with phase("Capa BI", timings):
                metrics["bi"] = refresh_bi_layer(engine, SQL_DIR / "bi_views.sql", force_deploy=args.deploy_bi,
                                                 changed=data_changed(metrics["load"]))
                log.info("  %s", metrics["bi"])
                checks += bi_checks(engine, sales)
        status = "success" if all(c.passed for c in checks) else "failed"
        metrics["timings_s"] = timings
        metrics["checks"] = [c.__dict__ for c in checks]
        failed = [c.name for c in checks if not c.passed]
        finish_run(engine, run_id, status, metrics, error=f"Validaciones fallidas: {failed}" if failed else None)
    except ValidationFailed as exc:  # ROLLBACK ya aplicado: no se ha publicado nada
        checks = exc.checks
        log.error("%s (se ha hecho ROLLBACK de la carga)", exc)
        finish_run(engine, run_id, "failed", {**metrics, "timings_s": timings,
                                              "checks": [c.__dict__ for c in checks]}, error=str(exc))
        print_summary(run_id, "failed", metrics, checks, time.perf_counter() - started)
        return 1
    except (SheetSchemaError, DataQualityError, MassDeleteError, FileNotFoundError) as exc:
        log.error("%s", exc)
        finish_run(engine, run_id, "failed", {**metrics, "timings_s": timings}, error=str(exc))
        return 2
    except Exception as exc:  # noqa: BLE001 - se registra cualquier fallo en la auditoría
        log.exception("Error no controlado en la ejecución #%s", run_id)
        finish_run(engine, run_id, "failed", {**metrics, "timings_s": timings}, error=repr(exc))
        return 1
    finally:
        engine.dispose()

    print_summary(run_id, status, metrics, checks, time.perf_counter() - started)
    return 0 if status == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
