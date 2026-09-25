"""Tests del generador de datos sintéticos y del modo --dry-run del pipeline (sin base de datos)."""

import json

import etl_pipeline
import generate_data
from analytics_engine.transform import transform_customers, transform_products, transform_sales
from tests.conftest import read_sheet


def _manifest(directory):
    return json.loads((directory / "_manifest.json").read_text(encoding="utf-8"))


def test_generator_is_deterministic(tmp_path):
    args = ["--sales-rows", "2000", "--customers", "300", "--products", "40", "--seed", "123"]
    assert generate_data.main(["--output-dir", str(tmp_path / "a"), *args]) == 0
    assert generate_data.main(["--output-dir", str(tmp_path / "b"), *args]) == 0
    a, b = _manifest(tmp_path / "a"), _manifest(tmp_path / "b")
    assert a["files"] == b["files"]  # mismas filas y mismo sha256 de cada CSV
    assert a["expected_etl"] == b["expected_etl"]


def test_default_dataset_meets_volume_requirement(tmp_path):
    """Requisito del proyecto: al menos 50.000 registros de ventas (también tras la limpieza)."""
    assert generate_data.main(["--output-dir", str(tmp_path)]) == 0
    manifest = _manifest(tmp_path)
    assert manifest["files"]["ventas.csv"]["rows"] >= 50_000
    assert manifest["expected_etl"]["fact_rows"] >= 50_000
    assert manifest["files"]["clientes.csv"]["rows"] >= 8_000
    assert manifest["files"]["productos.csv"]["rows"] >= 450


def test_sheets_mimic_google_sheets_exports(small_dataset):
    ventas = read_sheet(small_dataset, "ventas")
    assert ventas.columns[:3] == ["ID Pedido", "Línea", "Fecha Pedido "]  # cabecera editada a mano
    assert all(dtype == "String" or str(dtype) == "String" for dtype in ventas.dtypes)
    prices = ventas["Precio Unitario"].drop_nulls()
    assert prices.str.contains("€").any() and prices.str.contains(",").any()  # formatos regionales mezclados
    assert ventas.filter(ventas["ID Pedido"] == "TOTAL").height == 1  # fila de totales al pie


def test_dirty_rate_zero_produces_clean_sheets(tmp_path):
    args = ["--sales-rows", "1500", "--customers", "200", "--products", "30", "--dirty-rate", "0"]
    assert generate_data.main(["--output-dir", str(tmp_path), *args]) == 0
    products = transform_products(read_sheet(tmp_path, "productos"))
    customers = transform_customers(read_sheet(tmp_path, "clientes"))
    sales = transform_sales(read_sheet(tmp_path, "ventas"), products)
    assert products.rejected.is_empty() and customers.rejected.is_empty() and sales.rejected.is_empty()
    assert sales.metrics["exact_duplicates"] == 0 and sales.metrics["blank_rows"] == 0
    assert sales.clean.height == _manifest(tmp_path)["expected_etl"]["fact_rows"]


def test_dry_run_needs_no_database(small_dataset, tmp_path):
    rejected_dir = tmp_path / "rejected"
    assert etl_pipeline.main(["--dry-run", "--source-dir", str(small_dataset),
                              "--rejected-dir", str(rejected_dir)]) == 0
    assert (rejected_dir / "rechazos_ventas.csv").is_file()


def test_missing_source_files_exit_with_code_2(tmp_path):
    assert etl_pipeline.main(["--dry-run", "--source-dir", str(tmp_path), "--rejected-dir", str(tmp_path)]) == 2
