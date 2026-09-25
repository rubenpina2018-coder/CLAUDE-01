"""Tests de las transformaciones por hoja y de la dimensión fecha (sin base de datos)."""

from datetime import date

import polars as pl
import pytest

from analytics_engine.transform import (
    SheetSchemaError,
    build_date_dimension,
    easter_sunday,
    transform_customers,
    transform_products,
    transform_sales,
)
from tests.conftest import read_sheet, sheet

PRODUCT_HEADER = ["SKU", "Nombre Producto", "Categoría", "Subcategoría", "Marca", "Precio Lista",
                  "Coste Unitario", "Activo", "Fecha Alta"]
CUSTOMER_HEADER = ["ID Cliente", "Nombre", "Apellidos", "Email", "Teléfono", "Segmento", "Ciudad", "Región",
                   "País", "Fecha Registro", "Acepta Marketing"]
# Cabecera con espacios y mayúsculas "de hoja real": el ETL debe normalizarla
SALES_HEADER = ["ID Pedido", "Línea", "Fecha Pedido ", "ID Cliente", "SKU", "canal", "Método de Pago",
                "Cantidad", "Precio Unitario", "Descuento", "Estado"]


@pytest.fixture
def products():
    return transform_products(sheet(PRODUCT_HEADER, [
        ["SKU-AAA-0001", "Producto uno", "Hogar", "Cocina", "Marca", "100.00", "60.00", "Sí", "2020-01-01"],
        ["sku-aaa-0002", "PRODUCTO DOS", "HOGAR", "Cocina", "Marca", "50,00 €", "#N/A", "Sí", "2020-01-01"],
        ["SKU-AAA-0003", "Producto tres", "hogar", "Cocina", "", "1.299,99 €", "850,10", "no", "45366"],
        ["SKU-AAA-0003", "Producto tres", "Hogar", "Cocina", "", "1.299,99 €", "850,10", "no", "45366"],
    ]))


def test_products_rules(products):
    clean = products.clean.sort("sku")
    assert clean["sku"].to_list() == ["SKU-AAA-0001", "SKU-AAA-0003"]
    assert products.rejected_by_reason == {"coste_invalido": 1}
    assert products.rejected["record_key"].to_list() == ["SKU-AAA-0002"]
    assert products.metrics["key_duplicates_superseded"] == 1  # SKU-AAA-0003 aparece dos veces
    assert clean["category"].to_list() == ["Hogar", "Hogar"]      # grafía más frecuente
    assert clean["brand"].to_list() == ["Marca", "Sin marca"]
    assert clean["list_price_cents"].to_list() == [10_000, 129_999]
    assert clean["created_date"].to_list() == [date(2020, 1, 1), date(2024, 3, 15)]
    assert clean["is_active"].to_list() == [True, False]


def test_sales_rules(products):
    raw = sheet(SALES_HEADER, [
        # 2: válida -> 2 x 100 € con 10 % de descuento
        ["PED-0000001", "1", "2025-01-10", "CLI-00001", "SKU-AAA-0001", "Online", "Tarjeta", "2", "100.00", "10%", "Completado"],
        # 3: duplicado exacto (copiar/pegar)
        ["PED-0000001", "1", "2025-01-10", "CLI-00001", "SKU-AAA-0001", "Online", "Tarjeta", "2", "100.00", "10%", "Completado"],
        # 4: invitado, precio vacío (se imputa la tarifa), variantes de texto
        ["ped-0000002", "1", "11/01/2025", "", "sku-aaa-0001", "tienda física", "efectivo", " 1 ", "", "", "entregado"],
        # 5: fecha imposible
        ["PED-0000003", "1", "31/02/2025", "CLI-00001", "SKU-AAA-0001", "Online", "Tarjeta", "1", "100", "", "Completado"],
        # 6: producto cuyo maestro está rechazado (coste #N/A)
        ["PED-0000004", "1", "2025-01-12", "CLI-00001", "SKU-AAA-0002", "Online", "Tarjeta", "1", "50", "", "Completado"],
        # 7: SKU inexistente
        ["PED-0000005", "1", "2025-01-12", "CLI-00001", "SKU-ZZZ-9999", "Online", "Tarjeta", "1", "50", "", "Completado"],
        # 8: pedido cancelado -> no es venta
        ["PED-0000006", "1", "2025-01-13", "CLI-00001", "SKU-AAA-0001", "Online", "Tarjeta", "1", "100", "", "Cancelado"],
        # 9 y 10: misma línea corregida -> gana la última versión
        ["PED-0000007", "1", "2025-01-14", "CLI-00001", "SKU-AAA-0001", "Online", "Tarjeta", "1", "100", "", "Completado"],
        ["PED-0000007", "1", "2025-01-14", "CLI-00001", "SKU-AAA-0001", "Online", "Tarjeta", "3", "100", "", "Devuelto"],
        # 11: redondeo comercial -> 3 x 1.299,99 con 15 % = 584,9955 -> 585,00
        ["PED-0000008", "1", "45672", "CLI-00002", "SKU-AAA-0003", "", "COD", "3", "1.299,99 €", "0,15", "Completado"],
        # 12: cantidad no numérica
        ["PED-0000009", "1", "2025-01-15", "CLI-00001", "SKU-AAA-0001", "Online", "Tarjeta", "#VALUE!", "100", "", "Completado"],
        # 13: estado desconocido
        ["PED-0000010", "1", "2025-01-15", "CLI-00001", "SKU-AAA-0001", "Online", "Tarjeta", "1", "100", "", "¿?"],
        # 14: fila de totales y 15: fila vacía
        ["TOTAL", "", "", "", "", "", "", "9", "", "", ""],
        [None] * len(SALES_HEADER),
    ])
    result = transform_sales(raw, products)

    assert result.metrics["blank_rows"] == 1
    assert result.metrics["exact_duplicates"] == 1
    assert result.metrics["key_duplicates_superseded"] == 1
    assert result.metrics["cancelled_lines_excluded"] == 1
    assert result.metrics["prices_imputed"] == 1
    assert result.rejected_by_reason == {
        "clave_pedido_invalida": 1, "fecha_invalida": 1, "producto_rechazado": 1,
        "sku_desconocido": 1, "cantidad_invalida": 1, "estado_desconocido": 1,
    }
    assert dict(zip(result.rejected["source_row"], result.rejected["reason"]))[5] == "fecha_invalida"

    rows = {r["order_id"]: r for r in result.clean.to_dicts()}
    assert set(rows) == {"PED-0000001", "PED-0000002", "PED-0000007", "PED-0000008"}
    first = rows["PED-0000001"]
    assert (first["gross_cents"], first["discount_cents"], first["net_cents"], first["cost_cents"],
            first["profit_cents"]) == (20_000, 2_000, 18_000, 12_000, 6_000)
    guest = rows["PED-0000002"]
    assert guest["customer_id"] is None and guest["unit_price"] == 100.0
    assert (guest["channel_code"], guest["payment_method"], guest["order_status"]) == ("STORE", "Efectivo", "Completado")
    assert guest["order_date"] == date(2025, 1, 11) and guest["date_key"] == 20250111
    assert rows["PED-0000007"]["quantity"] == 3 and rows["PED-0000007"]["order_status"] == "Devuelto"
    rounded = rows["PED-0000008"]
    assert rounded["discount_cents"] == 58_500 and rounded["net_cents"] == 389_997 - 58_500
    assert rounded["channel_code"] == "UNKNOWN" and rounded["payment_method"] == "Contra reembolso"
    assert rounded["order_date"] == date(2025, 1, 15)  # serial 45672 de Google Sheets


def test_customers_keep_latest_version_and_enrich():
    result = transform_customers(sheet(CUSTOMER_HEADER, [
        ["CLI-00001", "ANA", "garcía lópez", "ana.old@mail.com", "612345678", "particular", "Madrid",
         "Comunidad de Madrid", "ES", "2024-01-01", "Sí"],
        ["cli-00001", "Ana", "García López", "ANA@MAIL.COM", "612 345 678", "Pyme", "madrid",
         "comunidad de madrid", "España", "01/01/2024", "no"],
        ["CLI-00002", "Luis", "Pérez", "correo pendiente", "(+52) 5512345678", "Gran Cuenta", "Monterrey",
         "Nuevo León", "Mexico", "pendiente", ""],
        ["TOTAL CLIENTES", "2", None, None, None, None, None, None, None, None, None],
    ]))
    assert result.rejected_by_reason == {"id_cliente_invalido": 1}
    assert result.metrics["key_duplicates_superseded"] == 1
    assert result.metrics["invalid_emails"] == 1 and result.metrics["invalid_signup_dates"] == 1
    ana, luis = result.clean.sort("customer_id").to_dicts()
    assert (ana["full_name"], ana["email"], ana["segment"]) == ("Ana García López", "ana@mail.com", "Pyme")
    assert (ana["phone"], ana["country"], ana["country_iso2"], ana["region_iso_code"]) == \
        ("+34612345678", "España", "ES", "ES-MD")
    assert ana["city"] == "Madrid" and ana["marketing_opt_in"] is False
    assert (luis["email"], luis["segment"], luis["country"], luis["phone"], luis["region_iso_code"]) == \
        (None, "Corporativo", "México", "+525512345678", "MX-NLE")
    assert luis["signup_date"] is None


def test_missing_required_columns_raise_clear_error():
    with pytest.raises(SheetSchemaError, match="faltan columnas obligatorias"):
        transform_products(sheet(["SKU", "Nombre Producto"], [["SKU-1", "x"]]))


def test_date_dimension_full_years_and_holidays():
    dim = build_date_dimension(date(2024, 2, 10), date(2025, 3, 1))
    assert dim.height == 366 + 365
    assert (dim["full_date"].min(), dim["full_date"].max()) == (date(2024, 1, 1), date(2025, 12, 31))
    assert dim["date_key"].is_unique().all()
    good_friday = dim.filter(pl.col("holiday_name") == "Viernes Santo")["full_date"].to_list()
    assert good_friday == [date(2024, 3, 29), date(2025, 4, 18)]
    day = dim.filter(pl.col("full_date") == date(2025, 3, 15)).to_dicts()[0]
    assert (day["month_name"], day["day_name"], day["day_of_week"], day["is_weekend"]) == ("Marzo", "Sábado", 6, True)
    assert (day["year_month_label"], day["year_quarter"], day["iso_week"]) == ("Mar 2025", "2025-T1", 11)


@pytest.mark.parametrize(("year", "easter"), [(2023, date(2023, 4, 9)), (2024, date(2024, 3, 31)),
                                              (2025, date(2025, 4, 20)), (2026, date(2026, 4, 5))])
def test_easter_sunday(year, easter):
    assert easter_sunday(year) == easter


def test_generated_dataset_matches_manifest(small_dataset, manifest):
    """Las transformaciones reproducen exactamente el resultado que el generador espera."""
    products = transform_products(read_sheet(small_dataset, "productos"))
    customers = transform_customers(read_sheet(small_dataset, "clientes"))
    sales = transform_sales(read_sheet(small_dataset, "ventas"), products)
    exp = manifest["expected_etl"]

    assert products.clean.height == exp["dim_product_rows"]
    assert customers.clean.height == exp["dim_customer_rows"]
    assert sales.clean.height == exp["fact_rows"]
    assert sales.rejected_by_reason == {k: v for k, v in exp["rejected"].items() if v}
    assert int(sales.clean["quantity"].sum()) == exp["quantity"]
    for name in ("gross", "discount", "net", "cost", "profit"):
        assert int(sales.clean[f"{name}_cents"].sum()) == exp[f"{name}_amount_cents"], name
    sheet_stats = manifest["sheets"]["ventas"]
    for metric in ("blank_rows", "exact_duplicates", "key_duplicates_superseded", "prices_imputed"):
        assert sales.metrics[metric] == sheet_stats[metric], metric
    assert sales.metrics["cancelled_lines_excluded"] == exp["cancelled_lines_excluded"]
    customer_stats = manifest["sheets"]["clientes"]
    for metric in ("invalid_emails", "missing_emails", "invalid_signup_dates"):
        assert customers.metrics[metric] == customer_stats[metric], metric
