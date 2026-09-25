"""Tests unitarios de las expresiones de limpieza (sin base de datos)."""

from datetime import date

import polars as pl
import pytest

from analytics_engine.cleaning import (
    canonicalize_by_mode,
    clean_text,
    fix_case,
    fold_text,
    map_values,
    normalize_code,
    normalize_phone,
    normalize_prefixed_id,
    parse_bool,
    parse_date,
    parse_email,
    parse_int,
    parse_percent_bp,
    to_cents,
)


def apply(expr_fn, values, *extra_columns):
    """Aplica una expresión a una columna ``v`` (y columnas extra ``x0``, ``x1``...) y devuelve una lista."""
    data = {"v": values, **{f"x{i}": col for i, col in enumerate(extra_columns)}}
    extra = [pl.col(f"x{i}") for i in range(len(extra_columns))]
    return pl.DataFrame(data, schema_overrides={"v": pl.String}).select(
        expr_fn(pl.col("v"), *extra).alias("out"))["out"].to_list()


@pytest.mark.parametrize(("raw", "cents"), [
    ("1299.99", 129_999),
    ("1.299,99 €", 129_999),   # es_ES con espacio duro, como exporta Google Sheets
    ("€1,299.99", 129_999),
    ("1,299.99", 129_999),
    ("1299,99", 129_999),
    ("27,91 €", 2_791),
    (" 5.63 ", 563),
    ("1,299", 129_900),              # millar en_US
    ("1.299", 129_900),              # millar es_ES
    ("0,125", 12),
    ("-12,50", -1_250),
    ("#N/A", None),
    ("consultar", None),
    ("", None),
    (None, None),
])
def test_to_cents_handles_mixed_locales(raw, cents):
    assert apply(to_cents, [raw]) == [cents]


@pytest.mark.parametrize(("raw", "bp"), [
    ("15%", 1500), ("0.15", 1500), ("0,15", 1500), ("15", 1500), ("12.5%", 1250),
    ("0%", 0), ("0", 0), ("0.5", 5000), ("1", 100), ("", None), (None, None),
])
def test_parse_percent_to_basis_points(raw, bp):
    assert apply(parse_percent_bp, [raw]) == [bp]


@pytest.mark.parametrize(("raw", "expected"), [
    ("2024-03-15", date(2024, 3, 15)),
    ("15/03/2024", date(2024, 3, 15)),
    ("15-03-2024", date(2024, 3, 15)),
    ("2024-03-15 14:32:00", date(2024, 3, 15)),
    ("15/03/2024 09:05", date(2024, 3, 15)),
    ("45366", date(2024, 3, 15)),       # número de serie de Google Sheets
    ("2024/03/15", date(2024, 3, 15)),
    ("15.03.2024", date(2024, 3, 15)),
    ("5/3/2024", date(2024, 3, 5)),     # día/mes sin ceros
    ("31/02/2025", None),
    ("2025-13-01", None),
    ("00/00/0000", None),
    ("#REF!", None),
    ("2024", None),                     # un año suelto no es un serial plausible
    ("sin fecha", None),
    (None, None),
])
def test_parse_date_formats(raw, expected):
    assert apply(parse_date, [raw]) == [expected]


def test_parse_int_is_strict():
    assert apply(parse_int, ["2", "2.0", " 3 ", "1.5", "dos", "#VALUE!", "0", "-1", ""]) == \
        [2, 2, 3, None, None, None, 0, -1, None]


def test_parse_bool_variants():
    values = ["Sí", "si", "SI", "TRUE", "1", "x", "No", "no", "FALSE", "0", "", "quizá"]
    assert apply(parse_bool, values) == [True] * 6 + [False] * 4 + [None, None]


def test_clean_text_nulls_sheet_errors_and_collapses_spaces():
    assert apply(clean_text, ["  Hola  mundo ", "#N/A", "#REF!", "N/A", "-", "", "ok"]) == \
        ["Hola mundo", None, None, None, None, None, "ok"]


def test_ids_and_codes():
    assert apply(normalize_prefixed_id, ["CLI-00042", "cli-00042", " CLI-00042 ", "CLI00042", "CLI-42",
                                         "TOTAL CLIENTES", None]) == ["CLI-00042"] * 5 + [None, None]
    assert apply(normalize_code, ["sku-ele-0001", " SKU-ELE-0001", "#N/A", "SKU--1"]) == \
        ["SKU-ELE-0001", "SKU-ELE-0001", None, None]


def test_emails_are_lowercased_and_invalid_ones_nulled():
    assert apply(parse_email, ["ANA.GIL@MAIL.COM", " a.b@gmail.com ", "maria.garcia@", "juan perez@gmail.com",
                               "sin email"]) == ["ana.gil@mail.com", "a.b@gmail.com", None, None, None]


def test_phones_to_e164():
    raw = ["+34 612 345 678", "612345678", "612-345-678", "(+34) 612345678", "0034612345678", "N/A"]
    assert apply(normalize_phone, raw, ["34"] * len(raw)) == ["+34612345678"] * 5 + [None]


def test_fix_case_respects_mixed_case_brands():
    assert apply(fix_case, ["juan pérez", "ALBA & CO", "NovaTech", "  ana "]) == \
        ["Juan Pérez", "Alba & Co", "NovaTech", "Ana"]


def test_map_values_uses_folded_keys():
    mapping = {"online": "ONLINE", "tienda fisica": "STORE"}
    assert apply(lambda e: map_values(e, mapping, "UNKNOWN"), ["Online", "TIENDA FÍSICA", " tienda  física ",
                                                               "zzz", None]) == \
        ["ONLINE", "STORE", "STORE", "UNKNOWN", "UNKNOWN"]


def test_canonicalize_by_mode_picks_most_frequent_spelling():
    df = pl.DataFrame({"city": ["Málaga", "MALAGA", "málaga", "Malaga", "Málaga", "Vigo", None]})
    assert canonicalize_by_mode(df, "city")["city"].to_list() == ["Málaga"] * 5 + ["Vigo", None]


def test_fold_text_python_version():
    assert fold_text("  Método   de Pago ") == "metodo de pago"
