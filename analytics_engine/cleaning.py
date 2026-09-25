"""Expresiones Polars reutilizables para limpiar datos exportados de hojas de cálculo.

Todas las funciones reciben y devuelven ``pl.Expr``, de modo que se componen
dentro de un único ``with_columns`` y Polars las ejecuta vectorizadas y en
paralelo, sin bucles Python por fila.

Convenciones de parseo (documentadas también en el README):

* Números: si el texto tiene forma es_ES (``1.299,99``, ``12,5``) se interpreta
  así; en otro caso como en_US (``1,299.99``). Casos ambiguos: ``1.299`` se lee
  como es_ES (1299) y ``1,299`` como en_US (1299), porque un importe monetario
  con 3 decimales no es plausible.
* Porcentajes: ``15%``, ``0.15``, ``0,15`` y ``15`` significan 15 %. Un número
  sin ``%`` mayor o igual que 1 se interpreta como porcentaje.
* Fechas: ISO (``2025-03-15``), ``dd/mm/aaaa``, ``dd-mm-aaaa``, ``aaaa/mm/dd``,
  ``dd.mm.aaaa``, con o sin hora, y números de serie de Google Sheets. Las
  fechas con barras se asumen día/mes (configuración regional es_ES).
"""

from __future__ import annotations

import re
import unicodedata

import polars as pl

# Marcadores que en una hoja significan "sin valor": celdas manuales y errores de fórmula.
NULL_TOKENS = [
    "", "-", "--", "n/a", "na", "n.a.", "null", "none", "nan", "s/d", "sin dato", "sin datos",
    "#n/a", "#ref!", "#value!", "#div/0!", "#name?", "#num!", "#null!", "#error!", "#error",
]

_ACCENTS = {
    "á": "a", "à": "a", "ä": "a", "â": "a", "é": "e", "è": "e", "ë": "e", "ê": "e",
    "í": "i", "ì": "i", "ï": "i", "î": "i", "ó": "o", "ò": "o", "ö": "o", "ô": "o",
    "ú": "u", "ù": "u", "ü": "u", "û": "u", "ñ": "n", "ç": "c",
}

# es_ES: miles con punto (primer grupo sin cero inicial) y/o coma decimal con 1-2 decimales
# (``1,299`` con 3 dígitos tras la coma es un millar en_US), o ``0,xxx``.
_ES_NUMBER = r"^-?[1-9]\d{0,2}(\.\d{3})+(,\d+)?$|^-?\d+,\d{1,2}$|^-?0,\d+$"
_DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d", "%d.%m.%Y")
# Google Sheets cuenta los días desde 1899-12-30; 25569 es el serial de 1970-01-01.
_SHEETS_EPOCH_OFFSET = 25_569
_EMAIL = r"^[a-z0-9._%+\-]+@[a-z0-9\-]+(\.[a-z0-9\-]+)*\.[a-z]{2,}$"
_TRUE = ["si", "s", "yes", "y", "true", "verdadero", "1", "x", "ok"]
_FALSE = ["no", "n", "false", "falso", "0"]


def fold_text(text: str) -> str:
    """Versión Python de :func:`fold` (para cabeceras y claves de diccionarios)."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text).strip().lower()


def clean_text(expr: pl.Expr) -> pl.Expr:
    """Colapsa espacios (incluido el espacio duro U+00A0 de Sheets) y anula los marcadores de vacío."""
    text = expr.cast(pl.String).str.replace_all(r"\s+", " ").str.strip_chars()
    return pl.when(text.str.to_lowercase().is_in(NULL_TOKENS)).then(None).otherwise(text)


def fold(expr: pl.Expr) -> pl.Expr:
    """Clave de comparación: texto limpio, en minúsculas y sin tildes."""
    return clean_text(expr).str.to_lowercase().str.replace_many(list(_ACCENTS), list(_ACCENTS.values()))


def fix_case(expr: pl.Expr) -> pl.Expr:
    """Pasa a Título los textos escritos TODO EN MAYÚSCULAS o todo en minúsculas.

    Los textos con mayúsculas mixtas se respetan (p. ej. ``NovaTech``)."""
    text = clean_text(expr)
    uniform = (text == text.str.to_uppercase()) | (text == text.str.to_lowercase())
    return pl.when(uniform).then(text.str.to_titlecase()).otherwise(text)


def _number_from_clean_text(text: pl.Expr) -> pl.Expr:
    """Número desde un texto YA limpio (evita repetir ``clean_text`` en expresiones compuestas)."""
    compact = text.str.replace_all(r"(?i)eur|usd|€|\$|\s", "")
    normalized = (
        pl.when(compact.str.contains(_ES_NUMBER))
        .then(compact.str.replace_all(".", "", literal=True).str.replace(",", ".", literal=True))
        .otherwise(compact.str.replace_all(",", "", literal=True))
    )
    return normalized.cast(pl.Float64, strict=False)


def parse_decimal(expr: pl.Expr) -> pl.Expr:
    """Número en formato es_ES o en_US, con o sin símbolo de moneda -> Float64 (nulo si no es válido)."""
    return _number_from_clean_text(clean_text(expr))


def to_cents(expr: pl.Expr) -> pl.Expr:
    """Importe -> céntimos enteros (Int64). Toda la aritmética monetaria posterior es exacta."""
    return (parse_decimal(expr) * 100).round(0).cast(pl.Int64)


def parse_int(expr: pl.Expr) -> pl.Expr:
    """Entero estricto: acepta ``2``, ``2.0`` o `` 2 ``; anula ``1.5``, ``dos`` o ``#VALUE!``."""
    value = parse_decimal(expr)
    return pl.when(value == value.round(0)).then(value.cast(pl.Int64))


def parse_percent_bp(expr: pl.Expr) -> pl.Expr:
    """Porcentaje -> puntos básicos enteros (1 % = 100 pb)."""
    text = clean_text(expr)
    value = _number_from_clean_text(text.str.replace_all("%", "", literal=True))
    fraction = pl.when(text.str.contains("%", literal=True) | (value >= 1)).then(value / 100).otherwise(value)
    return (fraction * 10_000).round(0).cast(pl.Int64)


def parse_date(expr: pl.Expr) -> pl.Expr:
    """Fecha en cualquiera de los formatos admitidos (ver docstring del módulo) -> Date."""
    text = clean_text(expr).str.replace(r"[ T]\d{1,2}:\d{2}(:\d{2}(\.\d+)?)?.*$", "")
    candidates = [text.str.strptime(pl.Date, fmt, strict=False) for fmt in _DATE_FORMATS]
    serial = text.cast(pl.Float64, strict=False)
    from_serial = pl.when(serial.is_between(20_000, 80_000)).then(
        (serial.floor().cast(pl.Int64) - _SHEETS_EPOCH_OFFSET).cast(pl.Int32).cast(pl.Date)
    )
    return pl.coalesce(*candidates, from_serial)


def parse_bool(expr: pl.Expr) -> pl.Expr:
    """Sí/No, TRUE/FALSE, 1/0, x... -> Boolean (nulo si no se reconoce)."""
    key = fold(expr)
    return pl.when(key.is_in(_TRUE)).then(True).when(key.is_in(_FALSE)).then(False).otherwise(None)


def parse_email(expr: pl.Expr) -> pl.Expr:
    """Email en minúsculas si es sintácticamente válido; nulo en otro caso.

    No se "reparan" espacios internos (``juan perez@...``): adivinar la dirección
    correcta es peor que marcarla como ausente."""
    email = clean_text(expr).str.to_lowercase()
    return pl.when(email.str.contains(_EMAIL)).then(email)


def normalize_code(expr: pl.Expr, pattern: str = r"^[A-Z0-9]+(-[A-Z0-9]+)*$") -> pl.Expr:
    """Código/ID en mayúsculas sin espacios; nulo si no respeta ``pattern``."""
    code = clean_text(expr).str.to_uppercase().str.replace_all(r"[\s_]", "")
    return pl.when(code.str.contains(pattern) & (code.str.len_chars() <= 20)).then(code)


def normalize_prefixed_id(expr: pl.Expr, width: int = 5) -> pl.Expr:
    """IDs tipo ``CLI-00042`` escritos como ``cli-00042``, `` CLI00042 `` o ``CLI-42`` -> ``CLI-00042``."""
    code = clean_text(expr).str.to_uppercase().str.replace_all(r"[\s_]", "")
    prefix = code.str.extract(r"^([A-Z]{2,5})-?\d{1,10}$", 1)
    number = code.str.extract(r"^[A-Z]{2,5}-?(\d{1,10})$", 1)
    return pl.when(prefix.is_not_null()).then(pl.concat_str(prefix, pl.lit("-"), number.str.zfill(width)))


def normalize_phone(expr: pl.Expr, dial_code: pl.Expr) -> pl.Expr:
    """Teléfono -> E.164 (``+34612345678``) usando el prefijo del país si falta; nulo si no es válido."""
    digits = clean_text(expr).str.replace_all(r"[^\d+]", "")
    e164 = (
        pl.when(digits.str.starts_with("+")).then(digits)
        .when(digits.str.starts_with("00")).then(pl.lit("+") + digits.str.slice(2))
        .otherwise(pl.lit("+") + dial_code + digits)
    )
    return pl.when(e164.str.contains(r"^\+\d{8,15}$")).then(e164)


def map_values(expr: pl.Expr, mapping: dict[str, str], default: str | None = None) -> pl.Expr:
    """Traduce variantes libres a valores canónicos comparando por clave normalizada (:func:`fold`)."""
    return fold(expr).replace_strict(mapping, default=default, return_dtype=pl.String)


def canonicalize_by_mode(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """Unifica variantes de escritura (``MÁLAGA``, ``malaga``, ``Málaga``) usando la grafía más frecuente.

    Canonicalización guiada por los datos: no necesita catálogos hard-codeados y
    se adapta a valores nuevos. Empates: gana el orden alfabético (determinista).
    Si la grafía ganadora está TODA en mayúsculas o minúsculas (el valor solo
    aparece "sucio"), se pasa a Título con :func:`fix_case`.
    """
    key = fold(pl.col(column))
    canonical = (
        df.select(key.alias("_key"), pl.col(column).alias("_value"))
        .drop_nulls("_key")
        .group_by("_key", "_value")
        .len()
        .sort(["_key", "len", "_value"], descending=[False, True, False])
        .unique("_key", keep="first", maintain_order=True)
        .select("_key", fix_case(pl.col("_value")).alias("_canonical"))
    )
    return (
        df.with_columns(key.alias("_key"))
        .join(canonical, on="_key", how="left", maintain_order="left")
        .with_columns(pl.col("_canonical").alias(column))
        .drop("_key", "_canonical")
    )
