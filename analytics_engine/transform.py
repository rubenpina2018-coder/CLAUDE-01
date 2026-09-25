"""Transformaciones por hoja: del CSV "sucio" exportado de Google Sheets a tablas limpias.

Cada ``transform_*`` devuelve un :class:`SheetResult` con:

* ``clean``    filas válidas, tipadas y normalizadas, listas para el staging;
* ``rejected`` filas descartadas con su número de fila en la hoja, el motivo y
  los valores originales (para que el responsable de la hoja pueda corregirlas);
* ``metrics``  contadores de calidad (duplicados, imputaciones, nulos...).

Orden de las reglas en todas las hojas:

1. Filas vacías y duplicados exactos (copiar/pegar) -> se descartan sin rechazo.
2. Duplicados por clave -> gana la ÚLTIMA fila de la hoja (versión vigente).
   Se deduplica antes de validar: si la versión vigente es inválida, el registro
   se rechaza en lugar de cargar silenciosamente una versión antigua.
3. Validación: la primera regla que falla determina el motivo del rechazo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import polars as pl

from analytics_engine.cleaning import (
    canonicalize_by_mode,
    clean_text,
    fix_case,
    fold,
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


class SheetSchemaError(ValueError):
    """La hoja no tiene las columnas obligatorias (o tiene columnas ambiguas)."""


@dataclass
class SheetResult:
    sheet: str
    clean: pl.DataFrame
    rejected: pl.DataFrame  # source_row, reason, record_key, record (JSON)
    metrics: dict = field(default_factory=dict)

    @property
    def rejected_by_reason(self) -> dict[str, int]:
        counts = self.rejected.group_by("reason").len().sort("reason")
        return dict(zip(counts["reason"].to_list(), counts["len"].to_list()))


# ---------------------------------------------------------------------------
# Diccionarios de referencia (claves normalizadas con ``fold``: minúsculas, sin tildes)
# ---------------------------------------------------------------------------

CHANNEL_CODES = {
    "online": "ONLINE", "web": "ONLINE", "tienda online": "ONLINE", "e-commerce": "ONLINE", "ecommerce": "ONLINE",
    "tienda fisica": "STORE", "tienda": "STORE", "fisica": "STORE", "retail": "STORE",
    "marketplace": "MARKETPLACE", "market place": "MARKETPLACE",
    "televenta": "PHONE", "telefono": "PHONE", "call center": "PHONE", "venta telefonica": "PHONE",
}
PAYMENT_METHODS = {
    "tarjeta": "Tarjeta", "tarjeta de credito": "Tarjeta", "tarjeta credito": "Tarjeta",
    "tarjeta de debito": "Tarjeta", "tarjeta debito": "Tarjeta", "visa/mastercard": "Tarjeta",
    "visa": "Tarjeta", "mastercard": "Tarjeta",
    "paypal": "PayPal",
    "transferencia": "Transferencia", "transferencia bancaria": "Transferencia", "transf.": "Transferencia",
    "efectivo": "Efectivo", "cash": "Efectivo", "metalico": "Efectivo",
    "contra reembolso": "Contra reembolso", "contrareembolso": "Contra reembolso",
    "contra-reembolso": "Contra reembolso", "cod": "Contra reembolso",
}
ORDER_STATUS = {
    "completado": "Completado", "entregado": "Completado", "enviado": "Completado", "cerrado": "Completado",
    "devuelto": "Devuelto", "devolucion": "Devuelto",
    "cancelado": "Cancelado", "anulado": "Cancelado",
}
SEGMENTS = {
    "particular": "Particular", "b2c": "Particular", "consumidor": "Particular",
    "pyme": "Pyme", "autonomo": "Pyme",
    "corporativo": "Corporativo", "gran cuenta": "Corporativo", "empresa": "Corporativo", "b2b": "Corporativo",
}
# clave normalizada -> (nombre canónico, ISO 3166-1 alfa-2, prefijo telefónico)
COUNTRIES = {
    **dict.fromkeys(["espana", "spain", "es", "esp"], ("España", "ES", "34")),
    **dict.fromkeys(["mexico", "mx", "mex"], ("México", "MX", "52")),
    **dict.fromkeys(["colombia", "co", "col"], ("Colombia", "CO", "57")),
    **dict.fromkeys(["argentina", "ar", "arg"], ("Argentina", "AR", "54")),
    **dict.fromkeys(["chile", "cl", "chl"], ("Chile", "CL", "56")),
    **dict.fromkeys(["peru", "pe", "per"], ("Perú", "PE", "51")),
    **dict.fromkeys(["portugal", "pt", "prt"], ("Portugal", "PT", "351")),
}
# "ISO país|región normalizada" -> ISO 3166-2 (enriquecimiento para mapas de Looker Studio)
REGION_ISO_CODES = {
    "ES|comunidad de madrid": "ES-MD", "ES|madrid": "ES-MD", "ES|cataluna": "ES-CT", "ES|catalunya": "ES-CT",
    "ES|comunidad valenciana": "ES-VC", "ES|andalucia": "ES-AN", "ES|pais vasco": "ES-PV",
    "ES|galicia": "ES-GA", "ES|castilla y leon": "ES-CL", "ES|aragon": "ES-AR", "ES|region de murcia": "ES-MC",
    "ES|murcia": "ES-MC", "ES|islas baleares": "ES-IB", "ES|illes balears": "ES-IB", "ES|canarias": "ES-CN",
    "ES|asturias": "ES-AS", "ES|cantabria": "ES-CB", "ES|navarra": "ES-NC", "ES|la rioja": "ES-RI",
    "ES|extremadura": "ES-EX", "ES|castilla-la mancha": "ES-CM",
    "MX|ciudad de mexico": "MX-CMX", "MX|jalisco": "MX-JAL", "MX|nuevo leon": "MX-NLE", "MX|puebla": "MX-PUE",
    "CO|bogota d.c.": "CO-DC", "CO|bogota": "CO-DC", "CO|antioquia": "CO-ANT", "CO|valle del cauca": "CO-VAC",
    "CO|atlantico": "CO-ATL",
    "AR|ciudad autonoma de buenos aires": "AR-C", "AR|caba": "AR-C", "AR|cordoba": "AR-X", "AR|santa fe": "AR-S",
    "AR|mendoza": "AR-M",
    "CL|region metropolitana": "CL-RM", "CL|valparaiso": "CL-VS", "CL|biobio": "CL-BI",
    "PE|lima": "PE-LIM", "PE|arequipa": "PE-ARE", "PE|la libertad": "PE-LAL",
}

# Cabeceras admitidas (normalizadas) -> nombre canónico de columna
PRODUCT_COLUMNS = {
    "sku": "sku", "codigo": "sku", "nombre producto": "product_name", "producto": "product_name",
    "categoria": "category", "subcategoria": "subcategory", "marca": "brand",
    "precio lista": "list_price", "pvp": "list_price", "coste unitario": "unit_cost", "coste": "unit_cost",
    "activo": "is_active", "fecha alta": "created_date",
}
PRODUCT_REQUIRED = {"sku", "product_name", "category", "subcategory", "list_price", "unit_cost"}

CUSTOMER_COLUMNS = {
    "id cliente": "customer_id", "nombre": "first_name", "apellidos": "last_name", "email": "email",
    "correo": "email", "telefono": "phone", "segmento": "segment", "ciudad": "city", "region": "region",
    "provincia": "region", "pais": "country", "fecha registro": "signup_date", "fecha alta": "signup_date",
    "acepta marketing": "marketing_opt_in",
}
CUSTOMER_REQUIRED = {"customer_id", "first_name", "country"}

SALES_COLUMNS = {
    "id pedido": "order_id", "linea": "order_line", "fecha pedido": "order_date", "fecha": "order_date",
    "id cliente": "customer_id", "sku": "sku", "canal": "channel", "metodo de pago": "payment_method",
    "cantidad": "quantity", "precio unitario": "unit_price", "descuento": "discount", "estado": "status",
}
SALES_REQUIRED = {"order_id", "order_line", "order_date", "sku", "quantity", "unit_price", "status"}

MIN_VALID_DATE, MAX_VALID_DATE = date(2000, 1, 1), date(2099, 12, 31)

# Columnas de salida (coinciden con las tablas staging.*)
PRODUCT_OUTPUT = ["sku", "product_name", "category", "subcategory", "brand", "list_price", "unit_cost",
                  "is_active", "created_date", "source_row"]
CUSTOMER_OUTPUT = ["customer_id", "first_name", "last_name", "full_name", "email", "phone", "segment", "city",
                   "region", "region_iso_code", "country", "country_iso2", "signup_date", "marketing_opt_in",
                   "source_row"]
SALES_OUTPUT = ["order_id", "order_line", "order_date", "date_key", "customer_id", "sku", "channel_code",
                "payment_method", "order_status", "quantity", "unit_price", "unit_cost", "discount_pct",
                "gross_amount", "discount_amount", "net_amount", "cost_amount", "profit_amount", "source_row"]
AMOUNT_COLUMNS = ["gross", "discount", "net", "cost", "profit"]


# ---------------------------------------------------------------------------
# Utilidades comunes
# ---------------------------------------------------------------------------


def _prepare(raw: pl.DataFrame, sheet: str, columns: dict[str, str], required: set[str]):
    """Numera las filas como en la hoja, descarta vacías y duplicados exactos y normaliza cabeceras."""
    raw = raw.with_row_index("source_row", offset=2).with_columns(pl.col("source_row").cast(pl.Int64))
    original = [c for c in raw.columns if c != "source_row"]
    rename: dict[str, str] = {}
    for col in original:
        target = columns.get(fold_text(col))
        if target is None:
            continue  # columnas no usadas (p. ej. "Notas")
        if target in rename.values():
            raise SheetSchemaError(f"Hoja '{sheet}': varias columnas corresponden a '{target}'")
        rename[col] = target
    missing = required - set(rename.values())
    if missing:
        raise SheetSchemaError(f"Hoja '{sheet}': faltan columnas obligatorias {sorted(missing)}; "
                               f"columnas encontradas: {original}")

    blank = pl.all_horizontal([clean_text(pl.col(c)).is_null() for c in original])
    non_blank = raw.filter(~blank)
    deduped = non_blank.unique(subset=original, keep="first", maintain_order=True)
    metrics = {
        "rows_read": raw.height,
        "blank_rows": raw.height - non_blank.height,
        "exact_duplicates": non_blank.height - deduped.height,
    }
    df = deduped.select("source_row", *[pl.col(src).alias(dst) for src, dst in rename.items()])
    absent = sorted(set(columns.values()) - set(rename.values()))
    if absent:  # columnas opcionales ausentes -> nulas
        df = df.with_columns([pl.lit(None, pl.String).alias(c) for c in absent])
    return df, raw, metrics


def _with_columns(df: pl.DataFrame, *exprs: pl.Expr, **named: pl.Expr) -> pl.DataFrame:
    """``with_columns`` ejecutado con el motor streaming de Polars.

    En modo eager una expresión elemento a elemento sobre un DataFrame de un solo
    bloque (lo habitual tras deduplicar) se evalúa en un único hilo y sin
    eliminar subexpresiones comunes. El motor streaming divide los datos en
    lotes (morsels) que procesa en paralelo: ~4x más rápido con 1M de filas.
    """
    return df.lazy().with_columns(*exprs, **named).collect(engine="streaming")


def _keep_last(df: pl.DataFrame, keys: list[str]) -> tuple[pl.DataFrame, int]:
    """Deduplica por clave quedándose con la última aparición en la hoja (versión vigente)."""
    has_key = pl.all_horizontal([pl.col(k).is_not_null() for k in keys])
    keyed = df.filter(has_key)
    latest = keyed.sort("source_row").unique(subset=keys, keep="last", maintain_order=True)
    return pl.concat([latest, df.filter(~has_key)]).sort("source_row"), keyed.height - latest.height


def _first_failing(rules: list[tuple[str, pl.Expr]]) -> pl.Expr:
    """Motivo de la primera regla que falla (o nulo). Las reglas deben ser seguras ante nulos."""
    reason = pl.lit(None, pl.String)
    for name, failed in reversed(rules):
        reason = pl.when(failed).then(pl.lit(name)).otherwise(reason)
    return reason


def _split(df: pl.DataFrame, rules: list[tuple[str, pl.Expr]], raw: pl.DataFrame, key: pl.Expr):
    """Separa filas válidas y rechazadas; los rechazos conservan los valores originales en JSON."""
    df = df.with_columns(_first_failing(rules).alias("_reason"))
    bad = df.filter(pl.col("_reason").is_not_null()).select(
        "source_row", pl.col("_reason").alias("reason"), key.cast(pl.String).alias("record_key"))
    original = [c for c in raw.columns if c != "source_row"]
    rejected = bad.join(
        raw.select("source_row", pl.struct(original).struct.json_encode().alias("record")),
        on="source_row", how="left",
    ).sort("source_row")
    return df.filter(pl.col("_reason").is_null()).drop("_reason"), rejected


def _count(df: pl.DataFrame, predicate: pl.Expr) -> int:
    return int(df.select(predicate.sum()).item() or 0)


# ---------------------------------------------------------------------------
# Hojas
# ---------------------------------------------------------------------------


def transform_products(raw: pl.DataFrame) -> SheetResult:
    df, raw, metrics = _prepare(raw, "productos", PRODUCT_COLUMNS, PRODUCT_REQUIRED)
    df = df.with_columns(sku=normalize_code(pl.col("sku")))
    df, metrics["key_duplicates_superseded"] = _keep_last(df, ["sku"])
    df = _with_columns(
        df,
        product_name=fix_case(pl.col("product_name")),
        category=clean_text(pl.col("category")),
        subcategory=clean_text(pl.col("subcategory")),
        brand=clean_text(pl.col("brand")),
        list_price_cents=to_cents(pl.col("list_price")),
        unit_cost_cents=to_cents(pl.col("unit_cost")),
        is_active=parse_bool(pl.col("is_active")),
        created_date=parse_date(pl.col("created_date")),
    )
    clean, rejected = _split(df, [
        ("sku_invalido", pl.col("sku").is_null()),
        ("atributos_obligatorios_vacios",
         pl.any_horizontal(pl.col("product_name").is_null(), pl.col("category").is_null(), pl.col("subcategory").is_null())),
        ("precio_invalido", pl.col("list_price_cents").is_null() | (pl.col("list_price_cents") <= 0)),
        ("coste_invalido", pl.col("unit_cost_cents").is_null() | (pl.col("unit_cost_cents") < 0)),
    ], raw, key=pl.col("sku"))
    for column in ("category", "subcategory", "brand"):
        clean = canonicalize_by_mode(clean, column)
    metrics["missing_brand"] = _count(clean, pl.col("brand").is_null())
    clean = clean.with_columns(
        brand=pl.col("brand").fill_null("Sin marca"),
        is_active=pl.col("is_active").fill_null(True),
        list_price=pl.col("list_price_cents") / 100,
        unit_cost=pl.col("unit_cost_cents") / 100,
    ).select(*PRODUCT_OUTPUT, "list_price_cents", "unit_cost_cents")
    metrics["valid_rows"] = clean.height
    return SheetResult("productos", clean, rejected, metrics)


def transform_customers(raw: pl.DataFrame) -> SheetResult:
    df, raw, metrics = _prepare(raw, "clientes", CUSTOMER_COLUMNS, CUSTOMER_REQUIRED)
    df = df.with_columns(customer_id=normalize_prefixed_id(pl.col("customer_id")))
    df, metrics["key_duplicates_superseded"] = _keep_last(df, ["customer_id"])
    country_key = fold(pl.col("country"))
    df = _with_columns(
        df,
        first_name=fix_case(pl.col("first_name")),
        last_name=fix_case(pl.col("last_name")),
        email_raw=clean_text(pl.col("email")),
        email=parse_email(pl.col("email")),
        segment=map_values(pl.col("segment"), SEGMENTS, default="Desconocido"),
        country_name=country_key.replace_strict({k: v[0] for k, v in COUNTRIES.items()}, default=None),
        country_iso2=country_key.replace_strict({k: v[1] for k, v in COUNTRIES.items()}, default=None),
        dial_code=country_key.replace_strict({k: v[2] for k, v in COUNTRIES.items()}, default=None),
        city=clean_text(pl.col("city")),
        region=clean_text(pl.col("region")),
        signup_raw=clean_text(pl.col("signup_date")),
        signup_date=parse_date(pl.col("signup_date")),
        marketing_opt_in=parse_bool(pl.col("marketing_opt_in")),
    ).with_columns(
        country=pl.coalesce(pl.col("country_name"), fix_case(pl.col("country")), pl.lit("Desconocido")),
        phone=normalize_phone(pl.col("phone"), pl.col("dial_code")),
        full_name=pl.concat_str(pl.col("first_name"), pl.col("last_name"), separator=" ", ignore_nulls=True),
    )
    clean, rejected = _split(df, [
        ("id_cliente_invalido", pl.col("customer_id").is_null()),
        ("nombre_vacio", pl.col("full_name").is_null() | (pl.col("full_name") == "")),
    ], raw, key=pl.col("customer_id"))
    metrics.update(
        invalid_emails=_count(clean, pl.col("email_raw").is_not_null() & pl.col("email").is_null()),
        missing_emails=_count(clean, pl.col("email_raw").is_null()),
        invalid_signup_dates=_count(clean, pl.col("signup_raw").is_not_null() & pl.col("signup_date").is_null()),
        unknown_countries=_count(clean, pl.col("country_iso2").is_null()),
        invalid_phones=_count(clean, pl.col("phone").is_null()),
    )
    for column in ("city", "region"):
        clean = canonicalize_by_mode(clean, column)
    clean = clean.with_columns(
        city=pl.col("city").fill_null("Desconocido"),
        region=pl.col("region").fill_null("Desconocido"),
        region_iso_code=pl.concat_str(pl.col("country_iso2"), pl.lit("|"), fold(pl.col("region")))
        .replace_strict(REGION_ISO_CODES, default=None),
    ).select(CUSTOMER_OUTPUT)
    metrics["valid_rows"] = clean.height
    return SheetResult("clientes", clean, rejected, metrics)


def transform_sales(raw: pl.DataFrame, products: SheetResult) -> SheetResult:
    df, raw, metrics = _prepare(raw, "ventas", SALES_COLUMNS, SALES_REQUIRED)
    df = _with_columns(
        df,
        order_id=normalize_code(pl.col("order_id"), pattern=r"^[A-Z]{2,5}-\d{1,12}$"),
        order_line=parse_int(pl.col("order_line")),
    ).with_columns(order_line=pl.when(pl.col("order_line") > 0).then(pl.col("order_line")))
    df, metrics["key_duplicates_superseded"] = _keep_last(df, ["order_id", "order_line"])
    df = _with_columns(
        df,
        order_date=parse_date(pl.col("order_date")),
        customer_id=normalize_prefixed_id(pl.col("customer_id")),
        sku=normalize_code(pl.col("sku")),
        channel_code=map_values(pl.col("channel"), CHANNEL_CODES, default="UNKNOWN"),
        payment_method=pl.when(clean_text(pl.col("payment_method")).is_null()).then(pl.lit("Desconocido"))
        .otherwise(map_values(pl.col("payment_method"), PAYMENT_METHODS, default="Otro")),
        quantity=parse_int(pl.col("quantity")),
        price_cents=to_cents(pl.col("unit_price")),
        discount_bp=parse_percent_bp(pl.col("discount")).fill_null(0),
        order_status=map_values(pl.col("status"), ORDER_STATUS, default=None),
    )
    catalog = products.clean.select("sku", "list_price_cents", "unit_cost_cents")
    df = df.join(catalog, on="sku", how="left", maintain_order="left")
    rejected_skus = products.rejected["record_key"].drop_nulls().unique().to_list()

    clean, rejected = _split(df, [
        ("clave_pedido_invalida", pl.col("order_id").is_null() | pl.col("order_line").is_null()),
        ("fecha_invalida", pl.col("order_date").is_null()
         | ~pl.col("order_date").is_between(MIN_VALID_DATE, MAX_VALID_DATE)),
        ("cantidad_invalida", pl.col("quantity").is_null() | (pl.col("quantity") <= 0)),
        ("producto_rechazado", pl.col("sku").is_in(rejected_skus).fill_null(False)),
        ("sku_desconocido", pl.col("sku").is_null() | pl.col("list_price_cents").is_null()),
        ("estado_desconocido", pl.col("order_status").is_null()),
        ("descuento_invalido", (pl.col("discount_bp") < 0) | (pl.col("discount_bp") >= 10_000)),
    ], raw, key=pl.concat_str(pl.col("order_id"), pl.lit("/"), pl.col("order_line").cast(pl.String)))

    # Precio ausente o ilegible -> se imputa el precio de tarifa del maestro de productos
    imputed = pl.col("price_cents").is_null() | (pl.col("price_cents") <= 0)
    metrics["prices_imputed"] = _count(clean, imputed)
    clean = clean.with_columns(
        unit_price_cents=pl.when(imputed).then(pl.col("list_price_cents")).otherwise(pl.col("price_cents")))

    # Regla de negocio: un pedido cancelado no es una venta (no entra en la tabla de hechos)
    metrics["cancelled_lines_excluded"] = _count(clean, pl.col("order_status") == "Cancelado")
    clean = clean.filter(pl.col("order_status") != "Cancelado")

    # Importes en céntimos enteros: aritmética exacta y redondeo comercial (mitad hacia arriba)
    gross = pl.col("quantity") * pl.col("unit_price_cents")
    clean = clean.with_columns(gross_cents=gross, discount_cents=(gross * pl.col("discount_bp") + 5_000) // 10_000,
                               cost_cents=pl.col("quantity") * pl.col("unit_cost_cents"))
    clean = clean.with_columns(net_cents=pl.col("gross_cents") - pl.col("discount_cents"))
    clean = clean.with_columns(profit_cents=pl.col("net_cents") - pl.col("cost_cents"))
    d = pl.col("order_date")
    clean = clean.with_columns(
        date_key=(d.dt.year() * 10_000 + d.dt.month().cast(pl.Int32) * 100 + d.dt.day().cast(pl.Int32)).cast(pl.Int32),
        unit_price=pl.col("unit_price_cents") / 100,
        unit_cost=pl.col("unit_cost_cents") / 100,
        discount_pct=pl.col("discount_bp") / 10_000,
        **{f"{name}_amount": pl.col(f"{name}_cents") / 100 for name in AMOUNT_COLUMNS},
    ).select(*SALES_OUTPUT, *[f"{name}_cents" for name in AMOUNT_COLUMNS])

    metrics.update(
        valid_rows=clean.height,
        unknown_channel_rows=_count(clean, pl.col("channel_code") == "UNKNOWN"),
        guest_rows=_count(clean, pl.col("customer_id").is_null()),
        returned_rows=_count(clean, pl.col("order_status") == "Devuelto"),
    )
    return SheetResult("ventas", clean, rejected, metrics)


# ---------------------------------------------------------------------------
# Dimensión fecha
# ---------------------------------------------------------------------------

MONTHS = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre",
          "Octubre", "Noviembre", "Diciembre"]
MONTHS_SHORT = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
WEEKDAYS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
WEEKDAYS_SHORT = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
FIXED_HOLIDAYS_ES = {
    (1, 1): "Año Nuevo", (1, 6): "Epifanía del Señor", (5, 1): "Fiesta del Trabajo",
    (8, 15): "Asunción de la Virgen", (10, 12): "Fiesta Nacional de España", (11, 1): "Todos los Santos",
    (12, 6): "Día de la Constitución", (12, 8): "Inmaculada Concepción", (12, 25): "Navidad",
}


def easter_sunday(year: int) -> date:
    """Domingo de Pascua (algoritmo gregoriano anónimo de Meeus/Jones/Butcher)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    lcoef = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * lcoef) // 451
    month, day = divmod(h + lcoef - 7 * m + 114, 31)
    return date(year, month, day + 1)


def spanish_holidays(first_year: int, last_year: int) -> dict[date, str]:
    holidays = {}
    for year in range(first_year, last_year + 1):
        holidays.update({date(year, m, d): name for (m, d), name in FIXED_HOLIDAYS_ES.items()})
        holidays[easter_sunday(year) - timedelta(days=2)] = "Viernes Santo"
    return holidays


def build_date_dimension(first: date, last: date) -> pl.DataFrame:
    """Calendario de años completos (1 ene del primer año - 31 dic del último), sin huecos."""
    start, end = date(first.year, 1, 1), date(last.year, 12, 31)
    holidays = spanish_holidays(start.year, end.year)
    holidays_df = pl.DataFrame({"full_date": list(holidays), "holiday_name": list(holidays.values())},
                               schema={"full_date": pl.Date, "holiday_name": pl.String})
    d = pl.col("full_date")
    months = dict(zip(range(1, 13), MONTHS))
    months_short = dict(zip(range(1, 13), MONTHS_SHORT))
    return (
        pl.DataFrame({"full_date": pl.date_range(start, end, "1d", eager=True)})
        .join(holidays_df, on="full_date", how="left", maintain_order="left")
        .with_columns(
            date_key=(d.dt.year() * 10_000 + d.dt.month().cast(pl.Int32) * 100 + d.dt.day().cast(pl.Int32)).cast(pl.Int32),
            year=d.dt.year().cast(pl.Int16),
            quarter=d.dt.quarter().cast(pl.Int16),
            quarter_name=pl.lit("T") + d.dt.quarter().cast(pl.String),
            year_quarter=d.dt.year().cast(pl.String) + pl.lit("-T") + d.dt.quarter().cast(pl.String),
            month=d.dt.month().cast(pl.Int16),
            month_name=d.dt.month().replace_strict(months, return_dtype=pl.String),
            month_short=d.dt.month().replace_strict(months_short, return_dtype=pl.String),
            year_month=d.dt.strftime("%Y-%m"),
            year_month_key=(d.dt.year() * 100 + d.dt.month().cast(pl.Int32)).cast(pl.Int32),
            year_month_label=d.dt.month().replace_strict(months_short, return_dtype=pl.String)
            + pl.lit(" ") + d.dt.year().cast(pl.String),
            month_start=d.dt.month_start(),
            month_end=d.dt.month_end(),
            iso_year=d.dt.iso_year().cast(pl.Int16),
            iso_week=d.dt.week().cast(pl.Int16),
            day_of_month=d.dt.day().cast(pl.Int16),
            day_of_year=d.dt.ordinal_day().cast(pl.Int16),
            day_of_week=d.dt.weekday().cast(pl.Int16),  # ISO: 1 = lunes
            day_name=d.dt.weekday().replace_strict(dict(zip(range(1, 8), WEEKDAYS)), return_dtype=pl.String),
            day_short=d.dt.weekday().replace_strict(dict(zip(range(1, 8), WEEKDAYS_SHORT)), return_dtype=pl.String),
            is_weekend=d.dt.weekday() >= 6,
            is_holiday=pl.col("holiday_name").is_not_null(),
        )
        .select(
            "date_key", "full_date", "year", "quarter", "quarter_name", "year_quarter", "month", "month_name",
            "month_short", "year_month", "year_month_key", "year_month_label", "month_start", "month_end",
            "iso_year", "iso_week", "day_of_month", "day_of_year", "day_of_week", "day_name", "day_short",
            "is_weekend", "is_holiday", "holiday_name",
        )
    )
