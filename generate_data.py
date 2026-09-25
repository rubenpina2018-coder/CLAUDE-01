#!/usr/bin/env python3
"""Generador de datos sintéticos que simulan exports CSV de Google Sheets.

Produce tres "pestañas" exportadas a CSV, tal y como las descargaría un equipo
comercial desde Google Sheets (Archivo > Descargar > CSV):

* ``productos.csv``  maestro de productos (SKU, categoría, precio, coste...)
* ``clientes.csv``   maestro de clientes (CRM mantenido a mano)
* ``ventas.csv``     líneas de pedido (>= 50.000 registros por defecto: 75.000)

Los datos son coherentes a nivel de negocio (estacionalidad, Black Friday,
crecimiento interanual, clientes que solo compran después de registrarse y que
abandonan con el tiempo) e incluyen la suciedad típica de una hoja de cálculo
editada por varias personas: formatos regionales mezclados (``1.299,99 €`` vs
``1,299.99``), fechas en varios formatos y como número de serie de Sheets,
mayúsculas/espacios inconsistentes, errores de fórmula (``#N/A``, ``#REF!``),
duplicados por copiar/pegar, filas de totales y filas en blanco.

Junto a los CSV se escribe ``_manifest.json`` con los parámetros, las anomalías
inyectadas y el resultado *esperado* del ETL (filas de hechos e importes al
céntimo), que los tests usan para validar el pipeline de extremo a extremo.

Uso::

    python generate_data.py                      # 75.000 líneas, semilla 42
    python generate_data.py --sales-rows 1000000 --customers 60000
    python generate_data.py --dirty-rate 0       # datos limpios
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl

from analytics_engine.config import RAW_DIR

SHEETS_EPOCH = date(1899, 12, 30)  # día 0 de los números de serie de Google Sheets

# ---------------------------------------------------------------------------
# Datos de referencia
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Category:
    name: str
    code: str
    weight: float  # peso en el catálogo
    return_rate: float  # tasa base de devolución
    brands: tuple[str, ...]
    # subcategoría -> (precio_min, precio_max, margen_min, margen_max, nombres)
    subcategories: dict[str, tuple[float, float, float, float, tuple[str, ...]]]


CATALOG: tuple[Category, ...] = (
    Category("Electrónica", "ELE", 0.25, 0.06, ("Voltix", "NovaTech", "Zentra", "Kairo", "Lumen"), {
        "Smartphones": (149, 1199, 0.12, 0.28, ("Smartphone", "Teléfono Móvil")),
        "Portátiles": (399, 2199, 0.08, 0.20, ("Portátil", "Ultrabook", "Convertible")),
        "Audio": (19, 349, 0.25, 0.45, ("Auriculares", "Altavoz Bluetooth", "Barra de Sonido")),
        "Accesorios Tech": (5, 79, 0.40, 0.65, ("Cargador USB-C", "Funda", "Cable HDMI", "Ratón Inalámbrico", "Teclado")),
    }),
    Category("Hogar", "HOG", 0.20, 0.04, ("CasaViva", "Hogarama", "Nórdika", "Terracota"), {
        "Cocina": (9, 399, 0.30, 0.50, ("Sartén", "Batidora", "Cafetera", "Juego de Cuchillos", "Freidora de Aire")),
        "Muebles": (59, 899, 0.30, 0.50, ("Silla", "Mesa Auxiliar", "Estantería", "Sofá", "Escritorio")),
        "Decoración": (7, 149, 0.45, 0.65, ("Lámpara", "Jarrón", "Espejo", "Cojín", "Marco de Fotos")),
    }),
    Category("Moda", "MOD", 0.25, 0.11, ("Alba & Co", "Urbano", "Mistral", "Brisa"), {
        "Ropa Mujer": (12, 179, 0.50, 0.70, ("Vestido", "Blusa", "Chaqueta", "Pantalón", "Falda")),
        "Ropa Hombre": (12, 159, 0.50, 0.70, ("Camisa", "Camiseta", "Pantalón", "Chaqueta", "Jersey")),
        "Calzado": (24, 219, 0.40, 0.60, ("Zapatillas", "Botas", "Sandalias", "Mocasines")),
    }),
    Category("Deportes", "DEP", 0.15, 0.05, ("Atlas Sport", "Cumbre", "Vértigo", "Pulso"), {
        "Fitness": (9, 599, 0.30, 0.50, ("Esterilla", "Mancuernas", "Banco de Pesas", "Cinta de Correr", "Banda Elástica")),
        "Ciclismo": (19, 1799, 0.20, 0.40, ("Bicicleta", "Casco", "Maillot", "Luces LED", "Candado")),
        "Outdoor": (14, 399, 0.35, 0.55, ("Tienda de Campaña", "Mochila", "Saco de Dormir", "Linterna", "Bastones")),
    }),
    Category("Belleza", "BEL", 0.08, 0.03, ("Aurea", "Botánica", "Nácar"), {
        "Cuidado Facial": (6, 89, 0.55, 0.75, ("Sérum", "Crema Hidratante", "Limpiador", "Mascarilla")),
        "Perfumería": (19, 149, 0.40, 0.60, ("Eau de Parfum", "Eau de Toilette", "Colonia")),
    }),
    Category("Juguetes", "JUG", 0.07, 0.03, ("Pequeñín", "Ludika", "Bloqi"), {
        "Juegos de Mesa": (9, 69, 0.35, 0.55, ("Juego de Mesa", "Puzzle", "Juego de Cartas")),
        "Construcción": (14, 249, 0.30, 0.50, ("Set de Construcción", "Kit de Robótica", "Bloques Magnéticos")),
    }),
)
MODEL_SUFFIXES = ("Pro", "Lite", "Max", "Plus", "Air", "Mini", "Ultra", "Classic", "Eco", "Sport", "Neo", "One")

# país -> (peso, prefijo telefónico, patrón del número nacional, {región: (peso, [(ciudad, peso)])})
GEOGRAPHY: dict[str, tuple[float, str, str, dict[str, tuple[float, list[tuple[str, float]]]]]] = {
    "España": (0.60, "34", "6########", {
        "Comunidad de Madrid": (0.22, [("Madrid", 0.8), ("Alcalá de Henares", 0.1), ("Móstoles", 0.1)]),
        "Cataluña": (0.18, [("Barcelona", 0.75), ("L'Hospitalet de Llobregat", 0.1), ("Girona", 0.15)]),
        "Comunidad Valenciana": (0.12, [("Valencia", 0.6), ("Alicante", 0.3), ("Castellón de la Plana", 0.1)]),
        "Andalucía": (0.15, [("Sevilla", 0.4), ("Málaga", 0.35), ("Granada", 0.15), ("Córdoba", 0.1)]),
        "País Vasco": (0.06, [("Bilbao", 0.6), ("San Sebastián", 0.25), ("Vitoria-Gasteiz", 0.15)]),
        "Galicia": (0.06, [("A Coruña", 0.4), ("Vigo", 0.4), ("Santiago de Compostela", 0.2)]),
        "Castilla y León": (0.05, [("Valladolid", 0.5), ("Salamanca", 0.3), ("Burgos", 0.2)]),
        "Aragón": (0.04, [("Zaragoza", 0.85), ("Huesca", 0.15)]),
        "Región de Murcia": (0.04, [("Murcia", 0.7), ("Cartagena", 0.3)]),
        "Islas Baleares": (0.03, [("Palma", 1.0)]),
        "Canarias": (0.04, [("Las Palmas de Gran Canaria", 0.55), ("Santa Cruz de Tenerife", 0.45)]),
        "Asturias": (0.03, [("Oviedo", 0.5), ("Gijón", 0.5)]),
    }),
    "México": (0.12, "52", "55########", {
        "Ciudad de México": (0.45, [("Ciudad de México", 1.0)]),
        "Jalisco": (0.25, [("Guadalajara", 0.7), ("Zapopan", 0.3)]),
        "Nuevo León": (0.20, [("Monterrey", 1.0)]),
        "Puebla": (0.10, [("Puebla", 1.0)]),
    }),
    "Colombia": (0.08, "57", "3#########", {
        "Bogotá D.C.": (0.50, [("Bogotá", 1.0)]),
        "Antioquia": (0.25, [("Medellín", 1.0)]),
        "Valle del Cauca": (0.15, [("Cali", 1.0)]),
        "Atlántico": (0.10, [("Barranquilla", 1.0)]),
    }),
    "Argentina": (0.07, "54", "911########", {
        "Ciudad Autónoma de Buenos Aires": (0.55, [("Buenos Aires", 1.0)]),
        "Córdoba": (0.20, [("Córdoba", 1.0)]),
        "Santa Fe": (0.15, [("Rosario", 1.0)]),
        "Mendoza": (0.10, [("Mendoza", 1.0)]),
    }),
    "Chile": (0.07, "56", "9########", {
        "Región Metropolitana": (0.65, [("Santiago", 1.0)]),
        "Valparaíso": (0.20, [("Valparaíso", 0.5), ("Viña del Mar", 0.5)]),
        "Biobío": (0.15, [("Concepción", 1.0)]),
    }),
    "Perú": (0.06, "51", "9########", {
        "Lima": (0.70, [("Lima", 1.0)]),
        "Arequipa": (0.18, [("Arequipa", 1.0)]),
        "La Libertad": (0.12, [("Trujillo", 1.0)]),
    }),
}

FIRST_NAMES = (
    "María", "Carmen", "Ana", "Laura", "Lucía", "Marta", "Elena", "Sofía", "Paula", "Isabel", "Cristina",
    "Raquel", "Pilar", "Valentina", "Camila", "Daniela", "Gabriela", "Mariana", "Fernanda", "Andrea",
    "Beatriz", "Rocío", "Nuria", "Silvia", "José", "Antonio", "Manuel", "Francisco", "David", "Juan",
    "Javier", "Daniel", "Carlos", "Miguel", "Alejandro", "Pablo", "Sergio", "Jorge", "Luis", "Alberto",
    "Santiago", "Mateo", "Sebastián", "Diego", "Andrés", "Felipe", "Tomás", "Ramón",
)
LAST_NAMES = (
    "García", "Rodríguez", "González", "Fernández", "López", "Martínez", "Sánchez", "Pérez", "Gómez",
    "Martín", "Jiménez", "Ruiz", "Hernández", "Díaz", "Moreno", "Muñoz", "Álvarez", "Romero", "Alonso",
    "Gutiérrez", "Navarro", "Torres", "Domínguez", "Vázquez", "Ramos", "Gil", "Ramírez", "Serrano",
    "Blanco", "Molina", "Morales", "Suárez", "Ortega", "Delgado", "Castro", "Ortiz", "Rubio", "Marín",
    "Sanz", "Iglesias", "Núñez", "Medina", "Garrido", "Castillo", "Cortés", "Vargas", "Rojas", "Herrera",
    "Silva", "Mendoza",
)
EMAIL_DOMAINS = ("gmail.com", "hotmail.com", "outlook.com", "yahoo.es", "icloud.com")
SEGMENT_PROPENSITY = {"Particular": 1.0, "Pyme": 1.6, "Corporativo": 2.2}

# Variantes "sucias" que aparecen en las hojas para cada valor canónico
CHANNEL_LABELS = {
    "ONLINE": ["Online", "online", "ONLINE", "Web", "Tienda Online", "E-commerce"],
    "STORE": ["Tienda Física", "Tienda Fisica", "tienda física", "TIENDA", "Tienda", "Retail"],
    "MARKETPLACE": ["Marketplace", "marketplace", "MarketPlace", "Market place"],
    "PHONE": ["Televenta", "televenta", "Teléfono", "Call Center", "TELEVENTA"],
}
PAYMENT_LABELS = {
    "CARD": ["Tarjeta", "tarjeta", "Tarjeta de crédito", "TARJETA CREDITO", "Visa/Mastercard"],
    "PAYPAL": ["PayPal", "paypal", "Paypal", "PAYPAL"],
    "TRANSFER": ["Transferencia", "transferencia bancaria", "Transf.", "TRANSFERENCIA"],
    "CASH": ["Efectivo", "efectivo", "Cash", "EFECTIVO"],
    "COD": ["Contra reembolso", "Contrareembolso", "contra-reembolso", "COD"],
}
STATUS_LABELS = {
    "COMPLETED": ["Completado", "completado", "COMPLETADO", "Entregado", "Enviado"],
    "RETURNED": ["Devuelto", "devuelto", "DEVUELTO", "Devolución"],
    "CANCELLED": ["Cancelado", "cancelado", "CANCELADO", "Anulado"],
}
SEGMENT_LABELS = {
    "Particular": ["Particular", "particular", "PARTICULAR", "B2C", "Consumidor"],
    "Pyme": ["Pyme", "PYME", "pyme", "PyME"],
    "Corporativo": ["Corporativo", "corporativo", "CORPORATIVO", "Gran Cuenta", "Empresa"],
}
COUNTRY_LABELS = {
    "España": ["España", "españa", "ESPAÑA", "Spain", "ES", "Espana"],
    "México": ["México", "Mexico", "MX", "méxico"],
    "Colombia": ["Colombia", "COLOMBIA", "CO"],
    "Argentina": ["Argentina", "ARGENTINA", "AR"],
    "Chile": ["Chile", "CHILE", "CL"],
    "Perú": ["Perú", "Peru", "PE", "PERÚ"],
}
BOOL_LABELS = {True: ["Sí", "si", "SI", "TRUE", "1", "x"], False: ["No", "no", "NO", "FALSE", "0"]}
NOTES = (
    "Cliente pide factura", "Entrega urgente", "Regalo, no incluir precio",
    'Cliente dice "llamar antes"', "Revisar dirección de envío", "Pedido telefónico, confirmar stock",
)

# Distribuciones de formato (reproducen la mezcla de configuraciones regionales)
MONEY_STYLE_P = (0.45, 0.25, 0.15, 0.10, 0.05)  # 1299.99 | 1.299,99 € | €1,299.99 | 1,299.99 | 1299,99
DATE_STYLE_P = (0.55, 0.25, 0.08, 0.07, 0.05)  # ISO | dd/mm/aaaa | dd-mm-aaaa | ISO+hora | serial Sheets
MONTH_SEASONALITY = np.array([1.05, 0.85, 0.90, 0.92, 0.95, 1.00, 1.05, 0.80, 0.90, 0.95, 1.35, 1.50])
WEEKDAY_FACTOR = np.array([1.00, 0.95, 0.95, 1.00, 1.10, 1.20, 0.90])  # lunes..domingo
YEARLY_GROWTH = 1.18


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def weighted_choice(rng: np.random.Generator, n: int, weights) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    return rng.choice(len(w), size=n, p=w / w.sum())


def pick(rng: np.random.Generator, options) -> str:
    return options[int(rng.integers(len(options)))]


def labels_for(rng, codes: np.ndarray, labels: dict, p_variant: float) -> np.ndarray:
    """Etiqueta canónica (primera de la lista) o, con probabilidad ``p_variant``, una variante."""
    out = np.empty(len(codes), dtype=object)
    variant = rng.random(len(codes)) < p_variant
    for code, options in labels.items():
        sel = codes == code
        out[sel & ~variant] = options[0]
        idx = np.flatnonzero(sel & variant)
        if idx.size:
            out[idx] = np.array(options, dtype=object)[rng.integers(0, len(options), idx.size)]
    return out


def case_noise(rng, values: np.ndarray, p_variant: float, p_spaces: float) -> np.ndarray:
    """Introduce variantes de mayúsculas/tildes y espacios sobrantes en textos libres."""
    out = values.astype(object).copy()
    for i in np.flatnonzero(rng.random(len(out)) < p_variant):
        v = out[i]
        if v is not None:
            out[i] = pick(rng, (v.lower(), v.upper(), strip_accents(v)))
    for i in np.flatnonzero(rng.random(len(out)) < p_spaces):
        v = out[i]
        if v is not None:
            out[i] = pick(rng, (f" {v}", f"{v} ", f"  {v}  ", v.replace(" ", "  ", 1)))
    return out


def format_money(cents: np.ndarray, styles: np.ndarray) -> np.ndarray:
    out = np.empty(len(cents), dtype=object)
    for i, (c, s) in enumerate(zip(cents.tolist(), styles.tolist())):
        units, cts = divmod(int(c), 100)
        if s == 0:
            out[i] = f"{units}.{cts:02d}"
        elif s == 1:  # es_ES: separador de miles "." y decimales ",", con espacio duro
            out[i] = f"{units:,}".replace(",", ".") + f",{cts:02d} €"
        elif s == 2:
            out[i] = f"€{units:,}.{cts:02d}"
        elif s == 3:
            out[i] = f"{units:,}.{cts:02d}"
        else:
            out[i] = f"{units},{cts:02d}"
    return out


def format_dates(rng, dates: list[date], styles: np.ndarray) -> np.ndarray:
    out = np.empty(len(dates), dtype=object)
    clock = rng.integers(8 * 3600, 23 * 3600, len(dates))
    for i, (d, s, secs) in enumerate(zip(dates, styles.tolist(), clock.tolist())):
        if s == 0:
            out[i] = d.isoformat()
        elif s == 1:
            out[i] = f"{d.day:02d}/{d.month:02d}/{d.year}"
        elif s == 2:
            out[i] = f"{d.day:02d}-{d.month:02d}-{d.year}"
        elif s == 3:
            out[i] = f"{d.isoformat()} {secs // 3600:02d}:{secs % 3600 // 60:02d}:{secs % 60:02d}"
        else:
            out[i] = str((d - SHEETS_EPOCH).days)
    return out


def format_bool(rng, values: np.ndarray, p_variant: float) -> np.ndarray:
    out = np.empty(len(values), dtype=object)
    variant = rng.random(len(values)) < p_variant
    for flag, options in BOOL_LABELS.items():
        sel = values == flag
        out[sel & ~variant] = options[0]
        idx = np.flatnonzero(sel & variant)
        out[idx] = np.array(options, dtype=object)[rng.integers(0, len(options), idx.size)]
    return out


def split_disjoint(rng, pool: np.ndarray, sizes: dict[str, int]) -> dict[str, np.ndarray]:
    """Reparte índices del ``pool`` en conjuntos disjuntos del tamaño pedido."""
    shuffled = rng.permutation(pool)
    out, start = {}, 0
    for name, size in sizes.items():
        size = min(size, max(len(shuffled) - start, 0))
        out[name] = np.sort(shuffled[start:start + size])
        start += size
    return out


def string_frame(data: dict, headers: list[str]) -> pl.DataFrame:
    """DataFrame de texto (como una hoja exportada). Los arrays ``object`` se pasan como listas:
    Polars no sabe inferir el tipo de un array ``object`` cuyo primer valor es ``None``."""
    return pl.DataFrame({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in data.items()},
                        schema={c: pl.String for c in headers})


def blank_frame(columns: list[str], n: int) -> pl.DataFrame:
    return pl.DataFrame({c: [None] * n for c in columns}, schema={c: pl.String for c in columns})


def take_rows(df: pl.DataFrame, idx) -> pl.DataFrame:
    """Selecciona filas por posición (``df[[]]`` seleccionaría columnas, no filas)."""
    idx = [int(i) for i in idx]
    return df[idx] if idx else df.clear()


def replace_rows(df: pl.DataFrame, idx: np.ndarray, new_rows: pl.DataFrame) -> pl.DataFrame:
    """Sustituye las filas ``idx`` de ``df`` por ``new_rows`` conservando la posición."""
    if idx.size == 0:
        return df
    indexed = df.with_row_index("_i")
    new_rows = new_rows.with_columns(pl.Series("_i", idx.tolist(), dtype=indexed["_i"].dtype)).select(indexed.columns)
    return pl.concat([indexed.filter(~pl.col("_i").is_in(idx.tolist())), new_rows]).sort("_i").drop("_i")


def overwrite(df: pl.DataFrame, rows: np.ndarray, column: str, values: list) -> pl.DataFrame:
    """Sobrescribe celdas concretas de una columna de texto."""
    if rows.size == 0:
        return df
    col = df[column].to_numpy().astype(object)
    col[rows] = values
    return df.with_columns(pl.Series(column, col.tolist(), dtype=pl.String))


def black_friday(year: int) -> date:
    first = date(year, 11, 1)
    first_thursday = 1 + (3 - first.weekday()) % 7
    return date(year, 11, first_thursday + 21 + 1)


# ---------------------------------------------------------------------------
# Generación de los datos "verdaderos" (limpios)
# ---------------------------------------------------------------------------


def generate_products(rng: np.random.Generator, n: int) -> pl.DataFrame:
    cat_idx = weighted_choice(rng, n, [c.weight for c in CATALOG])
    counters: dict[str, int] = defaultdict(int)
    records = []
    for ci in cat_idx:
        cat = CATALOG[ci]
        sub_name = pick(rng, list(cat.subcategories))
        pmin, pmax, mmin, mmax, nouns = cat.subcategories[sub_name]
        brand = pick(rng, cat.brands)
        price = math.exp(rng.uniform(math.log(pmin), math.log(pmax)))
        list_cents = int(price) * 100 + int(rng.choice([99, 95, 0], p=[0.6, 0.2, 0.2]))
        margin = rng.uniform(mmin, mmax)
        counters[cat.code] += 1
        records.append({
            "sku": f"SKU-{cat.code}-{counters[cat.code]:04d}",
            "product_name": f"{pick(rng, nouns)} {brand} {pick(rng, MODEL_SUFFIXES)} {rng.integers(10, 100)}",
            "category": cat.name,
            "subcategory": sub_name,
            "brand": brand,
            "list_price_cents": list_cents,
            "unit_cost_cents": max(1, round(list_cents * (1 - margin))),
            "is_active": bool(rng.random() < 0.94),
            "created_date": date(2018, 1, 1) + timedelta(days=int(rng.integers(0, 1826))),
            "return_rate": cat.return_rate,
        })
    df = pl.DataFrame(records)
    popularity = rng.lognormal(0.0, 1.0, n) * (df["list_price_cents"].to_numpy() / 100.0) ** -0.3
    return df.with_columns(popularity=pl.Series(popularity))


def generate_customers(rng: np.random.Generator, n: int, start: date, end: date) -> pl.DataFrame:
    total_days = (end - start).days + 1
    # 15 % llega con la campaña de lanzamiento; el resto crece con el negocio.
    launch = rng.random(n) < 0.15
    growth = np.floor(total_days * np.sqrt(rng.random(n)))
    uniform = np.floor(rng.random(n) * total_days)
    signup = np.where(launch, rng.integers(0, 60, n), np.where(rng.random(n) < 0.5, growth, uniform))
    signup = np.sort(np.minimum(signup.astype(np.int64), total_days - 1))  # IDs en orden de alta

    countries = list(GEOGRAPHY)
    c_idx = weighted_choice(rng, n, [GEOGRAPHY[c][0] for c in countries])
    country = np.array(countries, dtype=object)[c_idx]
    region = np.empty(n, dtype=object)
    city = np.empty(n, dtype=object)
    phone_nsn = np.empty(n, dtype=object)
    for ci, cname in enumerate(countries):
        rows = np.flatnonzero(c_idx == ci)
        _, _, pattern, regions = GEOGRAPHY[cname]
        rnames = list(regions)
        r_idx = weighted_choice(rng, rows.size, [regions[r][0] for r in rnames])
        for ri, rname in enumerate(rnames):
            rrows = rows[r_idx == ri]
            cities = regions[rname][1]
            region[rrows] = rname
            city[rrows] = np.array([c for c, _ in cities], dtype=object)[
                weighted_choice(rng, rrows.size, [w for _, w in cities])]
        digits = rng.integers(0, 10, (rows.size, pattern.count("#")))
        prefix = pattern.replace("#", "")
        phone_nsn[rows] = [prefix + "".join(map(str, d)) for d in digits]

    segment = np.array(["Particular", "Pyme", "Corporativo"], dtype=object)[weighted_choice(rng, n, [0.80, 0.14, 0.06])]
    first = np.array(FIRST_NAMES, dtype=object)[rng.integers(0, len(FIRST_NAMES), n)]
    compound = rng.random(n) < 0.10
    first[compound] = [f"{a} {b}" for a, b in zip(first[compound], np.array(FIRST_NAMES)[rng.integers(0, len(FIRST_NAMES), compound.sum())])]
    last1 = np.array(LAST_NAMES, dtype=object)[rng.integers(0, len(LAST_NAMES), n)]
    last2 = np.array(LAST_NAMES, dtype=object)[rng.integers(0, len(LAST_NAMES), n)]
    emails = []
    for i in range(n):
        user = f"{strip_accents(first[i].split()[0]).lower()}.{strip_accents(last1[i]).lower()}{rng.integers(1, 100)}"
        domain = pick(rng, EMAIL_DOMAINS) if segment[i] == "Particular" else f"{strip_accents(last2[i]).lower()}-{pick(rng, ('consulting', 'group', 'tech', 'retail'))}.com"
        emails.append(f"{user}@{domain}")

    propensity = rng.lognormal(0.0, 0.9, n) * np.array([SEGMENT_PROPENSITY[s] for s in segment])
    one_off = rng.random(n) < 0.25
    lifetime = np.where(one_off, rng.exponential(60, n), rng.exponential(900, n))
    return pl.DataFrame({
        "customer_id": [f"CLI-{i + 1:05d}" for i in range(n)],
        "first_name": first,
        "last_name": [f"{a} {b}" for a, b in zip(last1, last2)],
        "email": emails,
        "phone_cc": [GEOGRAPHY[c][1] for c in country],
        "phone_nsn": phone_nsn,
        "segment": segment,
        "city": city,
        "region": region,
        "country": country,
        "signup_day": signup,
        "signup_date": [start + timedelta(days=int(d)) for d in signup],
        "marketing_opt_in": rng.random(n) < 0.62,
        "propensity": propensity,
        "churn_day": signup + lifetime.astype(np.int64),
    }, strict=False)


def day_weights(rng, start: date, n_days: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    days = [start + timedelta(days=d) for d in range(n_days)]
    month = np.array([d.month for d in days])
    weekday = np.array([d.weekday() for d in days])
    years = np.arange(n_days) / 365.25
    weights = YEARLY_GROWTH ** years * MONTH_SEASONALITY[month - 1] * WEEKDAY_FACTOR[weekday]
    weights *= rng.lognormal(0.0, 0.12, n_days)
    promo_bf = np.zeros(n_days, dtype=bool)
    for year in range(start.year, days[-1].year + 1):
        bf = (black_friday(year) - start).days
        for k in range(-2, 4):  # jueves previo .. Cyber Monday
            if 0 <= bf + k < n_days:
                promo_bf[bf + k] = True
                weights[bf + k] *= 3.5 if k == 0 else 2.2
    day_of_year = np.array([d.timetuple().tm_yday for d in days])
    sales_season = ((month == 1) & (day_of_year >= 7)) | (month == 2) | (month == 7) | (month == 8)
    return weights, promo_bf, sales_season


def generate_sales(rng, n_lines: int, customers: pl.DataFrame, products: pl.DataFrame,
                   start: date, end: date) -> pl.DataFrame:
    total_days = (end - start).days + 1
    weights, promo_bf, sales_season = day_weights(rng, start, total_days)
    lines_p = np.array([0.55, 0.25, 0.12, 0.05, 0.03])
    n_orders = int(n_lines / float(np.dot(np.arange(1, 6), lines_p)) * 1.2) + 20

    # 1) Pedido de bienvenida: la mayoría de clientes compra en sus primeros 15 días.
    signup_day = customers["signup_day"].to_numpy()
    churn_day = customers["churn_day"].to_numpy()
    welcome_day = signup_day + rng.integers(0, 15, len(signup_day))
    welcome_cust = np.flatnonzero((rng.random(len(signup_day)) < 0.85) & (welcome_day < total_days))
    if welcome_cust.size > 0.6 * n_orders:  # datasets pequeños: no dominar la muestra
        welcome_cust = np.sort(rng.choice(welcome_cust, int(0.6 * n_orders), replace=False))
    welcome_days = welcome_day[welcome_cust]

    # 2) Recompras: fecha según estacionalidad; cliente ponderado por propensión
    #    entre los ya registrados y todavía no perdidos (churn).
    n_regular = n_orders - welcome_cust.size
    reg_days = rng.choice(total_days, size=n_regular, p=weights / weights.sum())
    order = np.argsort(signup_day, kind="stable")
    cum_w = np.cumsum(customers["propensity"].to_numpy()[order])
    eligible = np.searchsorted(signup_day[order], reg_days, side="right")
    reg_cust = np.full(n_regular, -1)
    pending = np.flatnonzero(eligible > 0)
    for _ in range(60):
        if pending.size == 0:
            break
        draw = rng.random(pending.size) * cum_w[eligible[pending] - 1]
        idx = np.minimum(np.searchsorted(cum_w, draw, side="right"), eligible[pending] - 1)
        ok = reg_days[pending] <= churn_day[order][idx]
        reg_cust[pending[ok]] = order[idx[ok]]
        pending = pending[~ok]
    if pending.size:  # sin candidatos activos: el último registrado
        reg_cust[pending] = order[eligible[pending] - 1]
    reg_cust[rng.random(n_regular) < 0.02] = -1  # compras como invitado (sin ID de cliente)

    # 3) Recorte aleatorio (no temporal) hasta el nº exacto de líneas y orden cronológico.
    o_day = np.concatenate([welcome_days, reg_days])
    o_cust = np.concatenate([welcome_cust, reg_cust])
    perm = rng.permutation(o_day.size)
    o_day, o_cust = o_day[perm], o_cust[perm]
    o_lines = rng.choice(np.arange(1, 6), size=o_day.size, p=lines_p)
    cut = int(np.searchsorted(np.cumsum(o_lines), n_lines))
    if cut >= o_lines.size:
        raise RuntimeError("No se generaron pedidos suficientes para el nº de líneas pedido")
    o_day, o_cust, o_lines = o_day[:cut + 1], o_cust[:cut + 1], o_lines[:cut + 1]
    o_lines[-1] -= int(o_lines.sum()) - n_lines
    chrono = np.lexsort((rng.random(o_day.size), o_day))
    o_day, o_cust, o_lines = o_day[chrono], o_cust[chrono], o_lines[chrono]
    n_ord = o_day.size

    # Atributos a nivel de pedido: canal (sin tiendas físicas fuera de España), pago, cancelación
    country = np.where(o_cust >= 0, customers["country"].to_numpy()[np.maximum(o_cust, 0)], "GUEST")
    is_es, is_guest = country == "España", country == "GUEST"
    years = o_day / 365.25
    p_online = np.select([is_es, is_guest], [0.46 + 0.03 * years, 0.35], default=0.68)
    p_store = np.select([is_es, is_guest], [0.32 - 0.03 * years, 0.60], default=0.0)
    p_market = np.select([is_es, is_guest], [0.16, 0.05], default=0.24)
    u = rng.random(n_ord)
    channel = np.select([u < p_online, u < p_online + p_store, u < p_online + p_store + p_market],
                        ["ONLINE", "STORE", "MARKETPLACE"], default="PHONE")
    payment_mix = {
        "ONLINE": (["CARD", "PAYPAL", "TRANSFER", "COD"], [0.58, 0.30, 0.05, 0.07]),
        "STORE": (["CARD", "CASH"], [0.72, 0.28]),
        "MARKETPLACE": (["CARD", "PAYPAL"], [0.65, 0.35]),
        "PHONE": (["CARD", "TRANSFER", "COD"], [0.45, 0.35, 0.20]),
    }
    payment = np.empty(n_ord, dtype=object)
    for ch, (codes, probs) in payment_mix.items():
        sel = np.flatnonzero(channel == ch)
        payment[sel] = np.array(codes, dtype=object)[weighted_choice(rng, sel.size, probs)]
    cancelled_order = rng.random(n_ord) < 0.025

    # Explosión a líneas
    li_order = np.repeat(np.arange(n_ord), o_lines)
    line_no = np.arange(n_lines) - np.repeat(np.cumsum(o_lines) - o_lines, o_lines) + 1
    li_day = o_day[li_order]
    li_cust = o_cust[li_order]
    li_channel = channel[li_order]
    segment = np.where(li_cust >= 0, customers["segment"].to_numpy()[np.maximum(li_cust, 0)], "Particular")

    prod = weighted_choice(rng, n_lines, products["popularity"].to_numpy())
    list_cents = products["list_price_cents"].to_numpy()[prod]
    qty = rng.choice([1, 2, 3, 4, 5], size=n_lines, p=[0.70, 0.19, 0.06, 0.03, 0.02])
    b2b = (segment != "Particular") & (rng.random(n_lines) < 0.5)
    qty = np.where(b2b, 1 + rng.poisson(3, n_lines), qty)
    qty = np.where(list_cents > 60000, np.where(rng.random(n_lines) < 0.9, 1, 2), qty)
    qty = np.clip(qty, 1, 25)

    # Precio de venta: tarifa ajustada por año (+3 %/año) y ajustes puntuales
    day_year = np.array([(start + timedelta(days=d)).year for d in range(total_days)])
    year = day_year[li_day]
    price_factor = (1.0 + 0.03 * (year - 2025)) * np.where(rng.random(n_lines) < 0.2, rng.normal(1.0, 0.03, n_lines), 1.0)
    price_cents = np.maximum(np.round(list_cents * price_factor), 1).astype(np.int64)

    # Descuentos en puntos básicos (1 % = 100 pb) según campaña
    in_bf, in_sales = promo_bf[li_day], sales_season[li_day]
    has_disc = rng.random(n_lines) < np.select([in_bf, in_sales], [0.80, 0.50], default=0.22)
    normal = np.array([500, 1000, 1500, 2000, 2500, 3000])[weighted_choice(rng, n_lines, [.25, .30, .20, .12, .08, .05])]
    rebajas = np.array([1000, 1500, 2000, 2500, 3000, 4000, 5000])[weighted_choice(rng, n_lines, [.10, .15, .20, .20, .15, .12, .08])]
    bf_disc = np.array([1500, 2000, 2500, 3000, 4000, 5000])[weighted_choice(rng, n_lines, [.10, .20, .25, .20, .15, .10])]
    discount_bp = np.where(has_disc, np.select([in_bf, in_sales], [bf_disc, rebajas], default=normal), 0)
    negotiated = (segment == "Corporativo") & (discount_bp == 0) & (rng.random(n_lines) < 0.35)
    discount_bp = np.where(negotiated, 1000, discount_bp)

    # Estado: cancelación a nivel de pedido, devolución a nivel de línea
    ret_factor = np.select([li_channel == "ONLINE", li_channel == "MARKETPLACE", li_channel == "STORE"], [1.3, 1.4, 0.6], 1.0)
    returned = rng.random(n_lines) < products["return_rate"].to_numpy()[prod] * ret_factor
    status = np.where(cancelled_order[li_order], "CANCELLED", np.where(returned, "RETURNED", "COMPLETED"))

    cust_ids = customers["customer_id"].to_numpy()
    return pl.DataFrame({
        "order_id": [f"PED-{i + 1:07d}" for i in li_order],
        "order_line": line_no,
        "order_day": li_day,
        "order_date": pl.Series(np.datetime64(start, "D") + li_day.astype("timedelta64[D]")).cast(pl.Date),
        "customer_idx": li_cust,
        "customer_id": [cust_ids[c] if c >= 0 else None for c in li_cust],
        "product_idx": prod,
        "sku": products["sku"].to_numpy()[prod],
        "channel_code": li_channel,
        "payment_code": payment[li_order],
        "quantity": qty,
        "unit_price_cents": price_cents,
        "discount_bp": discount_bp,
        "status_code": status,
    }, strict=False)


# ---------------------------------------------------------------------------
# "Ensuciado": construcción de las pestañas tal y como se exportan de Sheets
# ---------------------------------------------------------------------------

PRODUCT_HEADERS = ["SKU", "Nombre Producto", "Categoría", "Subcategoría", "Marca",
                   "Precio Lista", "Coste Unitario", "Activo", "Fecha Alta"]
CUSTOMER_HEADERS = ["ID Cliente", "Nombre", "Apellidos", "Email", "Teléfono", "Segmento",
                    "Ciudad", "Región", "País", "Fecha Registro", "Acepta Marketing"]
# Nota: "Fecha Pedido " lleva un espacio final, como ocurre a menudo en cabeceras editadas a mano.
SALES_HEADERS = ["ID Pedido", "Línea", "Fecha Pedido ", "ID Cliente", "SKU", "Canal", "Método de Pago",
                 "Cantidad", "Precio Unitario", "Descuento", "Estado", "Notas"]


def build_products_sheet(rng, products: pl.DataFrame, rate: float) -> tuple[pl.DataFrame, dict, set[str]]:
    n = products.height
    n_invalid = 2 if rate > 0 else 0
    # Los productos con coste ilegible son los menos vendidos (bloquean pocas ventas).
    invalid_idx = np.argsort(products["popularity"].to_numpy())[:n_invalid]
    stale_pool = np.setdiff1d(np.arange(n), invalid_idx)
    stale_idx = np.sort(rng.choice(stale_pool, int(round(n * 0.02 * rate)), replace=False))

    def render(df: pl.DataFrame, noisy: bool) -> pl.DataFrame:
        k = df.height
        r = rate if noisy else 0.0
        return string_frame({
            "SKU": case_noise(rng, df["sku"].to_numpy(), 0.03 * r, 0.02 * r),
            "Nombre Producto": case_noise(rng, df["product_name"].to_numpy(), 0.03 * r, 0.05 * r),
            "Categoría": case_noise(rng, df["category"].to_numpy(), 0.15 * r, 0.03 * r),
            "Subcategoría": case_noise(rng, df["subcategory"].to_numpy(), 0.10 * r, 0.03 * r),
            "Marca": case_noise(rng, df["brand"].to_numpy(), 0.08 * r, 0.03 * r),
            "Precio Lista": format_money(df["list_price_cents"].to_numpy(), weighted_choice(rng, k, MONEY_STYLE_P) if noisy else np.zeros(k, int)),
            "Coste Unitario": format_money(df["unit_cost_cents"].to_numpy(), weighted_choice(rng, k, MONEY_STYLE_P) if noisy else np.zeros(k, int)),
            "Activo": format_bool(rng, df["is_active"].to_numpy(), 0.3 * r),
            "Fecha Alta": format_dates(rng, df["created_date"].to_list(), weighted_choice(rng, k, DATE_STYLE_P) if noisy else np.zeros(k, int)),
        }, PRODUCT_HEADERS)

    body = render(products, noisy=True)
    # Coste ilegible (error de fórmula y celda vacía): producto no cargable
    body = overwrite(body, invalid_idx, "Coste Unitario", ["#N/A", None][:n_invalid])
    # Versión antigua (precio/coste -10 %) en la posición original y versión vigente añadida
    # al final de la hoja: el ETL debe quedarse con la última aparición de cada SKU.
    stale = take_rows(products, stale_idx).with_columns(
        (pl.col("list_price_cents") * 0.9).round().cast(pl.Int64),
        (pl.col("unit_cost_cents") * 0.9).round().cast(pl.Int64))
    body = replace_rows(body, stale_idx, render(stale, noisy=True))
    n_blank = 3 if rate > 0 else 0
    sheet = pl.concat([body, render(take_rows(products, stale_idx), noisy=False), blank_frame(PRODUCT_HEADERS, n_blank)])
    stats = {
        "rows_in_sheet": sheet.height,
        "blank_rows": n_blank,
        "stale_versions_superseded": int(stale_idx.size),
        "rejected": {"coste_invalido": n_invalid},
        "expected_products": n - n_invalid,
    }
    return sheet, stats, set(products["sku"].to_numpy()[invalid_idx].tolist())


def build_customers_sheet(rng, customers: pl.DataFrame, rate: float) -> tuple[pl.DataFrame, dict]:
    n = customers.height

    def render(df: pl.DataFrame, noisy: bool) -> pl.DataFrame:
        k = df.height
        r = rate if noisy else 0.0
        ids = df["customer_id"].to_numpy().astype(object)
        for i in np.flatnonzero(rng.random(k) < 0.03 * r):
            ids[i] = pick(rng, (ids[i].lower(), f" {ids[i]}", ids[i].replace("-", "")))
        phones = np.empty(k, dtype=object)
        style = weighted_choice(rng, k, [0.40, 0.25, 0.15, 0.10, 0.10]) if noisy else np.zeros(k, int)
        for i, (cc, nsn, s) in enumerate(zip(df["phone_cc"].to_list(), df["phone_nsn"].to_list(), style.tolist())):
            groups = " ".join(nsn[j:j + 3] for j in range(0, len(nsn), 3))
            phones[i] = (f"+{cc} {groups}", nsn, groups.replace(" ", "-"), f"(+{cc}) {nsn}", f"00{cc}{nsn}")[s]
        return string_frame({
            "ID Cliente": ids,
            "Nombre": case_noise(rng, df["first_name"].to_numpy(), 0.12 * r, 0.05 * r),
            "Apellidos": case_noise(rng, df["last_name"].to_numpy(), 0.12 * r, 0.05 * r),
            "Email": [e.upper() if rng.random() < 0.05 * r else (f" {e} " if rng.random() < 0.03 * r else e)
                      for e in df["email"].to_list()],
            "Teléfono": phones,
            "Segmento": labels_for(rng, df["segment"].to_numpy(), SEGMENT_LABELS, 0.15 * r),
            "Ciudad": case_noise(rng, df["city"].to_numpy(), 0.08 * r, 0.03 * r),
            "Región": case_noise(rng, df["region"].to_numpy(), 0.08 * r, 0.03 * r),
            "País": labels_for(rng, df["country"].to_numpy(), COUNTRY_LABELS, 0.15 * r),
            "Fecha Registro": format_dates(rng, df["signup_date"].to_list(), weighted_choice(rng, k, DATE_STYLE_P) if noisy else np.zeros(k, int)),
            "Acepta Marketing": format_bool(rng, df["marketing_opt_in"].to_numpy(), 0.3 * r),
        }, CUSTOMER_HEADERS)

    body = render(customers, noisy=True)
    # Emails inválidos o vacíos y fechas de alta ilegibles (el ETL los deja a NULL)
    idx = split_disjoint(rng, np.arange(n), {
        "bad_email": int(round(n * 0.012 * rate)), "no_email": int(round(n * 0.008 * rate)),
        "bad_signup": int(round(n * 0.005 * rate)), "stale": int(round(n * 0.01 * rate))})
    bad_emails = ("sin email", "N/A", "maria.garcia@", "correo pendiente", "juan perez@gmail.com")
    body = overwrite(body, idx["bad_email"], "Email", [pick(rng, bad_emails) for _ in idx["bad_email"]])
    body = overwrite(body, idx["no_email"], "Email", [None] * idx["no_email"].size)
    body = overwrite(body, idx["bad_signup"], "Fecha Registro", ["#REF!"] * idx["bad_signup"].size)
    # Versión antigua (email y segmento desactualizados) en la posición original; la vigente, al final.
    stale_idx = idx["stale"]
    stale_src = take_rows(customers, stale_idx).with_columns(
        pl.col("email").str.replace("@", ".old@"), pl.lit("Particular").alias("segment"))
    body = replace_rows(body, stale_idx, render(stale_src, noisy=True))
    finals = render(take_rows(customers, stale_idx), noisy=False)
    # Filas basura habituales: un alta sin ID y una fila de totales al pie
    junk = string_frame({"ID Cliente": [None, "TOTAL CLIENTES"], "Nombre": ["Pedro", str(n)], "Apellidos": ["Sin Alta", None],
                         "Email": ["pedro@gmail.com", None], "Teléfono": [None, None], "Segmento": ["Particular", None],
                         "Ciudad": ["Madrid", None], "Región": ["Comunidad de Madrid", None], "País": ["España", None],
                         "Fecha Registro": [None, None], "Acepta Marketing": [None, None]},
                        CUSTOMER_HEADERS)
    n_blank = 5 if rate > 0 else 0
    sheet = pl.concat([body, finals] + ([junk, blank_frame(CUSTOMER_HEADERS, n_blank)] if rate > 0 else []))
    stats = {
        "rows_in_sheet": sheet.height,
        "blank_rows": n_blank,
        "stale_versions_superseded": int(stale_idx.size),
        "invalid_emails": int(idx["bad_email"].size),
        "missing_emails": int(idx["no_email"].size),
        "invalid_signup_dates": int(idx["bad_signup"].size),
        "rejected": {"id_cliente_invalido": 2 if rate > 0 else 0},
        "expected_customers": n,
    }
    return sheet, stats


def build_sales_sheet(rng, sales: pl.DataFrame, products: pl.DataFrame, n_customers: int,
                      invalid_skus: set[str], rate: float) -> tuple[pl.DataFrame, dict, dict]:
    n = sales.height
    status = sales["status_code"].to_numpy()
    guest = sales["customer_id"].is_null().to_numpy()
    product_valid = ~sales["sku"].is_in(sorted(invalid_skus)).to_numpy()
    pool = np.flatnonzero((status != "CANCELLED") & product_valid)

    def k(p: float) -> int:
        return int(round(n * p * rate))

    orphan = split_disjoint(rng, np.intersect1d(pool, np.flatnonzero(~guest)), {"orphan": k(0.003)})["orphan"]
    anomalies = split_disjoint(rng, np.setdiff1d(pool, orphan), {
        "bad_date": k(0.0025), "bad_qty": k(0.0015), "unknown_sku": k(0.002), "bad_status": k(0.001),
        "missing_price": k(0.004), "empty_channel": k(0.004), "exact_dup": k(0.01), "edited": k(0.002)})
    anomalies["orphan"] = orphan

    # Verdad final (lo que el ETL debería cargar): precio imputado desde tarifa y cantidad editada
    list_price = dict(zip(products["sku"].to_list(), products["list_price_cents"].to_list()))
    unit_cost = dict(zip(products["sku"].to_list(), products["unit_cost_cents"].to_list()))
    price = sales["unit_price_cents"].to_numpy().copy()
    skus = sales["sku"].to_numpy()
    price[anomalies["missing_price"]] = [list_price[s] for s in skus[anomalies["missing_price"]]]
    qty = sales["quantity"].to_numpy().copy()
    qty_original = qty.copy()
    qty[anomalies["edited"]] += 1

    r = rate
    order_ids = sales["order_id"].to_numpy().astype(object)
    for i in np.flatnonzero(rng.random(n) < 0.005 * r):
        order_ids[i] = order_ids[i].lower()
    cust = sales["customer_id"].to_numpy().astype(object)
    for i in np.flatnonzero((rng.random(n) < 0.05 * r) & ~guest):
        cust[i] = pick(rng, (cust[i].lower(), f" {cust[i]} ", cust[i].replace("-", "")))
    cust[anomalies["orphan"]] = [f"CLI-{n_customers + 1000 + int(x):05d}" for x in rng.integers(0, 8000, anomalies["orphan"].size)]
    sku_out = case_noise(rng, skus, 0.0, 0.01 * r)
    for i in np.flatnonzero(rng.random(n) < 0.02 * r):
        sku_out[i] = sku_out[i].lower()
    sku_out[anomalies["unknown_sku"]] = [f"SKU-XXX-{9000 + int(x):04d}" for x in rng.integers(0, 999, anomalies["unknown_sku"].size)]

    qty_style = weighted_choice(rng, n, [0.90, 0.07, 0.03]) if r > 0 else np.zeros(n, int)
    qty_out = np.array([(f"{q}", f"{q}.0", f" {q} ")[s] for q, s in zip(qty_original.tolist(), qty_style.tolist())], dtype=object)
    disc_out = np.empty(n, dtype=object)
    disc_style = weighted_choice(rng, n, [0.5, 0.25, 0.15, 0.10]) if r > 0 else np.zeros(n, int)
    zero_style = weighted_choice(rng, n, [0.5, 0.3, 0.2]) if r > 0 else np.zeros(n, int)
    for i, (bp, s, z) in enumerate(zip(sales["discount_bp"].to_list(), disc_style.tolist(), zero_style.tolist())):
        pct = bp // 100
        disc_out[i] = (None, "0%", "0")[z] if bp == 0 else (f"{pct}%", f"{pct / 100:.2f}", f"{pct / 100:.2f}".replace(".", ","), f"{pct}")[s]

    sheet = string_frame({
        "ID Pedido": order_ids,
        "Línea": sales["order_line"].cast(pl.String),
        "Fecha Pedido ": format_dates(rng, sales["order_date"].to_list(), weighted_choice(rng, n, DATE_STYLE_P) if r > 0 else np.zeros(n, int)),
        "ID Cliente": cust,
        "SKU": sku_out,
        "Canal": labels_for(rng, sales["channel_code"].to_numpy(), CHANNEL_LABELS, 0.2 * r),
        "Método de Pago": labels_for(rng, sales["payment_code"].to_numpy(), PAYMENT_LABELS, 0.15 * r),
        "Cantidad": qty_out,
        "Precio Unitario": format_money(sales["unit_price_cents"].to_numpy(), weighted_choice(rng, n, MONEY_STYLE_P) if r > 0 else np.zeros(n, int)),
        "Descuento": disc_out,
        "Estado": labels_for(rng, status, STATUS_LABELS, 0.15 * r),
        "Notas": np.where(rng.random(n) < 0.03 * r, np.array(NOTES, dtype=object)[rng.integers(0, len(NOTES), n)], None),
    }, SALES_HEADERS)

    sheet = overwrite(sheet, anomalies["bad_date"], "Fecha Pedido ",
                      [pick(rng, ("", "N/A", "#REF!", "31/02/2025", "2025-13-01", "sin fecha", "00/00/0000")) or None for _ in anomalies["bad_date"]])
    sheet = overwrite(sheet, anomalies["bad_qty"], "Cantidad",
                      [pick(rng, ("", "0", "-1", "#VALUE!", "dos", "1.5")) or None for _ in anomalies["bad_qty"]])
    sheet = overwrite(sheet, anomalies["bad_status"], "Estado",
                      [pick(rng, ("Pendiente revisar", "¿?", "En proceso")) for _ in anomalies["bad_status"]])
    sheet = overwrite(sheet, anomalies["missing_price"], "Precio Unitario",
                      [pick(rng, ("", "#N/A", "consultar", "#DIV/0!")) or None for _ in anomalies["missing_price"]])
    sheet = overwrite(sheet, anomalies["empty_channel"], "Canal", [None] * anomalies["empty_channel"].size)

    # Duplicados exactos (copiar/pegar) justo debajo de la fila original
    repeat = np.ones(n, dtype=np.int64)
    repeat[anomalies["exact_dup"]] = 2
    body = sheet[np.repeat(np.arange(n), repeat).tolist()]
    # Algunas filas en blanco intercaladas
    n_mid_blank = 3 if r > 0 else 0
    cuts = np.sort(rng.choice(np.arange(1, body.height), n_mid_blank, replace=False)) if n_mid_blank else np.array([], int)
    parts, prev = [], 0
    for c in cuts.tolist():
        parts += [body[prev:c], blank_frame(SALES_HEADERS, 1)]
        prev = c
    parts.append(body[prev:])
    # Correcciones: la misma línea re-introducida al final con la cantidad corregida
    edited = take_rows(sheet, anomalies["edited"]).with_columns(
        pl.Series("Cantidad", [str(q) for q in qty[anomalies["edited"]]], dtype=pl.String))
    parts.append(edited)
    if r > 0:  # Fila de totales y filas vacías finales, típicas de un export de Sheets
        total = blank_frame(SALES_HEADERS, 1).with_columns(
            pl.lit("TOTAL").alias("ID Pedido"), pl.lit(str(int(qty_original.sum()))).alias("Cantidad"))
        parts += [total, blank_frame(SALES_HEADERS, 20)]
    sheet_out = pl.concat(parts)

    # Resultado esperado del ETL (mismas reglas, aritmética entera en céntimos)
    rejected_mask = np.zeros(n, dtype=bool)
    for key in ("bad_date", "bad_qty", "unknown_sku", "bad_status"):
        rejected_mask[anomalies[key]] = True
    loaded = (status != "CANCELLED") & product_valid & ~rejected_mask
    gross = qty * price
    discount = (gross * sales["discount_bp"].to_numpy() + 5000) // 10000
    cost = qty * np.array([unit_cost[s] for s in skus], dtype=np.int64)
    orphan_mask = np.zeros(n, dtype=bool)
    orphan_mask[anomalies["orphan"]] = True
    returned = status == "RETURNED"

    stats = {
        "rows_in_sheet": sheet_out.height,
        "blank_rows": n_mid_blank + (20 if r > 0 else 0),
        "exact_duplicates": int(anomalies["exact_dup"].size),
        "key_duplicates_superseded": int(anomalies["edited"].size),
        "prices_imputed": int(anomalies["missing_price"].size),
        "orphan_customer_ids": int(anomalies["orphan"].size),
        "empty_channels": int(anomalies["empty_channel"].size),
        "guest_lines": int(guest.sum()),
    }
    expected = {
        "fact_rows": int(loaded.sum()),
        "orders": int(sales.filter(pl.Series(loaded))["order_id"].n_unique()),
        "cancelled_lines_excluded": int(((status == "CANCELLED") & product_valid).sum()),
        "rejected": {
            "clave_pedido_invalida": 1 if r > 0 else 0,
            "fecha_invalida": int(anomalies["bad_date"].size),
            "cantidad_invalida": int(anomalies["bad_qty"].size),
            "sku_desconocido": int(anomalies["unknown_sku"].size),
            "producto_rechazado": int((~product_valid).sum()),
            "estado_desconocido": int(anomalies["bad_status"].size),
        },
        "unknown_customer_rows": int((loaded & (guest | orphan_mask)).sum()),
        "unknown_channel_rows": int(anomalies["empty_channel"].size),
        "returned_rows": int((loaded & returned).sum()),
        "quantity": int(qty[loaded].sum()),
        "gross_amount_cents": int(gross[loaded].sum()),
        "discount_amount_cents": int(discount[loaded].sum()),
        "net_amount_cents": int((gross - discount)[loaded].sum()),
        "cost_amount_cents": int(cost[loaded].sum()),
        "profit_amount_cents": int((gross - discount - cost)[loaded].sum()),
        "min_order_date": str(sales["order_date"].filter(pl.Series(loaded)).min()),
        "max_order_date": str(sales["order_date"].filter(pl.Series(loaded)).max()),
    }
    return sheet_out, stats, expected


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=RAW_DIR, help="carpeta de salida (por defecto data/raw)")
    p.add_argument("--sales-rows", type=int, default=75_000, help="líneas de venta 'reales' (antes de duplicados)")
    p.add_argument("--customers", type=int, default=8_000)
    p.add_argument("--products", type=int, default=450)
    p.add_argument("--start-date", type=date.fromisoformat, default=date(2023, 1, 1))
    p.add_argument("--end-date", type=date.fromisoformat, default=date(2026, 8, 31))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dirty-rate", type=float, default=1.0,
                   help="multiplicador de anomalías (0 = datos limpios, 1 = por defecto)")
    args = p.parse_args(argv)
    if args.sales_rows < 1 or args.customers < 1 or args.products < 10:
        p.error("--sales-rows y --customers deben ser >= 1 y --products >= 10")
    if args.end_date <= args.start_date:
        p.error("--end-date debe ser posterior a --start-date")
    return args


def generate(args: argparse.Namespace) -> dict:
    t0 = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    products = generate_products(rng, args.products)
    customers = generate_customers(rng, args.customers, args.start_date, args.end_date)
    sales = generate_sales(rng, args.sales_rows, customers, products, args.start_date, args.end_date)

    products_sheet, p_stats, invalid_skus = build_products_sheet(rng, products, args.dirty_rate)
    customers_sheet, c_stats = build_customers_sheet(rng, customers, args.dirty_rate)
    sales_sheet, s_stats, expected = build_sales_sheet(rng, sales, products, customers.height, invalid_skus, args.dirty_rate)

    out: Path = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    files = {"productos.csv": products_sheet, "clientes.csv": customers_sheet, "ventas.csv": sales_sheet}
    for name, df in files.items():
        df.write_csv(out / name)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "generate_data.py",
        "params": {
            "seed": args.seed, "sales_rows": args.sales_rows, "customers": args.customers,
            "products": args.products, "start_date": args.start_date.isoformat(),
            "end_date": args.end_date.isoformat(), "dirty_rate": args.dirty_rate,
        },
        "files": {name: {"rows": df.height, "sha256": sha256(out / name)} for name, df in files.items()},
        "sheets": {"productos": p_stats, "clientes": c_stats, "ventas": s_stats},
        "expected_etl": {
            "dim_product_rows": p_stats["expected_products"],
            "dim_customer_rows": c_stats["expected_customers"],
            **expected,
        },
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
    }
    (out / "_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = generate(args)
    exp = manifest["expected_etl"]
    print(f"Datos generados en {args.output_dir} ({manifest['elapsed_seconds']} s)")
    for name, meta in manifest["files"].items():
        print(f"  {name:<14} {meta['rows']:>9,} filas")
    print(f"  Periodo: {exp['min_order_date']} -> {exp['max_order_date']}")
    print(f"  Líneas de hecho esperadas tras el ETL: {exp['fact_rows']:,}  |  "
          f"Ventas netas esperadas: {exp['net_amount_cents'] / 100:,.2f} EUR")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
