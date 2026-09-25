# BI_MEASURES.md · Medidas DAX del modelo de ventas

Las **10 medidas analíticas críticas** del Star Schema de ventas en DAX (Power BI), con su definición de negocio, el código, el formato recomendado y los valores de referencia para validar el modelo contra PostgreSQL.

| # | Medida | Pregunta de negocio | Formato |
|---|--------|--------------------|---------|
| 1 | **Ventas Netas** | ¿Cuánto facturamos (tras descuentos)? | Moneda |
| 2 | **Ventas YTD** | ¿Cuánto llevamos acumulado en el año? | Moneda |
| 3 | **Crecimiento MoM %** | ¿Crecemos respecto al mes anterior? | % |
| 4 | **Crecimiento YoY %** | ¿Crecemos respecto al mismo periodo del año anterior? | % |
| 5 | **Margen Bruto %** | ¿Qué parte de cada euro vendido es beneficio? | % |
| 6 | **Ticket Medio** | ¿Cuánto gasta un cliente por pedido? | Moneda |
| 7 | **Nuevos Clientes** | ¿Cuántos clientes compran por primera vez? | Entero |
| 8 | **Tasa de Retención %** | ¿Qué % de los clientes del periodo anterior vuelve a comprar? | % |
| 9 | **Tasa de Devolución %** | ¿Qué % de la venta se devuelve? | % |
| 10 | **Valor de Vida del Cliente (CLV)** | ¿Cuánto beneficio deja de media un cliente en toda su relación? | Moneda |

> Todas las medidas se apoyan en un pequeño conjunto de **medidas base** (sección 2). Así cada regla de negocio se define **una sola vez**: si cambia la definición de "venta neta", solo se toca una medida.

---

## 1. Preparación del modelo en Power BI (imprescindible)

El DAX de este documento asume el siguiente modelo. Los pasos de conexión están en el [README](README.md#power-bi-desktop).

### 1.1 Tablas

Importa las vistas del esquema `bi` (nunca las tablas de `dw`) y renómbralas así:

| Vista PostgreSQL | Nombre de tabla en Power BI | Rol |
|------------------|-----------------------------|-----|
| `bi.fact_sales` | `Fact_Sales` | Hechos (grano: línea de pedido) |
| `bi.dim_date` | `Dim_Date` | Calendario |
| `bi.dim_customer` | `Dim_Customer` | Cliente (sin PII) |
| `bi.dim_product` | `Dim_Product` | Producto |
| `bi.dim_channel` | `Dim_Channel` | Canal de venta |

### 1.2 Relaciones (uno a varios, dirección de filtro **única**)

| Desde (varios) | Hacia (uno) |
|----------------|-------------|
| `Fact_Sales[date_key]` | `Dim_Date[date_key]` |
| `Fact_Sales[customer_key]` | `Dim_Customer[customer_key]` |
| `Fact_Sales[product_key]` | `Dim_Product[product_key]` |
| `Fact_Sales[channel_key]` | `Dim_Channel[channel_key]` |

Power BI suele detectarlas solo por nombre de columna. Comprueba que ninguna es bidireccional.

### 1.3 Marcar `Dim_Date` como tabla de fechas: obligatorio

*Herramientas de tabla → Marcar como tabla de fechas → columna `full_date`.*

La relación con los hechos usa una clave entera (`date_key`), no una fecha. Solo al marcar la tabla, las funciones de inteligencia de tiempo (`TOTALYTD`, `DATEADD`, `SAMEPERIODLASTYEAR`) eliminan los filtros de las demás columnas del calendario (año, mes, etiqueta). **Sin este paso, YTD, MoM, YoY y Retención devuelven resultados erróneos o vacíos.**

Recomendado: desactiva *Archivo → Opciones → Carga de datos → Fecha y hora automáticas*, porque genera tablas de fechas ocultas innecesarias.

### 1.4 Ordenación, visibilidad y tipos

| Columna | Ordenar por columna |
|---------|---------------------|
| `Dim_Date[month_name]`, `Dim_Date[month_short]` | `Dim_Date[month]` |
| `Dim_Date[year_month_label]` | `Dim_Date[year_month_key]` |
| `Dim_Date[day_name]`, `Dim_Date[day_short]` | `Dim_Date[day_of_week]` |
| `Dim_Product[price_band]` | `Dim_Product[price_band_sort]` |

* **Ocultar** en la vista de informe: todas las claves (`*_key`, `sales_key`), `Dim_Date[date_with_sales]` y las columnas numéricas de `Fact_Sales`. Los usuarios deben usar **medidas**, no columnas agregadas implícitamente.
* **Tipo de dato**: define los importes (`*_amount`, `unit_price`, `unit_cost`) como *Número decimal fijo* (tipo moneda), para evitar errores de redondeo en coma flotante.
* **Tabla de medidas** (opcional): crea una tabla vacía `_Medidas` (*Especificar datos*) y organiza las medidas en carpetas de visualización: `01 Ventas`, `02 Tiempo`, `03 Rentabilidad`, `04 Clientes`.

### 1.5 Convenciones del modelo que el DAX tiene en cuenta

* `customer_key = -1` es el miembro **"Cliente no identificado"** (compras como invitado o con un ID inexistente en el CRM). Su venta **sí** cuenta en importes, pero **no** en recuentos de clientes (activos, nuevos, retención, CLV).
* Los pedidos **cancelados** no están en la tabla de hechos: el ETL los excluye. Las líneas **devueltas** sí están, marcadas con `is_returned = TRUE`.
* `Dim_Date[date_with_sales]` vale `TRUE` hasta el último día con datos. Se usa para comparar periodos homogéneos (enero-agosto frente a enero-agosto) y para no mostrar acumulados en meses futuros.

---

## 2. Medidas base

```dax
Ventas Netas =
SUM ( Fact_Sales[net_amount] )
```

```dax
Coste de Ventas =
SUM ( Fact_Sales[cost_amount] )
```

```dax
Beneficio Bruto =
SUM ( Fact_Sales[profit_amount] )
```

```dax
Pedidos =
DISTINCTCOUNT ( Fact_Sales[order_id] )
```

```dax
Clientes Activos =
CALCULATE (
    DISTINCTCOUNT ( Fact_Sales[customer_key] ),
    KEEPFILTERS ( Fact_Sales[customer_key] <> -1 )    -- excluye "Cliente no identificado"
)
```

```dax
Importe Devuelto =
CALCULATE (
    [Ventas Netas],
    KEEPFILTERS ( Fact_Sales[is_returned] = TRUE () )
)
```

```dax
Ventas Mes Anterior =
CALCULATE (
    [Ventas Netas],
    CALCULATETABLE (
        DATEADD ( Dim_Date[full_date], -1, MONTH ),
        Dim_Date[date_with_sales] = TRUE ()           -- solo el tramo con datos del periodo actual
    )
)
```

```dax
Ventas Año Anterior =
CALCULATE (
    [Ventas Netas],
    CALCULATETABLE (
        SAMEPERIODLASTYEAR ( Dim_Date[full_date] ),
        Dim_Date[date_with_sales] = TRUE ()           -- ene-ago 2026 se compara con ene-ago 2025
    )
)
```

> **Por qué `KEEPFILTERS`.** Un filtro booleano en `CALCULATE` *reemplaza* los filtros existentes sobre esa columna. Con `KEEPFILTERS` se *intersecan*, así que un segmentador sobre la misma columna se sigue respetando.

---

## 3. Las 10 medidas críticas

### 1. Ventas Netas

Importe facturado después de descuentos: `quantity × unit_price − discount_amount`, calculado de forma exacta en el ETL. Es la base de todas las demás medidas (definida en la sección 2).

* **Formato:** moneda, 2 decimales (`#,0.00 €`).
* **Ojo:** incluye las líneas que luego se devolvieron (el ingreso se reconoció). Para la venta neta de devoluciones: `[Ventas Netas] - [Importe Devuelto]`.

### 2. Ventas YTD (acumulado del año)

```dax
Ventas YTD =
VAR _periodoConDatos =
    CALCULATE ( COUNTROWS ( Dim_Date ), KEEPFILTERS ( Dim_Date[date_with_sales] = TRUE () ) ) > 0
RETURN
    IF ( _periodoConDatos, TOTALYTD ( [Ventas Netas], Dim_Date[full_date] ) )
```

* Acumula desde el 1 de enero hasta el final del periodo visible (mes, trimestre o día).
* El `IF` deja en blanco los meses futuros (septiembre-diciembre de 2026). Sin él, la línea de YTD continuaría plana hasta diciembre.
* **Formato:** moneda. Para un ejercicio fiscal que no empieza en enero: `TOTALYTD ( [Ventas Netas], Dim_Date[full_date], "06-30" )`.

### 3. Crecimiento MoM % (mes contra mes)

```dax
Crecimiento MoM % =
VAR _actual = [Ventas Netas]
VAR _anterior = [Ventas Mes Anterior]
RETURN
    IF (
        NOT ISBLANK ( _actual ) && NOT ISBLANK ( _anterior ),
        DIVIDE ( _actual - _anterior, _anterior )
    )
```

* Úsala en visuales con eje **año-mes** (`Dim_Date[year_month_label]`).
* Devuelve blanco, y no −100 % ni +∞, cuando falta uno de los dos meses. `DIVIDE` protege además de la división por cero.
* **Formato:** porcentaje con 1 decimal. Útil con formato condicional verde/rojo.

### 4. Crecimiento YoY % (interanual)

```dax
Crecimiento YoY % =
VAR _actual = [Ventas Netas]
VAR _anterior = [Ventas Año Anterior]
RETURN
    IF (
        NOT ISBLANK ( _actual ) && NOT ISBLANK ( _anterior ),
        DIVIDE ( _actual - _anterior, _anterior )
    )
```

* Funciona en cualquier granularidad: día, mes, trimestre o año.
* **Comparación homogénea:** con los datos hasta el 31/08/2026, el año 2026 se compara con enero-agosto de 2025 (+19,72 %) y no con todo 2025, que daría un engañoso −30 %. Lo consigue `date_with_sales` dentro de `[Ventas Año Anterior]`.
* **Formato:** porcentaje con 1 decimal.

### 5. Margen Bruto %

```dax
Margen Bruto % =
DIVIDE ( [Beneficio Bruto], [Ventas Netas] )
```

* Beneficio bruto = venta neta − coste. El coste unitario se **congela** en la tabla de hechos al cargar la venta, así que un cambio posterior del coste estándar no reescribe la historia.
* Es un ratio **no aditivo**: nunca promedies márgenes de filas o categorías. Esta medida recalcula el cociente en cada contexto, así que el total es correcto.
* **Formato:** porcentaje con 1 decimal.

### 6. Ticket Medio (AOV)

```dax
Ticket Medio =
DIVIDE ( [Ventas Netas], [Pedidos] )
```

* Venta media por pedido (no por línea). `[Pedidos]` usa `DISTINCTCOUNT` sobre la dimensión degenerada `order_id`.
* **Formato:** moneda, 2 decimales.

### 7. Nuevos Clientes

```dax
Nuevos Clientes =
VAR _inicio = MIN ( Dim_Date[full_date] )
VAR _fin = MAX ( Dim_Date[full_date] )
RETURN
    CALCULATE (
        [Clientes Activos],
        KEEPFILTERS (
            Dim_Customer[first_order_date] >= _inicio
                && Dim_Customer[first_order_date] <= _fin
        )
    )
```

* Cuenta los clientes cuya **primera compra de la historia** cae dentro del periodo visible. `first_order_date` la calcula el ETL sobre todas las ventas, así que no depende de los filtros de producto o canal.
* Con un filtro de categoría responde a "clientes nuevos para la empresa que compraron esta categoría".
* **Formato:** número entero.

### 8. Tasa de Retención de Clientes %

```dax
Tasa de Retención % =
VAR _actuales =
    CALCULATETABLE (
        VALUES ( Fact_Sales[customer_key] ),
        KEEPFILTERS ( Fact_Sales[customer_key] <> -1 )
    )
VAR _anterioresMes =
    CALCULATETABLE (
        VALUES ( Fact_Sales[customer_key] ),
        CALCULATETABLE ( DATEADD ( Dim_Date[full_date], -1, MONTH ), Dim_Date[date_with_sales] = TRUE () ),
        KEEPFILTERS ( Fact_Sales[customer_key] <> -1 )
    )
VAR _anterioresAnio =
    CALCULATETABLE (
        VALUES ( Fact_Sales[customer_key] ),
        CALCULATETABLE ( SAMEPERIODLASTYEAR ( Dim_Date[full_date] ), Dim_Date[date_with_sales] = TRUE () ),
        KEEPFILTERS ( Fact_Sales[customer_key] <> -1 )
    )
VAR _esMensual = HASONEVALUE ( Dim_Date[year_month] )
VAR _retenidos =
    IF (
        _esMensual,
        COUNTROWS ( INTERSECT ( _actuales, _anterioresMes ) ),
        COUNTROWS ( INTERSECT ( _actuales, _anterioresAnio ) )
    )
VAR _base =
    IF ( _esMensual, COUNTROWS ( _anterioresMes ), COUNTROWS ( _anterioresAnio ) )
RETURN
    IF ( ISFILTERED ( Dim_Date ), DIVIDE ( _retenidos + 0, _base ) )
```

**Definición.** De los clientes que compraron en el **periodo anterior**, qué porcentaje ha vuelto a comprar en el **periodo actual**:

* **Contexto de un mes** (eje año-mes o segmentador de un mes): el periodo anterior es el **mes anterior**, es decir, la retención mensual.
* **Trimestre, año u otro rango:** el periodo anterior es el **mismo periodo del año anterior**, es decir, la retención interanual. Por ejemplo, de los clientes de 2024, el 55,55 % compró también en 2025.
* **Sin filtro de fecha** (fila de total general) devuelve blanco, porque la retención solo tiene sentido entre dos periodos.
* Los invitados (`-1`) quedan excluidos del numerador y del denominador.
* **Formato:** porcentaje con 1 decimal.
* `DIVIDE` protege de un periodo anterior sin clientes y `+ 0` muestra 0 % en vez de blanco cuando nadie repite.
* **Rendimiento:** `INTERSECT` sobre `VALUES` de una sola columna se resuelve en el motor de almacenamiento y es rápido incluso con millones de filas.

### 9. Tasa de Devolución %

```dax
Tasa de Devolución % =
DIVIDE ( [Importe Devuelto], [Ventas Netas] )
```

* Porcentaje de la venta neta que se devolvió. Analizado por `Dim_Product[category]` o `Dim_Channel[channel_name]` detecta problemas de calidad o de logística.
* **Formato:** porcentaje con 1 decimal.

### 10. Valor de Vida del Cliente (CLV)

```dax
Valor de Vida del Cliente (CLV) =
VAR _beneficioHistorico =
    CALCULATE (
        [Beneficio Bruto],
        REMOVEFILTERS ( Dim_Date ),
        KEEPFILTERS ( Fact_Sales[customer_key] <> -1 )
    )
VAR _clientesHistoricos =
    CALCULATE ( [Clientes Activos], REMOVEFILTERS ( Dim_Date ) )
RETURN
    DIVIDE ( _beneficioHistorico, _clientesHistoricos )
```

* **CLV histórico**: beneficio bruto medio que ha dejado cada cliente identificado a lo largo de toda su relación. Se usa beneficio y no venta, porque un cliente que compra mucho con grandes descuentos puede valer poco.
* Ignora los filtros de fecha (es un valor "de vida"), pero responde al resto: por `Dim_Customer[segment]` compara el valor de Particular, Pyme y Corporativo, y por `Dim_Customer[cohort_month]` compara cohortes.
* **Formato:** moneda, 2 decimales. El CLV predictivo (probabilístico) queda fuera del alcance de una medida DAX.

---

## 4. Añadir y probar todas las medidas de una vez (vista de consultas DAX)

En Power BI Desktop: **Vista de consultas DAX → nueva consulta**. Pega el script y pulsa *Ejecutar*. Obtendrás la tabla de validación por año; con el botón **"Actualizar modelo con cambios"** las medidas se añaden al modelo sin escribirlas una a una.

```dax
DEFINE
    MEASURE Fact_Sales[Ventas Netas] = SUM ( Fact_Sales[net_amount] )
    MEASURE Fact_Sales[Coste de Ventas] = SUM ( Fact_Sales[cost_amount] )
    MEASURE Fact_Sales[Beneficio Bruto] = SUM ( Fact_Sales[profit_amount] )
    MEASURE Fact_Sales[Pedidos] = DISTINCTCOUNT ( Fact_Sales[order_id] )
    MEASURE Fact_Sales[Clientes Activos] =
        CALCULATE ( DISTINCTCOUNT ( Fact_Sales[customer_key] ), KEEPFILTERS ( Fact_Sales[customer_key] <> -1 ) )
    MEASURE Fact_Sales[Importe Devuelto] =
        CALCULATE ( [Ventas Netas], KEEPFILTERS ( Fact_Sales[is_returned] = TRUE () ) )
    MEASURE Fact_Sales[Ventas Mes Anterior] =
        CALCULATE ( [Ventas Netas],
            CALCULATETABLE ( DATEADD ( Dim_Date[full_date], -1, MONTH ), Dim_Date[date_with_sales] = TRUE () ) )
    MEASURE Fact_Sales[Ventas Año Anterior] =
        CALCULATE ( [Ventas Netas],
            CALCULATETABLE ( SAMEPERIODLASTYEAR ( Dim_Date[full_date] ), Dim_Date[date_with_sales] = TRUE () ) )
    MEASURE Fact_Sales[Ventas YTD] =
        VAR _periodoConDatos =
            CALCULATE ( COUNTROWS ( Dim_Date ), KEEPFILTERS ( Dim_Date[date_with_sales] = TRUE () ) ) > 0
        RETURN IF ( _periodoConDatos, TOTALYTD ( [Ventas Netas], Dim_Date[full_date] ) )
    MEASURE Fact_Sales[Crecimiento MoM %] =
        VAR _actual = [Ventas Netas]
        VAR _anterior = [Ventas Mes Anterior]
        RETURN IF ( NOT ISBLANK ( _actual ) && NOT ISBLANK ( _anterior ), DIVIDE ( _actual - _anterior, _anterior ) )
    MEASURE Fact_Sales[Crecimiento YoY %] =
        VAR _actual = [Ventas Netas]
        VAR _anterior = [Ventas Año Anterior]
        RETURN IF ( NOT ISBLANK ( _actual ) && NOT ISBLANK ( _anterior ), DIVIDE ( _actual - _anterior, _anterior ) )
    MEASURE Fact_Sales[Margen Bruto %] = DIVIDE ( [Beneficio Bruto], [Ventas Netas] )
    MEASURE Fact_Sales[Ticket Medio] = DIVIDE ( [Ventas Netas], [Pedidos] )
    MEASURE Fact_Sales[Nuevos Clientes] =
        VAR _inicio = MIN ( Dim_Date[full_date] )
        VAR _fin = MAX ( Dim_Date[full_date] )
        RETURN CALCULATE ( [Clientes Activos],
            KEEPFILTERS ( Dim_Customer[first_order_date] >= _inicio && Dim_Customer[first_order_date] <= _fin ) )
    MEASURE Fact_Sales[Tasa de Retención %] =
        VAR _actuales =
            CALCULATETABLE ( VALUES ( Fact_Sales[customer_key] ), KEEPFILTERS ( Fact_Sales[customer_key] <> -1 ) )
        VAR _anterioresMes =
            CALCULATETABLE ( VALUES ( Fact_Sales[customer_key] ),
                CALCULATETABLE ( DATEADD ( Dim_Date[full_date], -1, MONTH ), Dim_Date[date_with_sales] = TRUE () ),
                KEEPFILTERS ( Fact_Sales[customer_key] <> -1 ) )
        VAR _anterioresAnio =
            CALCULATETABLE ( VALUES ( Fact_Sales[customer_key] ),
                CALCULATETABLE ( SAMEPERIODLASTYEAR ( Dim_Date[full_date] ), Dim_Date[date_with_sales] = TRUE () ),
                KEEPFILTERS ( Fact_Sales[customer_key] <> -1 ) )
        VAR _esMensual = HASONEVALUE ( Dim_Date[year_month] )
        VAR _retenidos =
            IF ( _esMensual, COUNTROWS ( INTERSECT ( _actuales, _anterioresMes ) ),
                COUNTROWS ( INTERSECT ( _actuales, _anterioresAnio ) ) )
        VAR _base = IF ( _esMensual, COUNTROWS ( _anterioresMes ), COUNTROWS ( _anterioresAnio ) )
        RETURN IF ( ISFILTERED ( Dim_Date ), DIVIDE ( _retenidos + 0, _base ) )
    MEASURE Fact_Sales[Tasa de Devolución %] = DIVIDE ( [Importe Devuelto], [Ventas Netas] )
    MEASURE Fact_Sales[Valor de Vida del Cliente (CLV)] =
        VAR _beneficioHistorico =
            CALCULATE ( [Beneficio Bruto], REMOVEFILTERS ( Dim_Date ), KEEPFILTERS ( Fact_Sales[customer_key] <> -1 ) )
        VAR _clientesHistoricos = CALCULATE ( [Clientes Activos], REMOVEFILTERS ( Dim_Date ) )
        RETURN DIVIDE ( _beneficioHistorico, _clientesHistoricos )

EVALUATE
SUMMARIZECOLUMNS (
    Dim_Date[year],
    "Ventas Netas", [Ventas Netas],
    "Ventas YTD", [Ventas YTD],
    "Crecimiento YoY %", [Crecimiento YoY %],
    "Margen Bruto %", [Margen Bruto %],
    "Ticket Medio", [Ticket Medio],
    "Nuevos Clientes", [Nuevos Clientes],
    "Tasa de Retención %", [Tasa de Retención %],
    "Tasa de Devolución %", [Tasa de Devolución %],
    "CLV", [Valor de Vida del Cliente (CLV)]
)
ORDER BY Dim_Date[year]
```

---

## 5. Valores de referencia (validación cruzada con PostgreSQL)

Calculados en SQL sobre la capa `bi` con el **dataset por defecto** (`python generate_data.py`: semilla 42, 75.000 líneas, datos hasta el 31/08/2026). Si tu modelo de Power BI muestra estos números, las relaciones, la tabla de fechas y las medidas están bien configuradas. Para recalcularlos sobre cualquier dataset: `psql -U bi_reader -d analytics_dw -f sql/kpi_validation.sql`.

**Resultado esperado de la consulta DAX de la sección 4 (por año):**

| year | Ventas Netas | Ventas YTD | Crec. YoY % | Margen % | Ticket Medio | Nuevos Clientes | Retención % | Devolución % | CLV |
|------|-------------:|-----------:|------------:|---------:|-------------:|----------------:|------------:|-------------:|----:|
| 2023 | 2.520.980,13 | 2.520.980,13 | (blanco) | 28,10 | 261,27 | 2.093 | (blanco) | 6,80 | 492,93 |
| 2024 | 2.847.292,68 | 2.847.292,68 | 12,94 | 30,17 | 267,75 | 1.558 | 48,64 | 7,17 | 492,93 |
| 2025 | 3.614.614,63 | 3.614.614,63 | 26,95 | 32,02 | 284,21 | 1.919 | 55,55 | 7,40 | 492,93 |
| 2026 | 2.518.492,81 | 2.518.492,81 | 19,72¹ | 33,50 | 293,50 | 1.538 | 46,74¹ | 7,60 | 492,93 |

¹ 2026 tiene datos hasta agosto: se compara con enero-agosto de 2025 (comparación homogénea).

**Otros contextos (`sql/kpi_validation.sql`):**

| Medida | Contexto | Valor esperado |
|--------|----------|---------------:|
| Ventas Netas | Ago 2026 | 292.336,02 € |
| Ventas YTD | Ago 2026 (ene-ago) | 2.518.492,81 € |
| Crecimiento MoM % | Ago 2026 vs jul 2026 | −24,03 % |
| Nuevos Clientes | Ago 2026 | 217 |
| Tasa de Retención % | Ago 2026 (clientes de julio que repiten) | 28,46 % |
| Valor de Vida del Cliente | Histórico | 492,93 € |

---

## 6. Equivalencias en Looker Studio

Looker Studio no tiene un motor de medidas como DAX. Por eso la capa `bi` entrega **tablas planas** y **KPIs ya calculados** que dan los mismos números:

| Medida DAX | Fuente en Looker Studio | Campo o campo calculado |
|------------|-------------------------|-------------------------|
| Ventas Netas | `bi.mv_sales_flat` | `SUM(net_amount)` |
| Ventas YTD | `bi.mv_kpi_monthly` | `net_sales_ytd` (agregación: Máx. en tablas por mes) |
| Crecimiento MoM % | `bi.mv_kpi_monthly` | `mom_growth_pct` (o *Periodo de comparación: periodo anterior* en un cuadro de mando) |
| Crecimiento YoY % | `bi.mv_kpi_monthly` | `yoy_growth_pct` (o *Periodo de comparación: año anterior*) |
| Margen Bruto % | `bi.mv_sales_flat` | `SUM(profit_amount) / SUM(net_amount)` |
| Ticket Medio | `bi.mv_sales_flat` | `SUM(net_amount) / COUNT_DISTINCT(order_id)` |
| Nuevos Clientes | `bi.mv_kpi_monthly` | `new_customers` |
| Tasa de Retención % | `bi.mv_kpi_monthly` / `bi.mv_customer_cohort` | `retention_rate_pct` (mensual) / `retention_pct` (mapa de calor de cohortes) |
| Tasa de Devolución % | `bi.mv_sales_flat` | `SUM(returned_amount) / SUM(net_amount)` |
| CLV | `bi.mv_customer_rfm` | `AVG(gross_profit)` |

Clientes activos en `mv_sales_flat`: `COUNT_DISTINCT(CASE WHEN customer_type != "No identificado" THEN customer_id END)`.

---

## 7. Buenas prácticas aplicadas

* **`DIVIDE` en lugar de `/`**: protege de la división por cero y devuelve blanco.
* **Variables (`VAR`)**: cada subexpresión se evalúa una vez, y el código se lee de arriba abajo.
* **Medidas base reutilizadas**: cada regla de negocio vive en un solo sitio.
* **Ratios recalculados en cada contexto** (nunca promedios de ratios): los totales son correctos.
* **Filtros de columna con `KEEPFILTERS`**: se respetan los segmentadores del usuario.
* **Inteligencia de tiempo sobre la tabla de fechas marcada** y comparaciones homogéneas con `date_with_sales`.
* **Miembro desconocido (-1) excluido de los recuentos de clientes**, pero incluido en los importes: no se pierde venta y no se inflan los clientes.
