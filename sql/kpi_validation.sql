-- =============================================================================
-- kpi_validation.sql  ·  Equivalente SQL de las 10 medidas DAX de BI_MEASURES.md
-- =============================================================================
-- Sirve para validar el modelo de Power BI (o un informe de Looker Studio): una
-- tarjeta o matriz con el mismo filtro debe mostrar exactamente estos valores.
-- Usa solo la capa bi, así que puede ejecutarlo el rol de solo lectura:
--
--     psql -h localhost -U bi_reader -d analytics_dw -f sql/kpi_validation.sql
--
-- Contextos de referencia pensados para el dataset por defecto (datos hasta el
-- 31/08/2026). Si generas otro rango de fechas, ajusta las fechas de los filtros.
-- =============================================================================
WITH sales AS (
    SELECT f.order_id, f.customer_key, f.net_amount, f.profit_amount, f.is_returned,
           d.full_date, d.year, d.month_start
    FROM bi.fact_sales f
    JOIN bi.dim_date d USING (date_key)
),
customers_by_month AS (  -- clientes identificados activos por mes / año
    SELECT DISTINCT customer_key, month_start FROM sales WHERE customer_key <> -1
),
customers_by_year AS (
    SELECT DISTINCT customer_key, year FROM sales WHERE customer_key <> -1
),
kpis (orden, kpi, contexto, valor, unidad) AS (
    SELECT 1, 'Ventas Netas', 'Año 2025', sum(net_amount), 'EUR'
      FROM sales WHERE year = 2025
    UNION ALL
    SELECT 1, 'Ventas Netas', 'Ago 2026', sum(net_amount), 'EUR'
      FROM sales WHERE month_start = DATE '2026-08-01'
    UNION ALL
    SELECT 2, 'Ventas YTD', 'Ago 2026 (acumulado ene-ago)', sum(net_amount), 'EUR'
      FROM sales WHERE full_date BETWEEN DATE '2026-01-01' AND DATE '2026-08-31'
    UNION ALL
    SELECT 3, 'Crecimiento MoM %', 'Ago 2026 vs jul 2026',
           100 * (sum(net_amount) FILTER (WHERE month_start = DATE '2026-08-01')
                  / sum(net_amount) FILTER (WHERE month_start = DATE '2026-07-01') - 1), '%'
      FROM sales
    UNION ALL
    SELECT 4, 'Crecimiento YoY %', 'Año 2025 vs 2024',
           100 * (sum(net_amount) FILTER (WHERE year = 2025) / sum(net_amount) FILTER (WHERE year = 2024) - 1), '%'
      FROM sales
    UNION ALL
    SELECT 4, 'Crecimiento YoY %', 'Año 2026 (ene-ago) vs ene-ago 2025',
           100 * (sum(net_amount) FILTER (WHERE full_date BETWEEN DATE '2026-01-01' AND DATE '2026-08-31')
                  / sum(net_amount) FILTER (WHERE full_date BETWEEN DATE '2025-01-01' AND DATE '2025-08-31') - 1), '%'
      FROM sales
    UNION ALL
    SELECT 5, 'Margen Bruto %', 'Año 2025', 100 * sum(profit_amount) / sum(net_amount), '%'
      FROM sales WHERE year = 2025
    UNION ALL
    SELECT 6, 'Ticket Medio', 'Año 2025', sum(net_amount) / count(DISTINCT order_id), 'EUR'
      FROM sales WHERE year = 2025
    UNION ALL
    SELECT 7, 'Nuevos Clientes', 'Año 2025', count(*), 'clientes'
      FROM bi.dim_customer WHERE first_order_date BETWEEN DATE '2025-01-01' AND DATE '2025-12-31'
    UNION ALL
    SELECT 7, 'Nuevos Clientes', 'Ago 2026', count(*), 'clientes'
      FROM bi.dim_customer WHERE first_order_date BETWEEN DATE '2026-08-01' AND DATE '2026-08-31'
    UNION ALL
    SELECT 8, 'Tasa de Retención %', 'Ago 2026 (clientes de jul 2026 que repiten)',
           100.0 * count(cur.customer_key) / count(*), '%'
      FROM customers_by_month prev
      LEFT JOIN customers_by_month cur
             ON cur.customer_key = prev.customer_key AND cur.month_start = DATE '2026-08-01'
     WHERE prev.month_start = DATE '2026-07-01'
    UNION ALL
    SELECT 8, 'Tasa de Retención %', 'Año 2025 (clientes de 2024 que repiten)',
           100.0 * count(cur.customer_key) / count(*), '%'
      FROM customers_by_year prev
      LEFT JOIN customers_by_year cur ON cur.customer_key = prev.customer_key AND cur.year = 2025
     WHERE prev.year = 2024
    UNION ALL
    SELECT 9, 'Tasa de Devolución %', 'Año 2025',
           100 * coalesce(sum(net_amount) FILTER (WHERE is_returned), 0) / sum(net_amount), '%'
      FROM sales WHERE year = 2025
    UNION ALL
    SELECT 10, 'Valor de Vida del Cliente (CLV)', 'Histórico (independiente de la fecha)',
           sum(profit_amount) FILTER (WHERE customer_key <> -1)
           / count(DISTINCT customer_key) FILTER (WHERE customer_key <> -1), 'EUR'
      FROM sales
)
SELECT orden, kpi, contexto, round(valor, 2) AS valor, unidad
FROM kpis
ORDER BY orden, contexto;
