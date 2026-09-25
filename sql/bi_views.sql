-- =============================================================================
-- bi_views.sql  ·  Capa de consumo para Power BI y Looker Studio
-- =============================================================================
-- etl_pipeline.py lo despliega automáticamente (y lo vuelve a desplegar si este
-- fichero cambia; en el resto de ejecuciones solo refresca las vistas
-- materializadas con REFRESH ... CONCURRENTLY). Despliegue manual, atómico:
--
--     psql -h localhost -U etl_user -d analytics_dw -v ON_ERROR_STOP=1 -1 -f sql/bi_views.sql
--
-- A) Modelo semántico en estrella (vistas) -> Power BI (Import o DirectQuery)
--      bi.fact_sales · bi.dim_date · bi.dim_customer · bi.dim_product · bi.dim_channel
-- B) Tablas planas desnormalizadas (vistas materializadas) -> Looker Studio
--      bi.mv_sales_flat       una fila por línea de venta con todos los atributos: sin joins
--      bi.mv_kpi_monthly      KPIs mensuales: MoM, YoY, YTD, margen, ticket medio, retención
--      bi.mv_customer_cohort  matriz de cohortes: retención y LTV por meses desde la 1ª compra
--      bi.mv_customer_rfm     cliente 360 + segmentación RFM
-- C) Vistas ligeras (siempre al día)
--      bi.v_product_performance  ranking de productos y clasificación ABC (Pareto 80/15/5)
--      bi.v_data_freshness       última carga correcta y fecha hasta la que hay datos
--      bi.v_data_quality         filas rechazadas por hoja y motivo en las últimas cargas
--
-- Principios: el rol bi_reader solo ve este esquema; las vistas no exponen PII
-- (email, teléfono) ni columnas técnicas; todas las medidas son aditivas salvo
-- los ratios, que se entregan ya calculados donde Looker no puede derivarlos.
-- =============================================================================

DROP MATERIALIZED VIEW IF EXISTS bi.mv_sales_flat, bi.mv_kpi_monthly, bi.mv_customer_cohort,
                                 bi.mv_customer_rfm CASCADE;
DROP VIEW IF EXISTS bi.fact_sales, bi.dim_date, bi.dim_customer, bi.dim_product, bi.dim_channel,
                    bi.v_product_performance, bi.v_data_freshness, bi.v_data_quality CASCADE;

-- =============================================================================
-- A) MODELO SEMÁNTICO EN ESTRELLA (Power BI)
-- =============================================================================

CREATE VIEW bi.dim_date AS
SELECT date_key, full_date, year, quarter, quarter_name, year_quarter, month, month_name, month_short,
       year_month, year_month_key, year_month_label, month_start, month_end, iso_year, iso_week,
       day_of_month, day_of_year, day_of_week, day_name, day_short, is_weekend, is_holiday, holiday_name,
       -- TRUE hasta el último día con ventas: permite comparaciones homogéneas (YTD vs YTD año anterior)
       -- y ocultar acumulados en fechas futuras (patrón "DateWithSales" en DAX)
       full_date <= (SELECT max(date_key) FROM dw.fact_sales)::TEXT::DATE AS date_with_sales
FROM dw.dim_date;
COMMENT ON VIEW bi.dim_date IS 'Power BI: tabla de fechas (marcar como tabla de fechas sobre full_date)';

CREATE VIEW bi.dim_customer AS
SELECT customer_key, customer_id, full_name, segment, city, region, region_iso_code, country, country_iso2,
       signup_date, marketing_opt_in, first_order_date,
       date_trunc('month', first_order_date)::DATE AS cohort_month,
       customer_key <> -1                          AS is_identified
FROM dw.dim_customer;
COMMENT ON VIEW bi.dim_customer IS 'Power BI: dimensión cliente sin PII (email y teléfono excluidos)';

CREATE VIEW bi.dim_product AS
SELECT product_key, sku, product_name, category, subcategory, brand, list_price, unit_cost, list_margin_pct,
       price_band, price_band_sort, is_active, created_date
FROM dw.dim_product;
COMMENT ON VIEW bi.dim_product IS 'Power BI: dimensión producto';

CREATE VIEW bi.dim_channel AS
SELECT channel_key, channel_code, channel_name, channel_type
FROM dw.dim_channel;
COMMENT ON VIEW bi.dim_channel IS 'Power BI: dimensión canal de venta';

CREATE VIEW bi.fact_sales AS
SELECT sales_key, order_id, order_line, date_key, customer_key, product_key, channel_key,
       order_status, is_returned, payment_method, quantity, unit_price, unit_cost, discount_pct,
       gross_amount, discount_amount, net_amount, cost_amount, profit_amount
FROM dw.fact_sales;
COMMENT ON VIEW bi.fact_sales IS 'Power BI: hechos de venta (grano línea de pedido)';

-- =============================================================================
-- B) TABLAS PLANAS MATERIALIZADAS (Looker Studio)
-- =============================================================================

-- B1 · Ventas desnormalizadas: la fuente principal de Looker Studio -----------
CREATE MATERIALIZED VIEW bi.mv_sales_flat AS
SELECT
    f.sales_key,
    f.order_id,
    f.order_line,
    -- Fecha del pedido
    d.full_date                                     AS order_date,
    d.year                                          AS order_year,
    d.quarter                                       AS order_quarter,
    d.year_quarter,
    d.month                                         AS order_month,
    d.month_name,
    d.year_month,
    d.month_start,
    d.iso_week,
    d.day_of_week,
    d.day_name,
    d.is_weekend,
    d.is_holiday,
    -- Cliente
    c.customer_id,
    c.full_name                                     AS customer_name,
    c.segment                                       AS customer_segment,
    c.city                                          AS customer_city,
    c.region                                        AS customer_region,
    c.region_iso_code                               AS customer_region_iso,
    c.country                                       AS customer_country,
    c.country_iso2                                  AS customer_country_iso2,
    c.first_order_date,
    date_trunc('month', c.first_order_date)::DATE   AS cohort_month,
    CASE WHEN c.customer_key = -1 THEN 'No identificado'
         WHEN d.full_date = c.first_order_date THEN 'Nuevo'
         ELSE 'Recurrente' END                      AS customer_type,
    -- Producto
    p.sku,
    p.product_name,
    p.category,
    p.subcategory,
    p.brand,
    p.price_band,
    -- Canal y pedido
    ch.channel_name,
    ch.channel_type,
    f.order_status,
    f.is_returned,
    f.payment_method,
    -- Medidas (aditivas: en Looker Studio agregar con SUM)
    f.quantity,
    f.unit_price,
    f.discount_pct,
    f.gross_amount,
    f.discount_amount,
    f.net_amount,
    f.cost_amount,
    f.profit_amount,
    CASE WHEN f.is_returned THEN f.net_amount ELSE 0 END AS returned_amount,
    CASE WHEN f.is_returned THEN 0 ELSE f.net_amount END AS net_amount_after_returns
FROM dw.fact_sales f
JOIN dw.dim_date d      ON d.date_key = f.date_key
JOIN dw.dim_customer c  ON c.customer_key = f.customer_key
JOIN dw.dim_product p   ON p.product_key = f.product_key
JOIN dw.dim_channel ch  ON ch.channel_key = f.channel_key;

CREATE UNIQUE INDEX ux_mv_sales_flat          ON bi.mv_sales_flat (sales_key);
CREATE INDEX ix_mv_sales_flat_date            ON bi.mv_sales_flat (order_date);
CREATE INDEX ix_mv_sales_flat_category_date   ON bi.mv_sales_flat (category, order_date);
CREATE INDEX ix_mv_sales_flat_country_date    ON bi.mv_sales_flat (customer_country, order_date);
COMMENT ON MATERIALIZED VIEW bi.mv_sales_flat IS
    'Looker Studio: ventas a nivel de línea con todos los atributos (sin joins). Importes aditivos.';

-- B2 · KPIs mensuales precalculados (crecimientos y retención) ----------------
CREATE MATERIALIZED VIEW bi.mv_kpi_monthly AS
WITH bounds AS (
    SELECT min(d.month_start) AS first_month, max(d.month_start) AS last_month
    FROM dw.fact_sales f JOIN dw.dim_date d USING (date_key)
),
months AS (  -- serie continua de meses: garantiza que LAG(1) y LAG(12) comparan el mes correcto
    SELECT DISTINCT d.month_start, d.year, d.month, d.year_month, d.year_month_label
    FROM dw.dim_date d CROSS JOIN bounds b
    WHERE d.month_start BETWEEN b.first_month AND b.last_month
),
sales AS (
    SELECT d.month_start,
           sum(f.net_amount)                                              AS net_sales,
           sum(f.gross_amount)                                            AS gross_sales,
           sum(f.discount_amount)                                         AS discounts,
           sum(f.cost_amount)                                             AS cost_of_sales,
           sum(f.profit_amount)                                           AS gross_profit,
           sum(f.quantity)                                                AS units,
           count(DISTINCT f.order_id)                                     AS orders,
           count(DISTINCT f.customer_key) FILTER (WHERE f.customer_key <> -1) AS active_customers,
           coalesce(sum(f.net_amount) FILTER (WHERE f.is_returned), 0)    AS returned_amount
    FROM dw.fact_sales f JOIN dw.dim_date d USING (date_key)
    GROUP BY d.month_start
),
customer_months AS (
    SELECT DISTINCT f.customer_key, d.month_start
    FROM dw.fact_sales f JOIN dw.dim_date d USING (date_key)
    WHERE f.customer_key <> -1
),
retained AS (  -- clientes activos en el mes que también compraron el mes anterior
    SELECT cur.month_start, count(*) AS retained_customers
    FROM customer_months cur
    JOIN customer_months prev
      ON prev.customer_key = cur.customer_key
     AND prev.month_start = (cur.month_start - INTERVAL '1 month')::DATE
    GROUP BY cur.month_start
),
new_customers AS (  -- clientes cuya primera compra cae en el mes
    SELECT date_trunc('month', first_order_date)::DATE AS month_start, count(*) AS new_customers
    FROM dw.dim_customer
    WHERE customer_key <> -1 AND first_order_date IS NOT NULL
    GROUP BY 1
),
base AS (
    SELECT m.month_start, m.year, m.month, m.year_month, m.year_month_label,
           coalesce(s.net_sales, 0)          AS net_sales,
           coalesce(s.gross_sales, 0)        AS gross_sales,
           coalesce(s.discounts, 0)          AS discounts,
           coalesce(s.cost_of_sales, 0)      AS cost_of_sales,
           coalesce(s.gross_profit, 0)       AS gross_profit,
           coalesce(s.units, 0)              AS units,
           coalesce(s.orders, 0)             AS orders,
           coalesce(s.active_customers, 0)   AS active_customers,
           coalesce(s.returned_amount, 0)    AS returned_amount,
           coalesce(n.new_customers, 0)      AS new_customers,
           coalesce(r.retained_customers, 0) AS retained_customers
    FROM months m
    LEFT JOIN sales s          USING (month_start)
    LEFT JOIN new_customers n  USING (month_start)
    LEFT JOIN retained r       USING (month_start)
)
SELECT
    month_start, year, month, year_month, year_month_label,
    net_sales, gross_sales, discounts, cost_of_sales, gross_profit, units, orders,
    active_customers, new_customers, retained_customers, returned_amount,
    sum(net_sales) OVER (PARTITION BY year ORDER BY month_start)            AS net_sales_ytd,
    lag(net_sales) OVER w                                                   AS net_sales_prev_month,
    lag(net_sales, 12) OVER w                                               AS net_sales_prev_year,
    round(net_sales / nullif(lag(net_sales) OVER w, 0) - 1, 4)              AS mom_growth_pct,
    round(net_sales / nullif(lag(net_sales, 12) OVER w, 0) - 1, 4)          AS yoy_growth_pct,
    round(gross_profit / nullif(net_sales, 0), 4)                           AS gross_margin_pct,
    round(net_sales / nullif(orders, 0), 2)                                 AS avg_order_value,
    round(discounts / nullif(gross_sales, 0), 4)                            AS discount_rate_pct,
    round(returned_amount / nullif(net_sales, 0), 4)                        AS return_rate_pct,
    lag(active_customers) OVER w                                            AS active_customers_prev_month,
    round(retained_customers::NUMERIC / nullif(lag(active_customers) OVER w, 0), 4) AS retention_rate_pct
FROM base
WINDOW w AS (ORDER BY month_start);

CREATE UNIQUE INDEX ux_mv_kpi_monthly ON bi.mv_kpi_monthly (month_start);
COMMENT ON MATERIALIZED VIEW bi.mv_kpi_monthly IS
    'Looker Studio: KPIs mensuales. Ratios (_pct) en tanto por uno: usar formato porcentaje. '
    'retention_rate_pct = clientes del mes anterior que repiten / clientes activos del mes anterior.';

-- B3 · Cohortes de clientes por mes de primera compra --------------------------
CREATE MATERIALIZED VIEW bi.mv_customer_cohort AS
WITH cohorts AS (
    SELECT customer_key, date_trunc('month', first_order_date)::DATE AS cohort_month
    FROM dw.dim_customer
    WHERE customer_key <> -1 AND first_order_date IS NOT NULL
),
activity AS (
    SELECT f.customer_key, d.month_start AS activity_month,
           sum(f.net_amount) AS net_sales, count(DISTINCT f.order_id) AS orders
    FROM dw.fact_sales f JOIN dw.dim_date d USING (date_key)
    WHERE f.customer_key <> -1
    GROUP BY f.customer_key, d.month_start
),
cohort_size AS (
    SELECT cohort_month, count(*) AS cohort_customers FROM cohorts GROUP BY cohort_month
),
cells AS (
    SELECT c.cohort_month, a.activity_month, cs.cohort_customers,
           count(*)          AS active_customers,
           sum(a.orders)     AS orders,
           sum(a.net_sales)  AS net_sales
    FROM cohorts c
    JOIN activity a     USING (customer_key)
    JOIN cohort_size cs USING (cohort_month)
    GROUP BY c.cohort_month, a.activity_month, cs.cohort_customers
)
SELECT
    cohort_month,
    to_char(cohort_month, 'YYYY-MM')                                            AS cohort_label,
    activity_month,
    ((extract(YEAR FROM activity_month) - extract(YEAR FROM cohort_month)) * 12
      + extract(MONTH FROM activity_month) - extract(MONTH FROM cohort_month))::INTEGER AS months_since_first_purchase,
    cohort_customers,
    active_customers,
    round(active_customers::NUMERIC / cohort_customers, 4)                      AS retention_pct,
    orders,
    net_sales,
    round(sum(net_sales) OVER (PARTITION BY cohort_month ORDER BY activity_month)
          / cohort_customers, 2)                                                AS cumulative_ltv_per_customer
FROM cells;

CREATE UNIQUE INDEX ux_mv_customer_cohort ON bi.mv_customer_cohort (cohort_month, activity_month);
COMMENT ON MATERIALIZED VIEW bi.mv_customer_cohort IS
    'Looker Studio: tabla dinámica cohort_label x months_since_first_purchase con retention_pct (mapa de calor)';

-- B4 · Cliente 360 y segmentación RFM -----------------------------------------
CREATE MATERIALIZED VIEW bi.mv_customer_rfm AS
WITH as_of AS (  -- fecha de referencia: último día con ventas (no la fecha del sistema)
    SELECT max(d.full_date) AS as_of_date
    FROM dw.fact_sales f JOIN dw.dim_date d USING (date_key)
),
agg AS (
    SELECT f.customer_key,
           min(d.full_date)                                             AS first_order_date,
           max(d.full_date)                                             AS last_order_date,
           count(DISTINCT f.order_id)                                   AS orders,
           sum(f.quantity)                                              AS units,
           sum(f.net_amount)                                            AS net_sales,
           sum(f.profit_amount)                                         AS gross_profit,
           coalesce(sum(f.net_amount) FILTER (WHERE f.is_returned), 0)  AS returned_amount
    FROM dw.fact_sales f JOIN dw.dim_date d USING (date_key)
    WHERE f.customer_key <> -1
    GROUP BY f.customer_key
),
scored AS (  -- puntuaciones 1-5 por percentil; los empates reciben la misma puntuación
    SELECT a.*, o.as_of_date,
           o.as_of_date - a.last_order_date                                          AS recency_days,
           least(5, 1 + floor(5 * percent_rank() OVER (ORDER BY a.last_order_date)))::INTEGER AS r_score,
           least(5, 1 + floor(5 * percent_rank() OVER (ORDER BY a.orders)))::INTEGER          AS f_score,
           least(5, 1 + floor(5 * percent_rank() OVER (ORDER BY a.net_sales)))::INTEGER       AS m_score
    FROM agg a CROSS JOIN as_of o
)
SELECT
    s.customer_key, c.customer_id, c.full_name AS customer_name, c.segment, c.city, c.region,
    c.country, c.country_iso2, c.signup_date,
    s.first_order_date, s.last_order_date, s.as_of_date, s.recency_days,
    s.last_order_date - s.first_order_date                       AS tenure_days,
    s.orders, s.units, s.net_sales, s.gross_profit, s.returned_amount,
    round(s.net_sales / nullif(s.orders, 0), 2)                   AS avg_order_value,
    round(s.gross_profit / nullif(s.net_sales, 0), 4)             AS gross_margin_pct,
    s.r_score, s.f_score, s.m_score,
    concat(s.r_score, s.f_score, s.m_score)                       AS rfm_code,
    CASE
        WHEN s.r_score >= 4 AND s.f_score >= 4                   THEN 'Campeones'
        WHEN s.r_score >= 3 AND s.f_score >= 3                   THEN 'Leales'
        WHEN s.r_score >= 4                                      THEN 'Nuevos / Prometedores'
        WHEN s.r_score = 3                                       THEN 'Necesitan atención'
        WHEN s.r_score = 1 AND s.f_score >= 4 AND s.m_score >= 4 THEN 'No se pueden perder'
        WHEN s.f_score >= 3                                      THEN 'En riesgo'
        ELSE 'Hibernando / Perdidos'
    END                                                           AS rfm_segment,
    CASE WHEN s.recency_days <= 90  THEN 'Activo'
         WHEN s.recency_days <= 365 THEN 'En riesgo de abandono'
         ELSE 'Inactivo' END                                      AS lifecycle_status
FROM scored s
JOIN dw.dim_customer c USING (customer_key);

CREATE UNIQUE INDEX ux_mv_customer_rfm ON bi.mv_customer_rfm (customer_key);
COMMENT ON MATERIALIZED VIEW bi.mv_customer_rfm IS
    'Looker Studio: una fila por cliente con compras; RFM calculado a la fecha del último dato';

-- =============================================================================
-- C) VISTAS LIGERAS
-- =============================================================================

CREATE VIEW bi.v_product_performance AS
WITH sales AS (
    SELECT product_key,
           sum(net_amount)                                           AS net_sales,
           sum(profit_amount)                                        AS gross_profit,
           sum(quantity)                                             AS units,
           count(DISTINCT order_id)                                  AS orders,
           coalesce(sum(net_amount) FILTER (WHERE is_returned), 0)   AS returned_amount
    FROM dw.fact_sales
    GROUP BY product_key
),
ranked AS (
    SELECT p.product_key, p.sku, p.product_name, p.category, p.subcategory, p.brand, p.price_band,
           p.list_price, p.unit_cost, p.list_margin_pct, p.is_active,
           coalesce(s.net_sales, 0)       AS net_sales,
           coalesce(s.gross_profit, 0)    AS gross_profit,
           coalesce(s.units, 0)           AS units,
           coalesce(s.orders, 0)          AS orders,
           coalesce(s.returned_amount, 0) AS returned_amount,
           sum(coalesce(s.net_sales, 0)) OVER (ORDER BY coalesce(s.net_sales, 0) DESC, p.product_key
                                               ROWS UNBOUNDED PRECEDING) AS cumulative_sales,
           sum(coalesce(s.net_sales, 0)) OVER ()                         AS total_sales
    FROM dw.dim_product p
    LEFT JOIN sales s USING (product_key)
)
SELECT product_key, sku, product_name, category, subcategory, brand, price_band, list_price, unit_cost,
       list_margin_pct, is_active, net_sales, gross_profit, units, orders, returned_amount,
       round(gross_profit / nullif(net_sales, 0), 4)            AS gross_margin_pct,
       round(returned_amount / nullif(net_sales, 0), 4)         AS return_rate_pct,
       rank() OVER (ORDER BY net_sales DESC)                    AS sales_rank,
       rank() OVER (PARTITION BY category ORDER BY net_sales DESC) AS category_rank,
       round(net_sales / nullif(total_sales, 0), 6)             AS sales_share_pct,
       round(cumulative_sales / nullif(total_sales, 0), 6)      AS cumulative_share_pct,
       CASE WHEN net_sales = 0 THEN 'Sin ventas'
            WHEN cumulative_sales - net_sales < 0.80 * total_sales THEN 'A'
            WHEN cumulative_sales - net_sales < 0.95 * total_sales THEN 'B'
            ELSE 'C' END                                        AS abc_class
FROM ranked;
COMMENT ON VIEW bi.v_product_performance IS
    'Ranking de productos y clasificación ABC: A = productos que acumulan el 80 % de la venta neta';

CREATE VIEW bi.v_data_freshness AS
SELECT
    (SELECT max(finished_at) FROM audit.etl_run WHERE status = 'success')          AS last_successful_load_at,
    (SELECT status FROM audit.etl_run ORDER BY run_id DESC LIMIT 1)                AS last_run_status,
    (SELECT max(d.full_date) FROM dw.fact_sales f JOIN dw.dim_date d USING (date_key)) AS data_through_date,
    (SELECT count(*) FROM dw.fact_sales)                                           AS fact_rows;
COMMENT ON VIEW bi.v_data_freshness IS 'Una fila: frescura de los datos para un indicador "Datos a fecha de..."';

CREATE VIEW bi.v_data_quality AS
SELECT r.run_id, r.started_at, r.status, rr.source_sheet, rr.reason, count(*) AS rejected_rows
FROM audit.etl_run r
JOIN audit.rejected_record rr USING (run_id)
WHERE r.run_id > (SELECT coalesce(max(run_id), 0) - 30 FROM audit.etl_run)
GROUP BY r.run_id, r.started_at, r.status, rr.source_sheet, rr.reason;
COMMENT ON VIEW bi.v_data_quality IS 'Filas rechazadas por hoja y motivo (últimas 30 ejecuciones)';

-- =============================================================================
-- PERMISOS: lectura del esquema bi para el rol de BI
-- =============================================================================
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'bi_reader') THEN
        GRANT USAGE ON SCHEMA bi TO bi_reader;
        GRANT SELECT ON ALL TABLES IN SCHEMA bi TO bi_reader;
    END IF;
END
$$;
