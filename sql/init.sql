-- =============================================================================
-- init.sql  ·  Modelo dimensional (Star Schema) del motor analítico de ventas
-- =============================================================================
-- Ejecutar como propietario de la base de datos (etl_user):
--
--     psql -h localhost -U etl_user -d analytics_dw -v ON_ERROR_STOP=1 -f sql/init.sql
--
-- El pipeline (etl_pipeline.py) también lo ejecuta en cada carga: el script es
-- IDEMPOTENTE y NO destructivo (CREATE ... IF NOT EXISTS / ON CONFLICT), así que
-- puede lanzarse las veces que se quiera sin perder datos.
--
--                        +----------------+
--                        |    dim_date    |
--                        +-------+--------+
--                                |
--  +----------------+    +-------+--------+    +----------------+
--  |  dim_customer  +----+   fact_sales   +----+  dim_product   |
--  +----------------+    +-------+--------+    +----------------+
--                                |
--                        +-------+--------+
--                        |  dim_channel   |
--                        +----------------+
--
-- Esquemas:
--   staging  tablas de aterrizaje UNLOGGED (se vacían y recargan con COPY en cada ejecución)
--   dw       modelo dimensional: dimensiones + tabla de hechos
--   bi       capa de consumo para Power BI / Looker Studio (ver sql/bi_views.sql)
--   audit    trazabilidad: ejecuciones del ETL y registros rechazados
-- =============================================================================

CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS dw;
CREATE SCHEMA IF NOT EXISTS bi;
CREATE SCHEMA IF NOT EXISTS audit;

COMMENT ON SCHEMA staging IS 'Aterrizaje transitorio del ETL (tablas UNLOGGED, se recargan en cada ejecución)';
COMMENT ON SCHEMA dw      IS 'Modelo dimensional (Star Schema) de ventas';
COMMENT ON SCHEMA bi      IS 'Capa de consumo para herramientas BI (solo vistas; acceso de solo lectura)';
COMMENT ON SCHEMA audit   IS 'Trazabilidad de ejecuciones del ETL y registros rechazados';

-- =============================================================================
-- DIMENSIONES
-- =============================================================================

-- -----------------------------------------------------------------------------
-- dim_date · calendario continuo de años completos (requisito de la inteligencia
-- de tiempo de DAX). Clave inteligente AAAAMMDD.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dw.dim_date (
    date_key          INTEGER      PRIMARY KEY,
    full_date         DATE         NOT NULL UNIQUE,
    year              SMALLINT     NOT NULL,
    quarter           SMALLINT     NOT NULL CHECK (quarter BETWEEN 1 AND 4),
    quarter_name      VARCHAR(2)   NOT NULL,
    year_quarter      VARCHAR(7)   NOT NULL,
    month             SMALLINT     NOT NULL CHECK (month BETWEEN 1 AND 12),
    month_name        VARCHAR(10)  NOT NULL,
    month_short       VARCHAR(3)   NOT NULL,
    year_month        VARCHAR(7)   NOT NULL,
    year_month_key    INTEGER      NOT NULL,
    year_month_label  VARCHAR(8)   NOT NULL,
    month_start       DATE         NOT NULL,
    month_end         DATE         NOT NULL,
    iso_year          SMALLINT     NOT NULL,
    iso_week          SMALLINT     NOT NULL CHECK (iso_week BETWEEN 1 AND 53),
    day_of_month      SMALLINT     NOT NULL CHECK (day_of_month BETWEEN 1 AND 31),
    day_of_year       SMALLINT     NOT NULL CHECK (day_of_year BETWEEN 1 AND 366),
    day_of_week       SMALLINT     NOT NULL CHECK (day_of_week BETWEEN 1 AND 7),
    day_name          VARCHAR(10)  NOT NULL,
    day_short         VARCHAR(3)   NOT NULL,
    is_weekend        BOOLEAN      NOT NULL,
    is_holiday        BOOLEAN      NOT NULL DEFAULT FALSE,
    holiday_name      VARCHAR(60),
    CONSTRAINT ck_dim_date_key CHECK (date_key = (EXTRACT(YEAR FROM full_date) * 10000
                                                + EXTRACT(MONTH FROM full_date) * 100
                                                + EXTRACT(DAY FROM full_date))::INTEGER)
);
COMMENT ON TABLE  dw.dim_date                  IS 'Dimensión calendario (años completos, sin huecos)';
COMMENT ON COLUMN dw.dim_date.date_key         IS 'Clave AAAAMMDD';
COMMENT ON COLUMN dw.dim_date.quarter_name     IS 'T1..T4';
COMMENT ON COLUMN dw.dim_date.year_month       IS 'AAAA-MM (texto ordenable)';
COMMENT ON COLUMN dw.dim_date.year_month_key   IS 'AAAAMM: columna de ordenación de year_month_label en Power BI';
COMMENT ON COLUMN dw.dim_date.year_month_label IS 'Etiqueta corta para ejes, p. ej. "Ene 2025"';
COMMENT ON COLUMN dw.dim_date.day_of_week      IS 'ISO 8601: 1 = lunes ... 7 = domingo';
COMMENT ON COLUMN dw.dim_date.is_holiday       IS 'Festivo nacional en España (incluye Viernes Santo)';

-- -----------------------------------------------------------------------------
-- dim_customer · SCD tipo 1 (se sobrescriben los atributos con la última versión
-- de la hoja). customer_key = -1 es el miembro "Cliente no identificado"
-- (ventas como invitado o con ID inexistente en el maestro).
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dw.dim_customer (
    customer_key      INTEGER      GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    customer_id       VARCHAR(20)  NOT NULL UNIQUE,
    first_name        VARCHAR(80),
    last_name         VARCHAR(120),
    full_name         VARCHAR(200) NOT NULL,
    email             VARCHAR(254),
    phone             VARCHAR(20),
    segment           VARCHAR(20)  NOT NULL,
    city              VARCHAR(80)  NOT NULL,
    region            VARCHAR(80)  NOT NULL,
    region_iso_code   VARCHAR(10),
    country           VARCHAR(60)  NOT NULL,
    country_iso2      CHAR(2),
    signup_date       DATE,
    marketing_opt_in  BOOLEAN,
    first_order_date  DATE,
    etl_run_id        BIGINT,
    created_at        TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ  NOT NULL DEFAULT now()
);
COMMENT ON TABLE  dw.dim_customer                  IS 'Dimensión cliente (SCD tipo 1). -1 = cliente no identificado';
COMMENT ON COLUMN dw.dim_customer.customer_id      IS 'Clave natural del CRM (CLI-00001)';
COMMENT ON COLUMN dw.dim_customer.email            IS 'PII: no se expone en la capa bi';
COMMENT ON COLUMN dw.dim_customer.phone            IS 'PII normalizada a E.164: no se expone en la capa bi';
COMMENT ON COLUMN dw.dim_customer.region_iso_code  IS 'ISO 3166-2 (p. ej. ES-MD), útil para mapas de Looker Studio';
COMMENT ON COLUMN dw.dim_customer.first_order_date IS 'Fecha de primera compra; la mantiene el ETL tras cargar los hechos';

-- -----------------------------------------------------------------------------
-- dim_product · SCD tipo 1. Los atributos derivados (margen de tarifa y banda de
-- precio) son columnas generadas: siempre coherentes con precio y coste.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dw.dim_product (
    product_key       INTEGER       GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    sku               VARCHAR(20)   NOT NULL UNIQUE,
    product_name      VARCHAR(200)  NOT NULL,
    category          VARCHAR(60)   NOT NULL,
    subcategory       VARCHAR(60)   NOT NULL,
    brand             VARCHAR(60)   NOT NULL,
    list_price        NUMERIC(12,2) NOT NULL CHECK (list_price > 0),
    unit_cost         NUMERIC(12,2) NOT NULL CHECK (unit_cost >= 0),
    list_margin_pct   NUMERIC(7,4)  GENERATED ALWAYS AS (round((list_price - unit_cost) / list_price, 4)) STORED,
    price_band        VARCHAR(24)   GENERATED ALWAYS AS (
                          CASE WHEN list_price < 50  THEN 'Económico (<50 €)'
                               WHEN list_price < 200 THEN 'Medio (50-200 €)'
                               WHEN list_price < 600 THEN 'Alto (200-600 €)'
                               ELSE 'Premium (>=600 €)' END) STORED,
    price_band_sort   SMALLINT      GENERATED ALWAYS AS (
                          CASE WHEN list_price < 50 THEN 1 WHEN list_price < 200 THEN 2
                               WHEN list_price < 600 THEN 3 ELSE 4 END) STORED,
    is_active         BOOLEAN       NOT NULL DEFAULT TRUE,
    created_date      DATE,
    etl_run_id        BIGINT,
    created_at        TIMESTAMPTZ   NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ   NOT NULL DEFAULT now()
);
COMMENT ON TABLE  dw.dim_product                 IS 'Dimensión producto (SCD tipo 1)';
COMMENT ON COLUMN dw.dim_product.unit_cost       IS 'Coste estándar vigente (el coste histórico se congela en fact_sales)';
COMMENT ON COLUMN dw.dim_product.price_band_sort IS 'Columna de ordenación de price_band en Power BI';

-- -----------------------------------------------------------------------------
-- dim_channel · dimensión de referencia estática (sembrada aquí).
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dw.dim_channel (
    channel_key       SMALLINT     PRIMARY KEY,
    channel_code      VARCHAR(20)  NOT NULL UNIQUE,
    channel_name      VARCHAR(40)  NOT NULL,
    channel_type      VARCHAR(20)  NOT NULL
);
COMMENT ON TABLE dw.dim_channel IS 'Canal de venta. -1 = canal desconocido';

-- Miembros de referencia y "desconocido" (claves reservadas <= 0)
INSERT INTO dw.dim_channel (channel_key, channel_code, channel_name, channel_type) VALUES
    (-1, 'UNKNOWN',     'Desconocido',   'Desconocido'),
    ( 1, 'ONLINE',      'Tienda Online', 'Digital'),
    ( 2, 'STORE',       'Tienda Física', 'Físico'),
    ( 3, 'MARKETPLACE', 'Marketplace',   'Digital'),
    ( 4, 'PHONE',       'Televenta',     'Asistido')
ON CONFLICT (channel_key) DO UPDATE
    SET channel_code = EXCLUDED.channel_code,
        channel_name = EXCLUDED.channel_name,
        channel_type = EXCLUDED.channel_type;

INSERT INTO dw.dim_customer (customer_key, customer_id, full_name, segment, city, region, country)
VALUES (-1, 'N/A', 'Cliente no identificado', 'Desconocido', 'Desconocido', 'Desconocido', 'Desconocido')
ON CONFLICT (customer_key) DO NOTHING;

-- =============================================================================
-- TABLA DE HECHOS · grano: una línea de pedido
-- Importes aditivos precalculados; las restricciones CHECK garantizan en la
-- propia base de datos que la aritmética es exacta y coherente.
-- =============================================================================
CREATE TABLE IF NOT EXISTS dw.fact_sales (
    sales_key         BIGINT        GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    order_id          VARCHAR(20)   NOT NULL,
    order_line        SMALLINT      NOT NULL CHECK (order_line > 0),
    date_key          INTEGER       NOT NULL REFERENCES dw.dim_date (date_key),
    customer_key      INTEGER       NOT NULL REFERENCES dw.dim_customer (customer_key),
    product_key       INTEGER       NOT NULL REFERENCES dw.dim_product (product_key),
    channel_key       SMALLINT      NOT NULL REFERENCES dw.dim_channel (channel_key),
    order_status      VARCHAR(20)   NOT NULL CHECK (order_status IN ('Completado', 'Devuelto')),
    is_returned       BOOLEAN       GENERATED ALWAYS AS (order_status = 'Devuelto') STORED,
    payment_method    VARCHAR(30)   NOT NULL,
    quantity          INTEGER       NOT NULL CHECK (quantity > 0),
    unit_price        NUMERIC(12,2) NOT NULL CHECK (unit_price > 0),
    unit_cost         NUMERIC(12,2) NOT NULL CHECK (unit_cost >= 0),
    discount_pct      NUMERIC(5,4)  NOT NULL DEFAULT 0 CHECK (discount_pct >= 0 AND discount_pct < 1),
    gross_amount      NUMERIC(14,2) NOT NULL,
    discount_amount   NUMERIC(14,2) NOT NULL,
    net_amount        NUMERIC(14,2) NOT NULL,
    cost_amount       NUMERIC(14,2) NOT NULL,
    profit_amount     NUMERIC(14,2) NOT NULL,
    etl_run_id        BIGINT,
    loaded_at         TIMESTAMPTZ   NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ   NOT NULL DEFAULT now(),
    CONSTRAINT uq_fact_sales_order_line UNIQUE (order_id, order_line),
    CONSTRAINT ck_fact_sales_gross      CHECK (gross_amount    = quantity * unit_price),
    CONSTRAINT ck_fact_sales_discount   CHECK (discount_amount = round(gross_amount * discount_pct, 2)),
    CONSTRAINT ck_fact_sales_net        CHECK (net_amount      = gross_amount - discount_amount),
    CONSTRAINT ck_fact_sales_cost       CHECK (cost_amount     = quantity * unit_cost),
    CONSTRAINT ck_fact_sales_profit     CHECK (profit_amount   = net_amount - cost_amount)
);
COMMENT ON TABLE  dw.fact_sales                 IS 'Hechos de venta: una fila por línea de pedido (pedidos cancelados excluidos)';
COMMENT ON COLUMN dw.fact_sales.order_id        IS 'Dimensión degenerada: nº de pedido';
COMMENT ON COLUMN dw.fact_sales.unit_cost       IS 'Coste unitario congelado en el momento de la carga';
COMMENT ON COLUMN dw.fact_sales.discount_pct    IS 'Fracción 0-1 (0.15 = 15 %)';
COMMENT ON COLUMN dw.fact_sales.gross_amount    IS 'quantity * unit_price';
COMMENT ON COLUMN dw.fact_sales.discount_amount IS 'round(gross_amount * discount_pct, 2)';
COMMENT ON COLUMN dw.fact_sales.net_amount      IS 'Venta neta = bruto - descuento (medida principal)';
COMMENT ON COLUMN dw.fact_sales.cost_amount     IS 'quantity * unit_cost';
COMMENT ON COLUMN dw.fact_sales.profit_amount   IS 'Beneficio bruto = neto - coste';

-- Índices sobre claves foráneas (joins, filtros de fecha y análisis por cliente)
CREATE INDEX IF NOT EXISTS ix_fact_sales_date     ON dw.fact_sales (date_key);
CREATE INDEX IF NOT EXISTS ix_fact_sales_customer ON dw.fact_sales (customer_key, date_key);
CREATE INDEX IF NOT EXISTS ix_fact_sales_product  ON dw.fact_sales (product_key);
CREATE INDEX IF NOT EXISTS ix_fact_sales_channel  ON dw.fact_sales (channel_key);

-- =============================================================================
-- STAGING · datos ya limpios y tipados por Polars, cargados con COPY.
-- UNLOGGED: sin WAL (más rápido); su contenido es desechable por diseño.
-- =============================================================================
CREATE UNLOGGED TABLE IF NOT EXISTS staging.stg_date (LIKE dw.dim_date INCLUDING DEFAULTS);

CREATE UNLOGGED TABLE IF NOT EXISTS staging.stg_product (
    sku               VARCHAR(20)   PRIMARY KEY,
    product_name      VARCHAR(200)  NOT NULL,
    category          VARCHAR(60)   NOT NULL,
    subcategory       VARCHAR(60)   NOT NULL,
    brand             VARCHAR(60)   NOT NULL,
    list_price        NUMERIC(12,2) NOT NULL,
    unit_cost         NUMERIC(12,2) NOT NULL,
    is_active         BOOLEAN       NOT NULL,
    created_date      DATE,
    source_row        INTEGER       NOT NULL
);

CREATE UNLOGGED TABLE IF NOT EXISTS staging.stg_customer (
    customer_id       VARCHAR(20)   PRIMARY KEY,
    first_name        VARCHAR(80),
    last_name         VARCHAR(120),
    full_name         VARCHAR(200)  NOT NULL,
    email             VARCHAR(254),
    phone             VARCHAR(20),
    segment           VARCHAR(20)   NOT NULL,
    city              VARCHAR(80)   NOT NULL,
    region            VARCHAR(80)   NOT NULL,
    region_iso_code   VARCHAR(10),
    country           VARCHAR(60)   NOT NULL,
    country_iso2      CHAR(2),
    signup_date       DATE,
    marketing_opt_in  BOOLEAN,
    source_row        INTEGER       NOT NULL
);

CREATE UNLOGGED TABLE IF NOT EXISTS staging.stg_sales (
    order_id          VARCHAR(20)   NOT NULL,
    order_line        SMALLINT      NOT NULL,
    order_date        DATE          NOT NULL,
    date_key          INTEGER       NOT NULL,
    customer_id       VARCHAR(20),
    sku               VARCHAR(20)   NOT NULL,
    channel_code      VARCHAR(20)   NOT NULL,
    payment_method    VARCHAR(30)   NOT NULL,
    order_status      VARCHAR(20)   NOT NULL,
    quantity          INTEGER       NOT NULL,
    unit_price        NUMERIC(12,2) NOT NULL,
    unit_cost         NUMERIC(12,2) NOT NULL,
    discount_pct      NUMERIC(5,4)  NOT NULL,
    gross_amount      NUMERIC(14,2) NOT NULL,
    discount_amount   NUMERIC(14,2) NOT NULL,
    net_amount        NUMERIC(14,2) NOT NULL,
    cost_amount       NUMERIC(14,2) NOT NULL,
    profit_amount     NUMERIC(14,2) NOT NULL,
    source_row        INTEGER       NOT NULL,
    PRIMARY KEY (order_id, order_line)
);

-- =============================================================================
-- AUDITORÍA
-- =============================================================================
CREATE TABLE IF NOT EXISTS audit.etl_run (
    run_id            BIGINT        GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pipeline          VARCHAR(60)   NOT NULL DEFAULT 'sales_star_schema',
    status            VARCHAR(10)   NOT NULL DEFAULT 'running'
                                    CHECK (status IN ('running', 'success', 'failed')),
    started_at        TIMESTAMPTZ   NOT NULL DEFAULT now(),
    finished_at       TIMESTAMPTZ,
    source            TEXT,
    rows_extracted    INTEGER,
    rows_rejected     INTEGER,
    rows_loaded       INTEGER,
    metrics           JSONB         NOT NULL DEFAULT '{}'::JSONB,
    error_message     TEXT
);
COMMENT ON TABLE audit.etl_run IS 'Una fila por ejecución del pipeline, con métricas de calidad y volumen';

CREATE TABLE IF NOT EXISTS audit.rejected_record (
    rejected_id       BIGINT        GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id            BIGINT        NOT NULL REFERENCES audit.etl_run (run_id) ON DELETE CASCADE,
    source_sheet      VARCHAR(40)   NOT NULL,
    source_row        INTEGER       NOT NULL,
    reason            VARCHAR(60)   NOT NULL,
    record            JSONB         NOT NULL
);
COMMENT ON TABLE  audit.rejected_record            IS 'Filas de las hojas que no superaron las reglas de calidad';
COMMENT ON COLUMN audit.rejected_record.source_row IS 'Número de fila en la hoja de cálculo (la fila 1 es la cabecera)';
COMMENT ON COLUMN audit.rejected_record.record     IS 'Valores originales de la fila, tal y como venían en la hoja';
CREATE INDEX IF NOT EXISTS ix_rejected_record_run ON audit.rejected_record (run_id, source_sheet, reason);

-- =============================================================================
-- PERMISOS · bi_reader (Power BI / Looker Studio) solo puede leer el esquema bi.
-- Las vistas de bi acceden a dw con los privilegios de su propietario, así que
-- bi_reader no necesita (ni tiene) acceso a dw, staging ni audit.
-- =============================================================================
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'bi_reader') THEN
        EXECUTE format('GRANT CONNECT ON DATABASE %I TO bi_reader', current_database());
        GRANT USAGE ON SCHEMA bi TO bi_reader;
        GRANT SELECT ON ALL TABLES IN SCHEMA bi TO bi_reader;
        ALTER DEFAULT PRIVILEGES IN SCHEMA bi GRANT SELECT ON TABLES TO bi_reader;
    ELSE
        RAISE NOTICE 'El rol bi_reader no existe: ejecuta setup_database.py para crearlo';
    END IF;
END
$$;
