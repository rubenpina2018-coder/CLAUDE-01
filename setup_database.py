#!/usr/bin/env python3
"""Crea los roles y la base de datos del motor analítico (se ejecuta una única vez).

Requiere credenciales de superusuario (variables ``PG_ADMIN_*`` del ``.env``).
Es idempotente: si los roles o la base de datos ya existen, sincroniza las
contraseñas con el ``.env`` y reaplica los permisos.

Roles creados:

* ``etl_user``  (``DW_USER``): propietario de la base de datos y del modelo; lo usa el ETL.
* ``bi_reader``: solo lectura del esquema ``bi`` para Power BI / Looker Studio.
  Por seguridad sus sesiones son de solo lectura y tienen un ``statement_timeout``.

Uso::

    python setup_database.py                 # crea etl_user, bi_reader y analytics_dw
    python setup_database.py --with-test-db  # además <DW_DATABASE>_test (tests de integración)
"""

from __future__ import annotations

import argparse
import os
import re
import sys

from sqlalchemy import text
from sqlalchemy.engine import Connection

from analytics_engine.config import DbSettings, load_dotenv

BI_READER_ROLE = "bi_reader"
IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


def ensure_role(conn: Connection, role: str, password: str) -> str:
    exists = conn.execute(text("SELECT 1 FROM pg_roles WHERE rolname = :r"), {"r": role}).scalar()
    verb = "ALTER" if exists else "CREATE"
    # psycopg2 interpola el literal de la contraseña en el cliente con el escapado correcto.
    conn.exec_driver_sql(f"{verb} ROLE {role} WITH LOGIN NOSUPERUSER NOCREATEROLE PASSWORD %(pw)s",
                         {"pw": password})
    return "actualizado" if exists else "creado"


def ensure_database(conn: Connection, database: str, owner: str) -> str:
    exists = conn.execute(text("SELECT 1 FROM pg_database WHERE datname = :d"), {"d": database}).scalar()
    if exists:
        conn.exec_driver_sql(f"ALTER DATABASE {database} OWNER TO {owner}")
    else:
        conn.exec_driver_sql(f"CREATE DATABASE {database} OWNER {owner}")
    conn.exec_driver_sql(f"REVOKE CONNECT, TEMPORARY ON DATABASE {database} FROM PUBLIC")
    conn.exec_driver_sql(f"GRANT CONNECT, TEMPORARY ON DATABASE {database} TO {owner}")
    conn.exec_driver_sql(f"GRANT CONNECT ON DATABASE {database} TO {BI_READER_ROLE}")
    return "ya existía" if exists else "creada"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--with-test-db", action="store_true",
                        help="crea también <DW_DATABASE>_test para los tests de integración")
    args = parser.parse_args(argv)

    load_dotenv()
    dw = DbSettings.from_env()
    admin = DbSettings.admin_from_env()
    bi_password = os.getenv("BI_READER_PASSWORD")
    if not dw.password or not bi_password:
        print("ERROR: define DW_PASSWORD y BI_READER_PASSWORD (copia .env.example a .env)", file=sys.stderr)
        return 2
    for name in (dw.user, dw.database):
        if not IDENTIFIER.match(name):
            print(f"ERROR: identificador no válido: {name!r} (usa minúsculas, dígitos y _)", file=sys.stderr)
            return 2

    databases = [dw.database] + ([f"{dw.database}_test"] if args.with_test_db else [])
    print(f"Conectando como superusuario: {admin.describe()}")
    engine = admin.create_engine(isolation_level="AUTOCOMMIT")
    try:
        with engine.connect() as conn:
            print(f"  rol {dw.user:<12} {ensure_role(conn, dw.user, dw.password)}")
            print(f"  rol {BI_READER_ROLE:<12} {ensure_role(conn, BI_READER_ROLE, bi_password)}")
            # Defensa en profundidad para el rol de BI: sesiones de solo lectura y
            # consultas acotadas en el tiempo aunque se le concedieran más permisos.
            conn.exec_driver_sql(f"ALTER ROLE {BI_READER_ROLE} SET default_transaction_read_only = on")
            conn.exec_driver_sql(f"ALTER ROLE {BI_READER_ROLE} SET statement_timeout = '120s'")
            # Nombres sin cualificar (p. ej. consultas personalizadas de Looker Studio) resuelven en bi
            conn.exec_driver_sql(f"ALTER ROLE {BI_READER_ROLE} SET search_path = bi, public")
            for db in databases:
                print(f"  base de datos {db:<18} {ensure_database(conn, db, owner=dw.user)}")
    finally:
        engine.dispose()
    print("Listo. Siguiente paso: python generate_data.py && python etl_pipeline.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
