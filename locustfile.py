"""Hito 5 — Prueba de carga de la API con Locust.

Cada usuario virtual encadena peticiones mezclando las dos consultas de la API:
1 de cada 5 pide el resumen global y 4 de cada 5 el historial de un usuario
aleatorio (la consulta más frecuente en un producto real).

Por defecto no hay tiempo de espera entre peticiones: con 100 usuarios hay
siempre 100 peticiones en vuelo, lo que mide la capacidad máxima de la API
(prueba de estrés). LOCUST_WAIT_MIN/MAX permiten simular tiempo de reflexión.

Ejecución (ver scripts/run_load_test.sh):
    locust -f locustfile.py --headless -u 100 -r 100 -t 30s --host http://127.0.0.1:8000
"""

from __future__ import annotations

import os
import random

from locust import FastHttpUser, between, task

MAX_USUARIO_ID = int(os.getenv("LOCUST_MAX_USUARIO_ID", "10000"))
WAIT_MIN = float(os.getenv("LOCUST_WAIT_MIN", "0"))
WAIT_MAX = float(os.getenv("LOCUST_WAIT_MAX", "0"))


class ClienteAPI(FastHttpUser):
    """Usuario virtual basado en geventhttpclient: mucha más carga por núcleo que HttpUser."""

    wait_time = between(WAIT_MIN, WAIT_MAX)

    @task(1)
    def resumen(self) -> None:
        self.client.get("/api/v1/resumen", name="GET /api/v1/resumen")

    @task(4)
    def historial_usuario(self) -> None:
        usuario_id = random.randint(1, MAX_USUARIO_ID)
        # `name` agrupa todas las URLs en una sola entrada de estadísticas.
        self.client.get(
            f"/api/v1/transacciones/{usuario_id}",
            name="GET /api/v1/transacciones/{usuario_id}",
        )
