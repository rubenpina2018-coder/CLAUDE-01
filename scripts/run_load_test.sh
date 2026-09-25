#!/usr/bin/env bash
# Hito 5 — Levanta la API en segundo plano y ejecuta Locust en modo headless.
#
# Parámetros por variables de entorno (valores por defecto entre paréntesis):
#   USERS (100)  SPAWN_RATE (100)  DURATION (30s)  API_WORKERS (4)
#   LOCUST_PROCESSES (2)  HOST (127.0.0.1)  PORT (8000)  RESULTS_DIR (results)
#
# Con un solo proceso Locust satura ~1 núcleo y se convierte en el cuello de
# botella (~3.100 RPS medidos); con 2 procesos el límite pasa a ser la API.
#
# Artefactos en $RESULTS_DIR: locust_console.log (salida de la terminal),
# locust.log, locust_stats*.csv, locust_report.html y api.log.
set -euo pipefail
cd "$(dirname "$0")/.."

USERS="${USERS:-100}"
SPAWN_RATE="${SPAWN_RATE:-100}"
DURATION="${DURATION:-30s}"
API_WORKERS="${API_WORKERS:-4}"
LOCUST_PROCESSES="${LOCUST_PROCESSES:-2}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
RESULTS_DIR="${RESULTS_DIR:-results}"
BIN="${BIN:-.venv/bin}"

mkdir -p "$RESULTS_DIR"

echo ">> Iniciando API: uvicorn con $API_WORKERS workers en http://$HOST:$PORT (segundo plano)"
"$BIN/uvicorn" main:app --host "$HOST" --port "$PORT" --workers "$API_WORKERS" \
  --no-access-log --log-level warning >"$RESULTS_DIR/api.log" 2>&1 &
API_PID=$!
trap 'kill "$API_PID" 2>/dev/null || true; wait "$API_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 60); do
  curl -sf "http://$HOST:$PORT/health" >/dev/null && break
  sleep 0.5
done
if ! curl -sf "http://$HOST:$PORT/health" >/dev/null; then
  echo "!! La API no respondió al health check" >&2
  cat "$RESULTS_DIR/api.log" >&2
  exit 1
fi
echo ">> API lista. Locust: $USERS usuarios, spawn $SPAWN_RATE/s, duración $DURATION"

LOCUST_ARGS=(-f locustfile.py --headless
  --users "$USERS" --spawn-rate "$SPAWN_RATE" --run-time "$DURATION"
  --host "http://$HOST:$PORT"
  --csv "$RESULTS_DIR/locust" --csv-full-history
  --html "$RESULTS_DIR/locust_report.html"
  --logfile "$RESULTS_DIR/locust.log")
if [ "$LOCUST_PROCESSES" -gt 1 ]; then
  LOCUST_ARGS+=(--processes "$LOCUST_PROCESSES")
fi

"$BIN/locust" "${LOCUST_ARGS[@]}" 2>&1 | tee "$RESULTS_DIR/locust_console.log"
