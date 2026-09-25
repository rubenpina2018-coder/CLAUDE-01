"""Smoke + latency test against a running inference server (stdlib only).

    python scripts/smoke_test.py --url http://127.0.0.1:8000 --requests 500 --slo-ms 200

Checks readiness, response contract, rejection of invalid payloads (HTTP 422) and
client-side round-trip latency of single-transaction requests against the SLO.
"""

from __future__ import annotations

import argparse
import http.client
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

SAMPLE = json.loads((Path(__file__).parent / "sample_request.json").read_text())


class Client:
    def __init__(self, url: str) -> None:
        parsed = urlparse(url)
        self.conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=10)

    def request(self, method: str, path: str, body: Any = None) -> tuple[int, Any, float]:
        payload = json.dumps(body).encode() if body is not None else None
        headers = {"content-type": "application/json"} if payload else {}
        started = time.perf_counter()
        self.conn.request(method, path, body=payload, headers=headers)
        response = self.conn.getresponse()
        data = response.read()
        elapsed_ms = (time.perf_counter() - started) * 1_000
        return response.status, json.loads(data) if data else None, elapsed_ms


def wait_ready(client: Client, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            status, body, _ = client.request("GET", "/health/ready")
            if status == 200:
                print(f"ready: {body}")
                return
        except (ConnectionError, OSError, http.client.HTTPException):
            client.conn.close()
        if time.monotonic() > deadline:
            raise TimeoutError("server did not become ready")
        time.sleep(0.5)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q / 100 * (len(ordered) - 1)))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--slo-ms", type=float, default=200.0)
    parser.add_argument("--ready-timeout", type=float, default=60.0)
    args = parser.parse_args()
    client = Client(args.url)
    failures: list[str] = []

    def check(condition: bool, message: str) -> None:
        print(("PASS " if condition else "FAIL ") + message)
        if not condition:
            failures.append(message)

    wait_ready(client, args.ready_timeout)
    status, info, _ = client.request("GET", "/model")
    check(status == 200, f"GET /model -> {status} ({info.get('name')} v{info.get('version')}, "
          f"{info.get('quantization')}, {info.get('file_bytes')} bytes)")  # fmt: skip

    status, body, first_ms = client.request("POST", "/predict", SAMPLE)
    check(
        status == 200,
        f"POST /predict (batch of {len(SAMPLE['instances'])}) -> {status} in {first_ms:.2f} ms",
    )
    predictions = body["predictions"] if status == 200 else []
    check(len(predictions) == len(SAMPLE["instances"]), "one prediction per instance")
    for prediction in predictions:
        print(f"     {prediction}")
    check(
        all(0.0 <= p["fraud_probability"] <= 1.0 for p in predictions)
        and {p["risk_level"] for p in predictions} <= {"low", "medium", "high"},
        "probabilities in [0, 1] and valid risk levels",
    )
    by_id = {p["transaction_id"]: p for p in predictions}
    check(
        by_id.get("txn-suspicious", {}).get("fraud_probability", 0)
        > by_id.get("txn-regular", {}).get("fraud_probability", 1),
        "suspicious transaction scores higher than the regular one",
    )

    instance = SAMPLE["instances"][0]
    invalid = {
        "unknown field": {"instances": [instance | {"unexpected": 1}]},
        "string amount": {"instances": [instance | {"amount": "12.5"}]},
        "bad category": {"instances": [instance | {"merchant_category": "casino"}]},
        "negative amount": {"instances": [instance | {"amount": -3.0}]},
        "velocity inconsistency": {
            "instances": [instance | {"txn_count_1h": 9, "txn_count_24h": 2}]
        },
        "empty batch": {"instances": []},
    }
    for name, payload in invalid.items():
        status, _, _ = client.request("POST", "/predict", payload)
        check(status == 422, f"rejects {name} with 422 (got {status})")

    single = {"instances": [SAMPLE["instances"][0]]}
    latencies = [client.request("POST", "/predict", single)[2] for _ in range(args.requests)]
    stats = {
        "n": len(latencies),
        "p50": statistics.median(latencies),
        "p95": percentile(latencies, 95),
        "p99": percentile(latencies, 99),
        "max": max(latencies),
    }
    print(
        "single-transaction round trip (ms): "
        + ", ".join(f"{k}={v:.2f}" for k, v in stats.items() if k != "n")
        + f" over {stats['n']} requests"
    )
    check(first_ms < args.slo_ms, f"first request {first_ms:.2f} ms < {args.slo_ms:.0f} ms")
    check(
        stats["max"] < args.slo_ms,
        f"slowest of {stats['n']} requests {stats['max']:.2f} ms < {args.slo_ms:.0f} ms",
    )

    batch = {"instances": SAMPLE["instances"] * 50}
    status, body, batch_ms = client.request("POST", "/predict", batch)
    check(
        status == 200 and batch_ms < args.slo_ms,
        f"batch of {len(batch['instances'])} -> {status} in {batch_ms:.2f} ms "
        f"(server {body.get('inference_ms')} ms)",
    )

    print(f"\n{'SMOKE TEST PASSED' if not failures else f'SMOKE TEST FAILED: {failures}'}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
