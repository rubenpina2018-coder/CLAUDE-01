"""Inference API: contracts, health, errors, startup checks and batch/online parity."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import SAMPLE_REQUEST
from fastapi.testclient import TestClient

from app.main import create_app
from app.schemas import MAX_BATCH_SIZE, PredictResponse
from app.settings import Settings
from fraud_detection.features import featurize_frame
from fraud_detection.runtime import QuantizedGBDT
from fraud_detection.schema import FEATURE_COLUMNS, ID_COLUMN

SAMPLE = json.loads(SAMPLE_REQUEST.read_text())


@pytest.fixture(scope="module")
def client(model_dir: Path) -> Iterator[TestClient]:
    app = create_app(Settings(model_path=model_dir / "model.gguf", log_level="WARNING"))
    with TestClient(app) as test_client:
        yield test_client


def test_health_and_model_metadata(client: TestClient, model_dir: Path) -> None:
    assert client.get("/health/live").json()["status"] == "ok"
    ready = client.get("/health/ready")
    assert ready.status_code == 200 and ready.json()["model_loaded"]
    info = client.get("/model").json()
    assert info["file_bytes"] == (model_dir / "model.gguf").stat().st_size
    assert info["version"] == ready.json()["model_version"] == info["sha256"][:12]
    assert 0 < info["review_threshold"] <= info["decision_threshold"] < 1


def test_predict_contract(client: TestClient) -> None:
    response = client.post("/predict", json=SAMPLE, headers={"x-request-id": "req-123"})
    assert response.status_code == 200
    assert response.headers["x-request-id"] == "req-123"
    assert response.headers["server-timing"].startswith("app;dur=")
    body = PredictResponse.model_validate(response.json())
    by_id = {p.transaction_id: p for p in body.predictions}
    assert by_id["txn-suspicious"].is_fraud and by_id["txn-suspicious"].risk_level == "high"
    assert not by_id["txn-regular"].is_fraud and by_id["txn-regular"].risk_level == "low"


@pytest.mark.parametrize(
    "payload",
    [
        {"instances": []},
        {"instances": [SAMPLE["instances"][0] | {"unexpected": 1}]},
        {"instances": [SAMPLE["instances"][0] | {"amount": "12.5"}]},
        {"instances": [SAMPLE["instances"][0] | {"channel": "fax"}]},
        {"instances": [{k: v for k, v in SAMPLE["instances"][0].items() if k != "amount"}]},
        {"transactions": SAMPLE["instances"]},
        {"instances": SAMPLE["instances"] * (MAX_BATCH_SIZE // 2 + 1)},
    ],
    ids=["empty", "extra-field", "str-number", "bad-enum", "missing-field", "wrong-key", "too-big"],
)
def test_invalid_payloads_return_422(client: TestClient, payload: dict) -> None:
    assert client.post("/predict", json=payload).status_code == 422


def test_online_predictions_match_batch_scoring(
    client: TestClient, model_dir: Path, test_frame: pd.DataFrame
) -> None:
    rows = test_frame.sample(300, random_state=5)
    payload = {
        "instances": json.loads(rows[[ID_COLUMN, *FEATURE_COLUMNS]].to_json(orient="records"))
    }
    response = client.post("/predict", json=payload)
    online = np.array([p["fraud_probability"] for p in response.json()["predictions"]])
    batch = QuantizedGBDT.load(model_dir / "model.gguf").predict_proba(featurize_frame(rows))
    np.testing.assert_allclose(online, batch, atol=1e-6)
    assert [p["transaction_id"] for p in response.json()["predictions"]] == rows[ID_COLUMN].tolist()


def test_single_request_latency_is_well_below_200ms(client: TestClient) -> None:
    single = {"instances": SAMPLE["instances"][:1]}
    timings = []
    for _ in range(50):
        start = time.perf_counter()
        assert client.post("/predict", json=single).status_code == 200
        timings.append((time.perf_counter() - start) * 1_000)
    assert max(timings) < 200


def test_startup_fails_fast_without_a_model(tmp_path: Path) -> None:
    app = create_app(Settings(model_path=tmp_path / "missing.gguf"))
    with pytest.raises(FileNotFoundError), TestClient(app):
        pass


def test_startup_rejects_a_model_for_another_feature_schema(
    model_dir: Path, tmp_path: Path
) -> None:
    model = QuantizedGBDT.load(model_dir / "model.gguf")
    model.save(
        tmp_path / "other.gguf", extra_metadata={"fraud.schema_fingerprint": "0123456789abcdef"}
    )
    app = create_app(Settings(model_path=tmp_path / "other.gguf"))
    with pytest.raises(RuntimeError, match="feature schema"), TestClient(app):
        pass
