"""Hito 4 - FastAPI inference server for the optimized (quantized GGUF) fraud model.

    uvicorn app.main:app --host 0.0.0.0 --port 8000

The model is loaded once at startup (fail fast: a missing/corrupt artifact or a
feature-schema mismatch prevents the process from becoming ready). Inference runs
on the numpy-only engine: no scikit-learn in the serving path.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse

from fraud_detection import __version__
from fraud_detection.features import featurize_records, schema_fingerprint
from fraud_detection.runtime import QuantizedGBDT

from .schemas import Health, ModelInfo, Prediction, PredictRequest, PredictResponse, RiskLevel
from .settings import Settings, get_settings

LOG = logging.getLogger("fraud_api")


@dataclass(frozen=True)
class ServedModel:
    engine: QuantizedGBDT
    info: ModelInfo

    @classmethod
    def load(cls, path: Path) -> ServedModel:
        engine = QuantizedGBDT.load(path)
        metadata = engine.metadata
        if metadata.get("fraud.schema_fingerprint") != schema_fingerprint():
            raise RuntimeError(
                f"{path} was trained for feature schema {metadata.get('fraud.schema_fingerprint')}"
                f" but this service implements {schema_fingerprint()}"
            )
        sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        decision = float(metadata["fraud.decision_threshold"])
        info = ModelInfo(
            name=str(metadata.get("general.name", "fraud-model")),
            version=sha256[:12],
            sha256=sha256,
            path=str(path),
            quantization=engine.leaf_type.name,
            file_bytes=path.stat().st_size,
            resident_bytes=engine.nbytes,
            n_trees=engine.n_trees,
            tree_depth=engine.depth,
            features=list(engine.feature_names),
            decision_threshold=decision,
            review_threshold=float(metadata.get("fraud.review_threshold", decision)),
            schema_fingerprint=metadata["fraud.schema_fingerprint"],
            trained_at=metadata.get("fraud.trained_at"),
            validation_roc_auc=metadata.get("fraud.validation.roc_auc"),
            validation_pr_auc=metadata.get("fraud.validation.pr_auc"),
        )
        engine.predict_proba(np.zeros((1, engine.n_features)))  # warm-up
        return cls(engine, info)

    def predict(self, request: PredictRequest) -> PredictResponse:
        started = time.perf_counter()
        proba = self.engine.predict_proba(featurize_records(request.instances))
        high = proba >= self.info.decision_threshold
        medium = proba >= self.info.review_threshold
        predictions = [
            Prediction(
                transaction_id=tx.transaction_id,
                fraud_probability=round(float(p), 6),
                is_fraud=bool(is_high),
                risk_level=RiskLevel.HIGH
                if is_high
                else RiskLevel.MEDIUM
                if is_medium
                else RiskLevel.LOW,
            )
            for tx, p, is_high, is_medium in zip(
                request.instances, proba, high, medium, strict=True
            )
        ]
        return PredictResponse(
            model_name=self.info.name,
            model_version=self.info.version,
            quantization=self.info.quantization,
            decision_threshold=self.info.decision_threshold,
            review_threshold=self.info.review_threshold,
            predictions=predictions,
            inference_ms=round((time.perf_counter() - started) * 1_000, 3),
        )


def _configure_logging(level: str) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    LOG.handlers[:] = [handler]
    LOG.setLevel(level)
    LOG.propagate = False


def _log(event: str, **fields: object) -> None:
    LOG.info(json.dumps({"ts": round(time.time(), 3), "event": event, **fields}, default=str))


def get_model(request: Request) -> ServedModel:
    model: ServedModel | None = request.app.state.model
    if model is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "model not loaded")
    return model


ModelDependency = Annotated[ServedModel, Depends(get_model)]


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    _configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        model = ServedModel.load(settings.model_path)
        app.state.model = model
        _log(
            "model_loaded",
            name=model.info.name,
            version=model.info.version,
            quantization=model.info.quantization,
            file_bytes=model.info.file_bytes,
            resident_bytes=model.info.resident_bytes,
        )
        yield
        app.state.model = None

    app = FastAPI(
        title="Fraud Detection Inference API",
        version=__version__,
        description="Scores card transactions with a quantized (GGUF) gradient-boosted model.",
        lifespan=lifespan,
        docs_url="/docs" if settings.docs_enabled else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.docs_enabled else None,
    )
    app.state.model = None

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            LOG.exception("unhandled error (request_id=%s)", request_id)
            response = JSONResponse(
                {"detail": "internal server error", "request_id": request_id}, status_code=500
            )
        elapsed_ms = (time.perf_counter() - started) * 1_000
        response.headers["x-request-id"] = request_id
        response.headers["server-timing"] = f"app;dur={elapsed_ms:.3f}"
        _log(
            "request",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            latency_ms=round(elapsed_ms, 3),
        )
        return response

    @app.get("/health/live", response_model=Health, tags=["health"])
    def live(request: Request) -> Health:
        """Liveness: the process is up."""
        model: ServedModel | None = request.app.state.model
        return Health(status="ok", model_loaded=model is not None)

    @app.get(
        "/health/ready",
        response_model=Health,
        tags=["health"],
        responses={503: {"model": Health, "description": "Model not loaded"}},
    )
    def ready(request: Request) -> Health | JSONResponse:
        """Readiness: the model is loaded and the service can take traffic."""
        model: ServedModel | None = request.app.state.model
        if model is None:
            return JSONResponse(
                Health(status="unavailable", model_loaded=False).model_dump(), status_code=503
            )
        return Health(status="ok", model_loaded=True, model_version=model.info.version)

    @app.get("/model", response_model=ModelInfo, tags=["model"])
    def model_info(model: ModelDependency) -> ModelInfo:
        """Metadata of the served artifact (version, quantization, footprint, thresholds)."""
        return model.info

    @app.post("/predict", response_model=PredictResponse, tags=["inference"])
    def predict(payload: PredictRequest, model: ModelDependency) -> PredictResponse:
        """Fraud probability, decision and risk band for a batch of transactions."""
        return model.predict(payload)

    return app


app = create_app()
