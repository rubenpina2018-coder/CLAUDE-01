# Fraud-detection inference API (Hito 5).
#
#   builder : installs the pinned serving dependencies into an isolated venv
#   runtime : slim, non-root image = venv + service code + optimized model
#
# The quantized model runs on a numpy-only engine, so the image ships no
# scikit-learn / scipy / pandas.
#
# Build (the model is produced by the pipeline: `make train` or `make pipeline-local`):
#   docker build -t fraud-api:1.0.0 .
#   docker build --build-arg MODEL_PATH=outputs/registry/fraud-detection-gbdt/3/model.gguf -t fraud-api:3 .
# Behind a TLS-inspecting proxy, pass its CA as a BuildKit secret (never stored in a layer):
#   docker build --secret id=pip_ca,src=/path/to/ca-bundle.crt -t fraud-api:1.0.0 .

ARG PYTHON_IMAGE=python:3.11-slim

# --------------------------------------------------------------------------- builder
FROM ${PYTHON_IMAGE} AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONDONTWRITEBYTECODE=1

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY requirements/serve.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=secret,id=pip_ca,required=false \
    if [ -s /run/secrets/pip_ca ]; then export PIP_CERT=/run/secrets/pip_ca; fi \
    && pip install --only-binary=:all: -r /tmp/requirements.txt \
    # Slim the venv: bundled test suites and the installer itself are not needed at
    # runtime (no pip in production = smaller image and attack surface).
    && find /opt/venv -type d -name tests -prune -exec rm -rf {} + \
    && pip uninstall --yes pip setuptools

# --------------------------------------------------------------------------- runtime
FROM ${PYTHON_IMAGE} AS runtime

ARG MODEL_PATH=outputs/model/model.gguf
ARG VERSION=1.0.0

LABEL org.opencontainers.image.title="fraud-detection-api" \
      org.opencontainers.image.description="Card-fraud scoring API (quantized GGUF GBDT, FastAPI)" \
      org.opencontainers.image.version="${VERSION}"

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FRAUD_API_MODEL_PATH=/app/model/model.gguf \
    FRAUD_API_LOG_LEVEL=INFO \
    WEB_CONCURRENCY=1

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY src/fraud_detection ./src/fraud_detection
COPY app ./app
COPY ${MODEL_PATH} ./model/model.gguf

# Byte-compile the service and fail the build early if the model artifact is
# missing, corrupt, or built for a different feature schema.
RUN python -m compileall -q app src \
    && python -c "from pathlib import Path; from app.main import ServedModel; \
m = ServedModel.load(Path('/app/model/model.gguf')); \
print('model OK:', m.info.name, m.info.version, m.info.quantization, m.info.file_bytes, 'bytes')"

USER 10001:10001
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=2).status != 200)"]

# WEB_CONCURRENCY sets the number of uvicorn worker processes.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--no-access-log", "--proxy-headers", "--timeout-graceful-shutdown", "20"]
