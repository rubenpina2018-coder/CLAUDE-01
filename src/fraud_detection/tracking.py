"""Optional metric logging to Azure ML (MLflow) — a no-op outside Azure ML jobs."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any

LOG = logging.getLogger(__name__)


def log_metrics(metrics: Mapping[str, Any], prefix: str = "") -> bool:
    """Log numeric metrics to the current Azure ML job run.

    Azure ML jobs export ``MLFLOW_TRACKING_URI``/``MLFLOW_RUN_ID`` and the job
    environment ships ``mlflow`` + ``azureml-mlflow``; locally nothing happens.
    """
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        return False
    try:
        import mlflow
    except ImportError:
        LOG.warning("MLFLOW_TRACKING_URI is set but mlflow is not installed; skipping")
        return False
    numeric = {
        f"{prefix}{key}": float(value)
        for key, value in metrics.items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }
    mlflow.log_metrics(numeric)
    return True
