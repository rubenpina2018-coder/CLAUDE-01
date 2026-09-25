"""Featurizer shared by training and serving.

Turns raw transaction columns into the dense float64 matrix the model consumes.
Training (pandas DataFrame) and serving (validated API payloads) both go through
``featurize_columns``: there is exactly one implementation, hence no skew.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np

from .schema import BOOLEAN_COLUMNS, CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS

DERIVED_FEATURES: Final[tuple[str, ...]] = ("amount_to_avg_ratio",)
ONE_HOT_FEATURES: Final[tuple[str, ...]] = tuple(
    f"{column}={value}" for column, vocab in CATEGORICAL_COLUMNS.items() for value in vocab
)
FEATURE_NAMES: Final[tuple[str, ...]] = (
    NUMERIC_COLUMNS + DERIVED_FEATURES + BOOLEAN_COLUMNS + ONE_HOT_FEATURES
)

_AMOUNT = NUMERIC_COLUMNS.index("amount")
_AVG_AMOUNT = NUMERIC_COLUMNS.index("customer_avg_amount_30d")


def featurize_columns(columns: Mapping[str, Sequence[Any] | np.ndarray]) -> np.ndarray:
    """Build the (n_rows, n_features) float64 design matrix. Missing values stay NaN."""
    n_rows = len(columns[FEATURE_COLUMNS[0]])
    X = np.empty((n_rows, len(FEATURE_NAMES)), dtype=np.float64)
    j = 0
    for name in NUMERIC_COLUMNS:
        X[:, j] = np.asarray(columns[name], dtype=np.float64)  # None -> NaN
        j += 1
    with np.errstate(divide="ignore", invalid="ignore"):
        X[:, j] = X[:, _AMOUNT] / X[:, _AVG_AMOUNT]  # NaN when the 30-day average is unknown
    j += 1
    for name in BOOLEAN_COLUMNS:
        X[:, j] = np.asarray(columns[name], dtype=np.float64)
        j += 1
    for name, vocabulary in CATEGORICAL_COLUMNS.items():
        values = np.asarray(columns[name], dtype=object)
        for value in vocabulary:  # unknown categories -> all-zero one-hot block
            X[:, j] = values == value
            j += 1
    return X


def featurize_records(records: Sequence[Any]) -> np.ndarray:
    """Featurize objects exposing the raw columns as attributes (e.g. ``Transaction``)."""
    return featurize_columns(
        {name: [getattr(record, name) for record in records] for name in FEATURE_COLUMNS}
    )


def featurize_frame(frame: Any) -> np.ndarray:
    """Featurize a pandas DataFrame holding (at least) the raw feature columns."""
    missing = [c for c in FEATURE_COLUMNS if c not in frame.columns]
    if missing:
        raise KeyError(f"missing feature columns: {missing}")
    return featurize_columns({name: frame[name].to_numpy() for name in FEATURE_COLUMNS})


def schema_fingerprint() -> str:
    """Stable hash of the feature layout, embedded in model artifacts.

    The server refuses to load a model trained against a different layout.
    """
    spec = {"features": FEATURE_NAMES, "categorical": CATEGORICAL_COLUMNS}
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
