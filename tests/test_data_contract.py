"""Data contract, featurizer (train/serve parity) and synthetic generator."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from conftest import SAMPLE_REQUEST
from pydantic import TypeAdapter, ValidationError

import generate_data
from fraud_detection.features import (
    FEATURE_NAMES,
    featurize_frame,
    featurize_records,
    schema_fingerprint,
)
from fraud_detection.schema import FEATURE_COLUMNS, SCENARIO_COLUMN, TARGET_COLUMN, Transaction

VALID = json.loads(SAMPLE_REQUEST.read_text())["instances"][0]


def test_valid_transaction_is_accepted() -> None:
    tx = Transaction.model_validate_json(json.dumps(VALID))
    assert tx.merchant_category == "electronics" and tx.card_present is False


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"amount": "12.5"}, "no str -> float coercion"),
        ({"amount": 0}, "amount must be > 0"),
        ({"hour_of_day": 24}, "hour out of range"),
        ({"hour_of_day": 3.0}, "no float -> int coercion"),
        ({"is_new_device": 1}, "no int -> bool coercion"),
        ({"merchant_category": "casino"}, "unknown category"),
        ({"surprise": True}, "extra fields are forbidden"),
        ({"distance_from_home_km": float("inf")}, "non-finite numbers"),
        ({"txn_count_1h": 9, "txn_count_24h": 2}, "24h velocity must cover 1h"),
        ({"channel": "online", "card_present": True}, "card cannot be present online"),
    ],
)
def test_invalid_transactions_are_rejected(override: dict, reason: str) -> None:
    payload = json.dumps(VALID | override).replace("Infinity", "1e999")
    with pytest.raises(ValidationError):
        Transaction.model_validate_json(payload)


def test_nullable_fields_accept_null() -> None:
    tx = Transaction.model_validate_json(
        json.dumps(VALID | {"customer_avg_amount_30d": None, "distance_from_home_km": None})
    )
    X = featurize_records([tx])
    assert np.isnan(X[0, FEATURE_NAMES.index("customer_avg_amount_30d")])
    assert np.isnan(X[0, FEATURE_NAMES.index("amount_to_avg_ratio")])


def test_featurizer_layout_and_one_hot() -> None:
    X = featurize_records([Transaction.model_validate_json(json.dumps(VALID))])
    assert X.shape == (1, len(FEATURE_NAMES)) and X.dtype == np.float64
    row = dict(zip(FEATURE_NAMES, X[0], strict=True))
    assert row["merchant_category=electronics"] == 1.0
    assert sum(v for k, v in row.items() if k.startswith("merchant_category=")) == 1.0
    assert row["channel=online"] == 1.0 and row["is_new_device"] == 1.0
    assert row["amount_to_avg_ratio"] == pytest.approx(
        VALID["amount"] / VALID["customer_avg_amount_30d"]
    )


def test_training_and_serving_featurization_are_identical(dataset: pd.DataFrame) -> None:
    """The batch (pandas) path and the API (Pydantic objects) path must agree exactly."""
    rows = dataset.sample(500, random_state=0)
    records = TypeAdapter(list[Transaction]).validate_json(
        rows[list(FEATURE_COLUMNS)].to_json(orient="records")
    )
    np.testing.assert_array_equal(featurize_frame(rows), featurize_records(records))


def test_schema_fingerprint_is_stable() -> None:
    assert schema_fingerprint() == schema_fingerprint()
    assert len(schema_fingerprint()) == 16


def test_generator_is_deterministic_and_realistic(dataset: pd.DataFrame) -> None:
    config = generate_data.GeneratorConfig(n_transactions=40_000, n_customers=4_000, seed=11)
    pd.testing.assert_frame_equal(generate_data.generate(config), dataset)
    assert 0.015 < dataset[TARGET_COLUMN].mean() < 0.025
    frauds = dataset.loc[dataset[TARGET_COLUMN] == 1, SCENARIO_COLUMN]
    assert set(frauds) == set(generate_data.SCENARIO_MIX)
    assert (dataset.loc[dataset[TARGET_COLUMN] == 0, SCENARIO_COLUMN] == "none").all()
    # Structural missingness: no 30-day average for accounts younger than 30 days.
    young = dataset["account_age_days"] < 30
    assert dataset.loc[young, "customer_avg_amount_30d"].isna().all()
    assert dataset.loc[~young, "customer_avg_amount_30d"].notna().all()
    assert dataset["event_timestamp"].is_monotonic_increasing


def test_every_generated_row_satisfies_the_contract(dataset: pd.DataFrame) -> None:
    generate_data.validate_sample(dataset, n=len(dataset))
