"""Pipeline step 1 - Data preparation.

* Contract validation: every row goes through ``fraud_detection.schema.Transaction``
  (the model that validates API payloads) in JSON mode. Invalid rows are
  quarantined; the step fails if they exceed ``--max_invalid_fraction``.
* Label / timestamp checks and de-duplication on ``transaction_id``.
* Out-of-time split: the most recent ``--test_fraction`` of the period becomes the
  hold-out test set, never seen by training or threshold selection.

Usage:
    python src/prep_data.py --raw_data data/transactions.csv \
        --train_data outputs/prep/train --test_data outputs/prep/test
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import TypeAdapter, ValidationError

from fraud_detection.data_io import read_transactions, write_json
from fraud_detection.schema import (
    FEATURE_COLUMNS,
    ID_COLUMN,
    SCENARIO_COLUMN,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    Transaction,
)

LOG = logging.getLogger("prep_data")
REQUIRED_COLUMNS = (ID_COLUMN, TIMESTAMP_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN)
_ROWS_ADAPTER = TypeAdapter(list[Transaction])


class DataQualityError(RuntimeError):
    pass


def contract_violations(frame: pd.DataFrame) -> dict[int, str]:
    """Positional index -> first violation, for rows breaking the data contract."""
    violations: dict[int, str] = {}
    payload = frame[[ID_COLUMN, *FEATURE_COLUMNS]].to_json(orient="records")
    try:
        _ROWS_ADAPTER.validate_json(payload)
    except ValidationError as exc:
        for error in exc.errors():
            row, *field = error["loc"]
            where = ".".join(map(str, field)) or "row"
            violations.setdefault(int(row), f"{where}: {error['msg']}")
    labels = frame[TARGET_COLUMN]
    for row in np.flatnonzero(~labels.isin([0, 1]).to_numpy()):
        violations.setdefault(int(row), f"{TARGET_COLUMN}: must be 0 or 1")
    timestamps = pd.to_datetime(frame[TIMESTAMP_COLUMN], utc=True, errors="coerce")
    for row in np.flatnonzero(timestamps.isna().to_numpy()):
        violations.setdefault(int(row), f"{TIMESTAMP_COLUMN}: not an ISO-8601 timestamp")
    return violations


def prepare(
    raw: pd.DataFrame, test_fraction: float, max_invalid_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise DataQualityError(f"raw data is missing required columns: {missing}")
    raw = raw.reset_index(drop=True)
    violations = contract_violations(raw)
    invalid_fraction = len(violations) / max(len(raw), 1)
    if invalid_fraction > max_invalid_fraction:
        raise DataQualityError(
            f"{len(violations)} invalid rows ({invalid_fraction:.2%}) exceed the "
            f"{max_invalid_fraction:.2%} budget; first: {next(iter(violations.values()))}"
        )
    bad = np.zeros(len(raw), dtype=bool)
    bad[list(violations)] = True
    rejected = raw[bad].assign(rejection_reason=[violations[i] for i in np.flatnonzero(bad)])
    clean = raw[~bad]
    deduplicated = clean.drop_duplicates(subset=ID_COLUMN, keep="first")

    timestamps = pd.to_datetime(deduplicated[TIMESTAMP_COLUMN], utc=True)
    ordered = deduplicated.iloc[np.argsort(timestamps.to_numpy(), kind="stable")]
    cut = int(len(ordered) * (1 - test_fraction))
    train, test = ordered.iloc[:cut], ordered.iloc[cut:]
    if train[TARGET_COLUMN].nunique() < 2 or test[TARGET_COLUMN].nunique() < 2:
        raise DataQualityError("both splits must contain fraudulent and legitimate rows")

    def describe(part: pd.DataFrame) -> dict[str, object]:
        return {
            "rows": len(part),
            "fraud_rate": float(part[TARGET_COLUMN].mean()),
            "from": str(part[TIMESTAMP_COLUMN].iloc[0]),
            "to": str(part[TIMESTAMP_COLUMN].iloc[-1]),
        }

    profile = {
        "rows_in": len(raw),
        "rows_rejected": int(bad.sum()),
        "duplicates_removed": len(clean) - len(deduplicated),
        "train": describe(train),
        "test": describe(test),
        "missing_rate": {c: float(deduplicated[c].isna().mean()) for c in FEATURE_COLUMNS},
    }
    if SCENARIO_COLUMN in test:
        profile["test_fraud_by_scenario"] = (
            test.loc[test[TARGET_COLUMN] == 1, SCENARIO_COLUMN].value_counts().to_dict()
        )
    return train, test, rejected, profile


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw_data", type=Path, required=True)
    parser.add_argument("--train_data", type=Path, required=True, help="output folder")
    parser.add_argument("--test_data", type=Path, required=True, help="output folder")
    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument("--max_invalid_fraction", type=float, default=0.01)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    args = parse_args(argv)
    raw = read_transactions(args.raw_data)
    train, test, rejected, profile = prepare(raw, args.test_fraction, args.max_invalid_fraction)
    args.train_data.mkdir(parents=True, exist_ok=True)
    args.test_data.mkdir(parents=True, exist_ok=True)
    train.to_csv(args.train_data / "train.csv", index=False)
    test.to_csv(args.test_data / "test.csv", index=False)
    if len(rejected):
        (args.train_data / "quarantine").mkdir(exist_ok=True)
        rejected.to_csv(args.train_data / "quarantine" / "rejected_rows.csv", index=False)
    write_json(args.train_data / "data_profile.json", profile)
    LOG.info(
        "rows in=%d rejected=%d duplicates=%d | train=%d (fraud %.3f%%) test=%d (fraud %.3f%%)",
        profile["rows_in"], profile["rows_rejected"], profile["duplicates_removed"],
        len(train), 100 * profile["train"]["fraud_rate"],
        len(test), 100 * profile["test"]["fraud_rate"],
    )  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
