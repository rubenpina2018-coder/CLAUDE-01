"""Hito 1 - Synthetic card-transaction dataset for fraud detection.

Scenario-based simulator:

* Legitimate behaviour is driven by per-customer profiles (spend level, channel
  preference, activity, travel propensity, night-owl habits).
* Fraud is injected through distinct attack patterns: account takeover, card
  testing, card skimming, synthetic identities and "subtle" fraud that mimics
  the customer's normal behaviour.
* Realistic overlap: hard negatives (travellers, big legitimate purchases, new
  phones, forgotten passwords), unreported fraud (label noise), structural and
  random missing values, heavy-tailed amounts.

Usage:
    python src/generate_data.py --output data/transactions.csv --n_transactions 200000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from fraud_detection.schema import (
    FEATURE_COLUMNS,
    ID_COLUMN,
    SCENARIO_COLUMN,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    Transaction,
)

LOG = logging.getLogger("generate_data")

LEGIT: Final = "none"
SCENARIO_MIX: Final[dict[str, float]] = {
    "account_takeover": 0.30,
    "card_testing": 0.22,
    "card_skimming": 0.20,
    "synthetic_identity": 0.10,
    "subtle": 0.18,
}
# Legitimate merchant mix per channel.
MERCHANT_MIX: Final[dict[str, dict[str, float]]] = {
    "pos": {
        "grocery": 0.30, "restaurants": 0.22, "fuel": 0.14, "fashion": 0.12,
        "electronics": 0.06, "travel": 0.04, "utilities": 0.04, "gaming": 0.03,
        "gift_cards": 0.05,
    },
    "digital": {
        "fashion": 0.18, "utilities": 0.14, "restaurants": 0.13, "electronics": 0.12,
        "gaming": 0.12, "grocery": 0.12, "travel": 0.10, "gift_cards": 0.05,
        "crypto_exchange": 0.04,
    },
}  # fmt: skip
# Typical ticket size of each category relative to the customer's spend level.
CATEGORY_AMOUNT: Final[dict[str, float]] = {
    "grocery": 1.0, "restaurants": 0.8, "fuel": 1.1, "fashion": 1.6, "utilities": 1.8,
    "electronics": 3.0, "travel": 4.5, "gaming": 0.6, "gift_cards": 1.5,
    "crypto_exchange": 3.5, "cash_withdrawal": 2.2,
}  # fmt: skip
MINUTES_PER_MONTH: Final = 60 * 24 * 30


class GeneratorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    n_transactions: int = Field(default=200_000, ge=1_000, le=20_000_000)
    n_customers: int = Field(default=20_000, ge=100, le=5_000_000)
    fraud_rate: float = Field(default=0.02, gt=0.0, lt=0.5)
    unreported_fraud_rate: float = Field(
        default=0.03, ge=0.0, lt=0.5, description="Share of frauds never reported (label noise)."
    )
    days: int = Field(default=120, ge=7, le=3_650)
    start_date: date = date(2026, 1, 1)
    seed: int = 42


def _choice(rng: np.random.Generator, mix: dict[str, float], size: int) -> np.ndarray:
    labels = np.array(list(mix), dtype=object)
    probs = np.fromiter(mix.values(), dtype=float)
    return labels[rng.choice(len(labels), size=size, p=probs / probs.sum())]


def _hours(rng: np.random.Generator, size: int, night_owl: np.ndarray) -> np.ndarray:
    daytime = rng.normal(14.5, 4.0, size)
    late = rng.normal(23.5, 2.5, size)
    return np.floor(np.where(night_owl, late, daytime)).astype(np.int64) % 24


def _night_hours(rng: np.random.Generator, size: int) -> np.ndarray:
    return rng.integers(0, 6, size)


def _customers(cfg: GeneratorConfig, rng: np.random.Generator) -> dict[str, np.ndarray]:
    n = cfg.n_customers
    is_new = rng.random(n) < 0.12
    account_age = np.where(is_new, rng.integers(0, 90, n), 90 + rng.exponential(1_400, n))
    return {
        "spend": rng.lognormal(3.4, 0.7, n),  # median ticket ~30 EUR
        "account_age_days": account_age.clip(0, 36_000).astype(np.int64),
        "customer_age": rng.normal(43, 15, n).clip(18, 95).astype(np.int64),
        "activity": rng.lognormal(0.0, 0.8, n),
        "travel": rng.beta(0.6, 25.0, n),
        "night_owl": rng.random(n) < 0.06,
        "digital": rng.beta(2.0, 3.0, n),
    }


def _legitimate(
    rng: np.random.Generator, n: int, cust: dict[str, np.ndarray], c: np.ndarray
) -> dict[str, np.ndarray]:
    digital = rng.random(n) < cust["digital"][c]
    channel = np.where(
        digital,
        np.where(rng.random(n) < 0.65, "online", "mobile_app"),
        np.where(rng.random(n) < 0.9, "pos", "atm"),
    ).astype(object)
    merchant = np.where(
        digital, _choice(rng, MERCHANT_MIX["digital"], n), _choice(rng, MERCHANT_MIX["pos"], n)
    )
    merchant[channel == "atm"] = "cash_withdrawal"

    category_scale = np.vectorize(CATEGORY_AMOUNT.__getitem__, otypes=[float])(merchant)
    amount = cust["spend"][c] * category_scale * rng.lognormal(0.0, 0.55, n)
    big_ticket = rng.random(n) < 0.025  # hard negatives: legit large purchases
    amount[big_ticket] *= rng.uniform(3, 10, big_ticket.sum())
    atm = channel == "atm"
    amount[atm] = np.maximum(20, np.round(amount[atm] / 10) * 10)

    count_1h = rng.poisson(0.08 * cust["activity"][c])
    is_international = rng.random(n) < cust["travel"][c]
    new_device = digital & (rng.random(n) < 0.025)  # hard negatives: new phone
    failed = np.where(
        digital,
        rng.poisson(0.03, n) + (rng.random(n) < 0.01) * rng.integers(1, 4, n),
        0,
    )
    return {
        "amount": amount,
        "hour_of_day": _hours(rng, n, cust["night_owl"][c]),
        "txn_count_1h": count_1h,
        "txn_count_24h": count_1h + rng.poisson(1.8 * cust["activity"][c]),
        "is_international": is_international,
        "far_domestic": np.zeros(n, dtype=bool),
        "is_new_device": new_device,
        "failed_logins_24h": failed,
        "merchant_category": merchant,
        "channel": channel,
        "card_present": np.where(channel == "pos", rng.random(n) < 0.96, channel == "atm"),
        "account_age_days": cust["account_age_days"][c].copy(),
        "customer_age": cust["customer_age"][c].copy(),
    }


def _inject_fraud(
    rng: np.random.Generator,
    cols: dict[str, np.ndarray],
    scenario: np.ndarray,
    spend: np.ndarray,
) -> None:
    """Overwrite the features of fraudulent rows according to their attack pattern."""

    def digital_channel(k: int, p_online: float) -> np.ndarray:
        return np.where(rng.random(k) < p_online, "online", "mobile_app").astype(object)

    idx = np.flatnonzero(scenario == "account_takeover")
    k = idx.size
    cols["channel"][idx] = digital_channel(k, 0.55)
    cols["merchant_category"][idx] = _choice(
        rng, {"electronics": .30, "gift_cards": .22, "crypto_exchange": .20, "travel": .16,
              "fashion": .12}, k)  # fmt: skip
    cols["amount"][idx] = spend[idx] * rng.lognormal(np.log(6.0), 0.6, k)
    cols["is_new_device"][idx] = rng.random(k) < 0.8
    cols["failed_logins_24h"][idx] = rng.poisson(1.5, k) + (rng.random(k) < 0.6) * rng.integers(
        1, 6, k
    )
    night = rng.random(k) < 0.5
    cols["hour_of_day"][idx[night]] = _night_hours(rng, night.sum())
    cols["is_international"][idx] = rng.random(k) < 0.35
    cols["far_domestic"][idx] = rng.random(k) < 0.5  # VPN / proxy exit nodes
    cols["txn_count_1h"][idx] = rng.poisson(0.8, k)
    cols["txn_count_24h"][idx] = cols["txn_count_1h"][idx] + rng.poisson(2.5, k)

    idx = np.flatnonzero(scenario == "card_testing")
    k = idx.size
    cols["channel"][idx] = digital_channel(k, 0.85)
    cols["merchant_category"][idx] = _choice(
        rng, {"gaming": .35, "gift_cards": .20, "utilities": .18, "fashion": .15,
              "crypto_exchange": .12}, k)  # fmt: skip
    cols["amount"][idx] = rng.uniform(0.5, 4.0, k)
    cols["txn_count_1h"][idx] = 3 + rng.poisson(4.0, k)
    cols["txn_count_24h"][idx] = cols["txn_count_1h"][idx] + rng.poisson(5.0, k)
    cols["is_new_device"][idx] = rng.random(k) < 0.5
    cols["failed_logins_24h"][idx] = rng.poisson(0.4, k)
    cols["hour_of_day"][idx] = rng.integers(0, 24, k)
    cols["is_international"][idx] = rng.random(k) < 0.3

    idx = np.flatnonzero(scenario == "card_skimming")
    k = idx.size
    atm = rng.random(k) < 0.3
    cols["channel"][idx] = np.where(atm, "atm", "pos").astype(object)
    cols["merchant_category"][idx] = np.where(
        atm,
        "cash_withdrawal",
        _choice(rng, {"electronics": .35, "fashion": .25, "gift_cards": .15, "travel": .13,
                      "fuel": .12}, k),
    )  # fmt: skip
    cols["amount"][idx] = np.where(
        atm, 20.0 * rng.integers(10, 51, k), spend[idx] * rng.lognormal(np.log(7.0), 0.6, k)
    )
    cols["card_present"][idx] = True
    cols["is_new_device"][idx] = False
    cols["failed_logins_24h"][idx] = 0
    cols["is_international"][idx] = rng.random(k) < 0.45
    cols["far_domestic"][idx] = True
    night = rng.random(k) < 0.35
    cols["hour_of_day"][idx[night]] = _night_hours(rng, night.sum())
    cols["txn_count_1h"][idx] = rng.poisson(1.2, k)
    cols["txn_count_24h"][idx] = cols["txn_count_1h"][idx] + rng.poisson(3.0, k)

    idx = np.flatnonzero(scenario == "synthetic_identity")
    k = idx.size
    cols["account_age_days"][idx] = rng.integers(0, 60, k)
    cols["customer_age"][idx] = rng.integers(18, 31, k)
    cols["channel"][idx] = digital_channel(k, 0.7)
    cols["merchant_category"][idx] = _choice(
        rng, {"electronics": .35, "gift_cards": .20, "crypto_exchange": .20, "travel": .15,
              "fashion": .10}, k)  # fmt: skip
    cols["amount"][idx] = spend[idx] * rng.lognormal(np.log(3.5), 0.5, k)
    cols["is_new_device"][idx] = rng.random(k) < 0.3
    cols["failed_logins_24h"][idx] = rng.poisson(0.2, k)

    idx = np.flatnonzero(scenario == "subtle")  # mimics the customer's own behaviour
    k = idx.size
    cols["amount"][idx] *= rng.uniform(1.2, 3.0, k)
    night = rng.random(k) < 0.25
    cols["hour_of_day"][idx[night]] = _night_hours(rng, night.sum())
    digital = np.isin(cols["channel"][idx], ["online", "mobile_app"])
    cols["is_new_device"][idx] = digital & (rng.random(k) < 0.15)
    cols["failed_logins_24h"][idx] = np.where(digital, rng.poisson(0.2, k), 0)


def generate(cfg: GeneratorConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_transactions
    cust = _customers(cfg, rng)
    c = rng.choice(cfg.n_customers, size=n, p=cust["activity"] / cust["activity"].sum())
    spend = cust["spend"][c]
    cols = _legitimate(rng, n, cust, c)

    scenario = np.full(n, LEGIT, dtype=object)
    fraud_idx = rng.choice(n, size=max(1, round(cfg.fraud_rate * n)), replace=False)
    scenario[fraud_idx] = _choice(rng, SCENARIO_MIX, fraud_idx.size)
    _inject_fraud(rng, cols, scenario, spend)

    # Derived fields, recomputed after injection so every row is internally consistent.
    channel = cols["channel"]
    digital = np.isin(channel, ["online", "mobile_app"])
    cols["card_present"] = np.where(digital, False, cols["card_present"]).astype(bool)
    cols["is_new_device"] = (cols["is_new_device"] & digital).astype(bool)
    count_1h = cols["txn_count_1h"]
    cols["minutes_since_last_txn"] = np.where(
        count_1h > 0,
        60 * rng.beta(1.0, np.maximum(count_1h, 1)),
        60 + rng.exponential(720 / cust["activity"][c]),
    ).clip(0.05, MINUTES_PER_MONTH)
    intl, far = cols.pop("is_international"), cols.pop("far_domestic")
    distance = np.where(
        intl,
        rng.lognormal(7.2, 0.6, n),
        np.where(far, rng.lognormal(4.8, 1.0, n), rng.lognormal(1.5, 1.0, n)),
    )
    distance[rng.random(n) < 0.02] = np.nan  # geolocation lookup failures
    cols["distance_from_home_km"] = distance.clip(0, 19_000)
    cols["is_international"] = intl
    avg = spend * 1.8 * rng.lognormal(0.0, 0.15, n)
    cols["customer_avg_amount_30d"] = np.where(cols["account_age_days"] < 30, np.nan, avg)

    label = scenario != LEGIT
    unreported = label & (rng.random(n) < cfg.unreported_fraud_rate)
    label &= ~unreported
    scenario[unreported] = LEGIT  # the bank never learned about these

    start = datetime.combine(cfg.start_date, datetime.min.time(), tzinfo=UTC).timestamp()
    epoch = (
        start + rng.integers(0, cfg.days, n) * 86_400 + cols["hour_of_day"] * 3_600
        + rng.integers(0, 3_600, n)
    )  # fmt: skip

    frame = pd.DataFrame(
        {
            "customer_id": [f"cus_{i:07d}" for i in c],
            TIMESTAMP_COLUMN: pd.to_datetime(epoch, unit="s", utc=True),
            "amount": np.round(np.maximum(cols["amount"], 0.5), 2),
            "customer_avg_amount_30d": np.round(np.maximum(cols["customer_avg_amount_30d"], 1), 2),
            "account_age_days": cols["account_age_days"],
            "customer_age": cols["customer_age"],
            "txn_count_1h": count_1h,
            "txn_count_24h": cols["txn_count_24h"],
            "minutes_since_last_txn": np.round(cols["minutes_since_last_txn"], 2),
            "distance_from_home_km": np.round(cols["distance_from_home_km"], 1),
            "hour_of_day": cols["hour_of_day"],
            "failed_logins_24h": cols["failed_logins_24h"],
            "is_new_device": cols["is_new_device"],
            "card_present": cols["card_present"],
            "is_international": cols["is_international"],
            "merchant_category": cols["merchant_category"].astype(str),
            "channel": channel.astype(str),
            TARGET_COLUMN: label.astype(np.int64),
            SCENARIO_COLUMN: scenario.astype(str),
        }
    )
    frame = frame.sort_values(TIMESTAMP_COLUMN, kind="stable").reset_index(drop=True)
    frame.insert(0, ID_COLUMN, [f"txn_{i:09d}" for i in range(n)])
    frame[TIMESTAMP_COLUMN] = frame[TIMESTAMP_COLUMN].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return frame


def validate_sample(frame: pd.DataFrame, n: int = 2_000, seed: int = 0) -> None:
    """Self-check: generated rows must satisfy the API data contract."""
    sample = frame.sample(min(n, len(frame)), random_state=seed)
    payload = sample[[ID_COLUMN, *FEATURE_COLUMNS]].to_json(orient="records")
    TypeAdapter(list[Transaction]).validate_json(payload)


def summarize(frame: pd.DataFrame) -> dict[str, object]:
    fraud = frame[frame[TARGET_COLUMN] == 1]
    return {
        "rows": len(frame),
        "fraud_rate": round(float(frame[TARGET_COLUMN].mean()), 5),
        "fraud_by_scenario": fraud[SCENARIO_COLUMN].value_counts().to_dict(),
        "missing_rate": {
            c: round(float(frame[c].isna().mean()), 4)
            for c in ("customer_avg_amount_30d", "distance_from_home_km")
        },
        "time_range": [frame[TIMESTAMP_COLUMN].iloc[0], frame[TIMESTAMP_COLUMN].iloc[-1]],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=Path("data/transactions.csv"))
    defaults = GeneratorConfig()
    parser.add_argument("--n_transactions", type=int, default=defaults.n_transactions)
    parser.add_argument("--n_customers", type=int, default=defaults.n_customers)
    parser.add_argument("--fraud_rate", type=float, default=defaults.fraud_rate)
    parser.add_argument("--days", type=int, default=defaults.days)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    args = parse_args(argv)
    cfg = GeneratorConfig(
        n_transactions=args.n_transactions,
        n_customers=args.n_customers,
        fraud_rate=args.fraud_rate,
        days=args.days,
        seed=args.seed,
    )
    LOG.info("Generating dataset with %s", cfg.model_dump_json())
    frame = generate(cfg)
    validate_sample(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    LOG.info("Wrote %s\n%s", args.output, json.dumps(summarize(frame), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
