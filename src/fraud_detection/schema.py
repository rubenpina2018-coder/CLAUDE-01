"""Data contract for card transactions: the single source of truth.

The same Pydantic model validates the rows of the training set (prep step of
the pipeline) and the payloads of the inference API, so training and serving
cannot silently drift apart.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MerchantCategory(StrEnum):
    GROCERY = "grocery"
    RESTAURANTS = "restaurants"
    FUEL = "fuel"
    FASHION = "fashion"
    UTILITIES = "utilities"
    ELECTRONICS = "electronics"
    TRAVEL = "travel"
    GAMING = "gaming"
    GIFT_CARDS = "gift_cards"
    CRYPTO_EXCHANGE = "crypto_exchange"
    CASH_WITHDRAWAL = "cash_withdrawal"


class Channel(StrEnum):
    POS = "pos"
    ONLINE = "online"
    MOBILE_APP = "mobile_app"
    ATM = "atm"


DIGITAL_CHANNELS: Final = frozenset({Channel.ONLINE, Channel.MOBILE_APP})

# Column groups of the raw dataset (order matters: it defines the model input).
NUMERIC_COLUMNS: Final[tuple[str, ...]] = (
    "amount",
    "customer_avg_amount_30d",
    "account_age_days",
    "customer_age",
    "txn_count_1h",
    "txn_count_24h",
    "minutes_since_last_txn",
    "distance_from_home_km",
    "hour_of_day",
    "failed_logins_24h",
)
BOOLEAN_COLUMNS: Final[tuple[str, ...]] = ("is_new_device", "card_present", "is_international")
CATEGORICAL_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    "merchant_category": tuple(m.value for m in MerchantCategory),
    "channel": tuple(c.value for c in Channel),
}
FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    NUMERIC_COLUMNS + BOOLEAN_COLUMNS + tuple(CATEGORICAL_COLUMNS)
)
NULLABLE_COLUMNS: Final[frozenset[str]] = frozenset(
    {"customer_avg_amount_30d", "distance_from_home_km"}
)

ID_COLUMN: Final = "transaction_id"
TIMESTAMP_COLUMN: Final = "event_timestamp"
TARGET_COLUMN: Final = "is_fraud"
# Ground-truth fraud pattern, kept only for slice-based evaluation. Never a feature.
SCENARIO_COLUMN: Final = "fraud_scenario"


TRANSACTION_EXAMPLE: Final[dict[str, object]] = {
    "transaction_id": "txn-000042",
    "amount": 1249.99,
    "customer_avg_amount_30d": 58.2,
    "account_age_days": 412,
    "customer_age": 37,
    "txn_count_1h": 2,
    "txn_count_24h": 5,
    "minutes_since_last_txn": 3.5,
    "distance_from_home_km": 2310.0,
    "hour_of_day": 3,
    "failed_logins_24h": 4,
    "merchant_category": "electronics",
    "channel": "online",
    "is_new_device": True,
    "card_present": False,
    "is_international": True,
}


class Transaction(BaseModel):
    """Features of one card transaction, as known at authorization time.

    Strict mode: no silent coercions ("12.5" is not a float, 1 is not a bool).
    Enum fields accept their string values (JSON has no enum type).
    """

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
        json_schema_extra={"examples": [TRANSACTION_EXAMPLE]},
    )

    transaction_id: str | None = Field(
        default=None, min_length=1, max_length=64, description="Caller-side identifier (echoed)."
    )
    amount: float = Field(gt=0, le=1_000_000, allow_inf_nan=False, description="Amount in EUR.")
    customer_avg_amount_30d: float | None = Field(
        default=None,
        gt=0,
        le=1_000_000,
        allow_inf_nan=False,
        description="Customer's average ticket over 30 days; null for accounts < 30 days old.",
    )
    account_age_days: int = Field(ge=0, le=36_500)
    customer_age: int = Field(ge=18, le=120)
    txn_count_1h: int = Field(ge=0, le=1_000, description="Customer transactions in the last hour.")
    txn_count_24h: int = Field(
        ge=0, le=10_000, description="Customer transactions in the last 24h."
    )
    minutes_since_last_txn: float = Field(ge=0, le=5_256_000, allow_inf_nan=False)
    distance_from_home_km: float | None = Field(
        default=None,
        ge=0,
        le=20_100,
        allow_inf_nan=False,
        description="Distance to the customer's home; null when geolocation failed.",
    )
    hour_of_day: int = Field(ge=0, le=23)
    failed_logins_24h: int = Field(ge=0, le=1_000)
    merchant_category: MerchantCategory = Field(strict=False)
    channel: Channel = Field(strict=False)
    is_new_device: bool
    card_present: bool
    is_international: bool

    @model_validator(mode="after")
    def _check_cross_field_consistency(self) -> Transaction:
        if self.txn_count_24h < self.txn_count_1h:
            raise ValueError("txn_count_24h must be greater than or equal to txn_count_1h")
        if self.card_present and self.channel in DIGITAL_CHANNELS:
            raise ValueError(f"card_present=true is impossible for channel '{self.channel}'")
        return self
