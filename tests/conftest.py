"""Shared fixtures: one small synthetic dataset and one trained model per test session."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import pytest

import generate_data
import prep_data
import train_and_optimize

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_REQUEST = ROOT / "scripts" / "sample_request.json"


@pytest.fixture(scope="session")
def dataset() -> pd.DataFrame:
    config = generate_data.GeneratorConfig(n_transactions=40_000, n_customers=4_000, seed=11)
    return generate_data.generate(config)


@pytest.fixture(scope="session")
def dataset_csv(dataset: pd.DataFrame, tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data") / "transactions.csv"
    dataset.to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def prepared(dataset_csv: Path, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("prep")
    args = ["--raw_data", str(dataset_csv), "--train_data", str(root / "train")]
    assert prep_data.main([*args, "--test_data", str(root / "test")]) == 0
    return root / "train", root / "test"


@pytest.fixture(scope="session")
def model_dir(prepared: tuple[Path, Path], tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("model")
    args = ["--train_data", str(prepared[0]), "--model_output", str(output)]
    assert train_and_optimize.main(args) == 0
    return output


@pytest.fixture(scope="session")
def reference_model(model_dir: Path):
    return joblib.load(model_dir / "baseline_model.joblib")["model"]


@pytest.fixture(scope="session")
def test_frame(prepared: tuple[Path, Path]) -> pd.DataFrame:
    return pd.read_csv(prepared[1] / "test.csv")
