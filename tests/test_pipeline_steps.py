"""Pipeline steps: data preparation, evaluation gate, registry and registration."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import create_autospec

import pandas as pd
import pytest
from azure.core.exceptions import ResourceNotFoundError

import evaluate
import prep_data
import register_model
from fraud_detection import tracking
from fraud_detection.registry import AzureMLModelRegistry, LocalModelRegistry, get_registry
from fraud_detection.schema import TARGET_COLUMN, TIMESTAMP_COLUMN


@pytest.fixture
def local_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("FRAUD_REGISTRY", "local")
    monkeypatch.setenv("FRAUD_LOCAL_REGISTRY_DIR", str(tmp_path / "registry"))
    return tmp_path / "registry"


def test_prep_quarantines_contract_violations(dataset: pd.DataFrame) -> None:
    raw = dataset.head(5_000).copy()
    raw.loc[3, "amount"] = -10.0
    raw.loc[4, "merchant_category"] = "casino"
    raw.loc[5, TARGET_COLUMN] = 2
    raw.loc[6, TIMESTAMP_COLUMN] = "yesterday"
    raw = pd.concat([raw, raw.iloc[[10]]])  # duplicate transaction id
    train, test, rejected, profile = prep_data.prepare(raw, 0.2, max_invalid_fraction=0.01)
    assert sorted(rejected.index) == [3, 4, 5, 6]
    assert rejected.loc[3, "rejection_reason"].startswith("amount")
    assert profile["duplicates_removed"] == 1
    assert len(train) + len(test) == len(raw) - 4 - 1
    assert train[TIMESTAMP_COLUMN].max() <= test[TIMESTAMP_COLUMN].min()  # out-of-time split


def test_prep_fails_when_invalid_rows_exceed_budget(dataset: pd.DataFrame) -> None:
    raw = dataset.head(1_000).copy()
    raw.loc[:49, "hour_of_day"] = 99
    with pytest.raises(prep_data.DataQualityError, match="exceed"):
        prep_data.prepare(raw, 0.2, max_invalid_fraction=0.01)
    with pytest.raises(prep_data.DataQualityError, match="missing required columns"):
        prep_data.prepare(raw.drop(columns=["channel"]), 0.2, 0.01)


def _evaluate(model_dir: Path, test_dir: Path, out: Path, *extra: str) -> dict:
    args = ["--model_input", str(model_dir), "--test_data", str(test_dir)]
    assert evaluate.main([*args, "--evaluation_output", str(out), *extra]) == 0
    return json.loads((out / "evaluation.json").read_text())


def _register(model_dir: Path, evaluation_dir: Path, out: Path) -> dict:
    args = ["--model_input", str(model_dir), "--evaluation_input", str(evaluation_dir)]
    assert register_model.main([*args, "--registration_output", str(out)]) == 0
    return json.loads((out / "registration.json").read_text())


def test_evaluate_and_register_champion_challenger(
    model_dir: Path, prepared: tuple[Path, Path], tmp_path: Path, local_registry: Path
) -> None:
    first = _evaluate(model_dir, prepared[1], tmp_path / "eval1")
    assert first["gate"]["passed"], first["gate"]["checks"]
    assert first["champion"] is None
    assert set(first["slices"]["recall_by_scenario"]) >= {"card_testing", "account_takeover"}
    assert (tmp_path / "eval1" / "evaluation_report.md").read_text().startswith("# Evaluation")
    registration = _register(model_dir, tmp_path / "eval1", tmp_path / "reg1")
    assert registration["status"] == "registered" and registration["version"] == "1"
    assert (local_registry / "fraud-detection-gbdt" / "1" / "model.gguf").is_file()

    # Second run: the registered champion is re-scored on the same test set.
    second = _evaluate(model_dir, prepared[1], tmp_path / "eval2")
    assert second["champion"]["version"] == "1"
    champion_check = next(c for c in second["gate"]["checks"] if "champion" in c["name"])
    assert champion_check["value"] == pytest.approx(0.0) and champion_check["passed"]
    assert _register(model_dir, tmp_path / "eval2", tmp_path / "reg2")["version"] == "2"


def test_failed_gate_skips_registration(
    model_dir: Path, prepared: tuple[Path, Path], tmp_path: Path, local_registry: Path
) -> None:
    result = _evaluate(model_dir, prepared[1], tmp_path / "eval", "--min_pr_auc", "0.999")
    assert not result["gate"]["passed"]
    registration = _register(model_dir, tmp_path / "eval", tmp_path / "reg")
    assert registration["status"] == "skipped" and "pr_auc" in registration["reason"]
    assert not local_registry.exists()


def test_local_registry_versions(tmp_path: Path) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "model.gguf").write_bytes(b"GGUF")
    registry = LocalModelRegistry(tmp_path / "registry")
    assert registry.latest("m") is None
    assert registry.register("m", package, {"pr_auc": "0.8"}, "d").version == "1"
    assert registry.register("m", package, {"pr_auc": "0.9"}, "d").version == "2"
    latest = registry.latest("m")
    assert latest.version == "2" and latest.tags == {"pr_auc": "0.9"}
    assert (registry.download(latest, tmp_path) / "model.json").is_file()


def test_azureml_registry_uses_the_model_operations_api(tmp_path: Path) -> None:
    from azure.ai.ml.constants import AssetTypes
    from azure.ai.ml.entities import Model
    from azure.ai.ml.operations import ModelOperations

    client = types.SimpleNamespace(models=create_autospec(ModelOperations, instance=True))
    client.models.get.side_effect = ResourceNotFoundError("no such model")
    client.models.create_or_update.side_effect = lambda model: Model(
        name=model.name, version="7", tags=model.tags, path="azureml://x"
    )
    registry = AzureMLModelRegistry(client)
    assert registry.latest("fraud") is None
    client.models.get.assert_called_once_with(name="fraud", label="latest")
    registered = registry.register("fraud", tmp_path, {"pr_auc": "0.8"}, "desc")
    submitted: Model = client.models.create_or_update.call_args.args[0]
    assert submitted.type == AssetTypes.CUSTOM_MODEL and submitted.path == str(tmp_path)
    assert registered.version == "7" and registered.tags == {"pr_auc": "0.8"}


def test_registry_backend_selection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("AZUREML_RUN_ID", raising=False)
    monkeypatch.delenv("FRAUD_REGISTRY", raising=False)
    assert get_registry(local_dir=tmp_path).backend == "local"
    with pytest.raises(ValueError):
        get_registry("s3")


def test_metric_tracking_is_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert tracking.log_metrics({"auc": 0.9}) is False
    logged: dict = {}
    fake_mlflow = types.SimpleNamespace(log_metrics=logged.update)
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "azureml://fake")
    assert tracking.log_metrics({"auc": 0.9, "n": 3, "flag": True, "name": "x"}, prefix="test_")
    assert logged == {"test_auc": 0.9, "test_n": 3.0}
