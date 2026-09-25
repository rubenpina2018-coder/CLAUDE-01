"""Tree compiler + numpy runtime: exactness vs. scikit-learn, quantization fidelity."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from fraud_detection.features import FEATURE_NAMES, featurize_frame
from fraud_detection.quantization import GGMLType
from fraud_detection.runtime import MISSING_BIN, QuantizedGBDT
from fraud_detection.schema import TARGET_COLUMN
from train_and_optimize import (
    BASELINE_FILE,
    OptimizationGate,
    compile_hgb,
    optimize_model,
    quantize_model,
)


@pytest.fixture(scope="module")
def forest(reference_model):
    return compile_hgb(reference_model)


@pytest.fixture(scope="module")
def stress_X(test_frame: pd.DataFrame) -> np.ndarray:
    """Real rows plus injected NaNs, out-of-range values and exact split points."""
    rng = np.random.default_rng(3)
    X = featurize_frame(test_frame.sample(4_000, random_state=1))
    X[rng.random(X.shape) < 0.05] = np.nan
    extremes = X[:200].copy()
    extremes[:100] *= 1e6
    extremes[100:] = -1.0
    return np.vstack([X, extremes])


def test_compiled_model_is_exact_in_float64(reference_model, forest, stress_X) -> None:
    model = quantize_model(forest, GGMLType.F64)
    np.testing.assert_allclose(
        model.decision_function(stress_X), reference_model.decision_function(stress_X), atol=1e-9
    )


def test_split_points_are_evaluated_exactly(reference_model, forest) -> None:
    """Inputs sitting exactly on (and around) every kept split point."""
    model = quantize_model(forest, GGMLType.F64)
    rows = []
    for j, edges in enumerate(forest.bin_edges):
        for value in edges[:: max(1, len(edges) // 20)]:
            for probe in (value, np.nextafter(value, np.inf), np.nextafter(value, -np.inf)):
                row = np.zeros(len(FEATURE_NAMES))
                row[j] = probe
                rows.append(row)
    X = np.array(rows)
    np.testing.assert_allclose(
        model.decision_function(X), reference_model.decision_function(X), atol=1e-9
    )


def test_dense_layout_invariants(forest) -> None:
    n_trees, n_internal = forest.split_feature.shape
    assert n_internal == 2**forest.depth - 1
    assert forest.leaf_values.shape == (n_trees, 2**forest.depth)
    assert max(len(e) for e in forest.bin_edges) < MISSING_BIN
    padding = forest.split_bin == MISSING_BIN
    assert padding.any(), "shallow branches must be padded with always-left nodes"


@pytest.mark.parametrize(
    ("qtype", "max_auc_drop"),
    [(GGMLType.F32, 1e-9), (GGMLType.F16, 1e-4), (GGMLType.Q8_0, 1e-3), (GGMLType.Q4_0, 5e-3)],
)
def test_quantized_variants_keep_ranking_quality(
    reference_model, forest, test_frame, qtype, max_auc_drop
) -> None:
    X, y = featurize_frame(test_frame), test_frame[TARGET_COLUMN].to_numpy()
    reference_auc = roc_auc_score(y, reference_model.predict_proba(X)[:, 1])
    auc = roc_auc_score(y, quantize_model(forest, qtype).predict_proba(X))
    assert reference_auc - auc <= max_auc_drop


def test_gguf_round_trip_is_lossless(forest, stress_X, tmp_path: Path) -> None:
    for qtype in (GGMLType.F32, GGMLType.F16, GGMLType.Q8_0, GGMLType.Q4_0):
        model = quantize_model(forest, qtype, {"fraud.decision_threshold": 0.3})
        model.save(tmp_path / f"{qtype.name}.gguf")
        loaded = QuantizedGBDT.load(tmp_path / f"{qtype.name}.gguf")
        np.testing.assert_array_equal(loaded.predict_proba(stress_X), model.predict_proba(stress_X))
        assert loaded.metadata["fraud.decision_threshold"] == 0.3
        assert loaded.leaf_type is qtype


def test_quantization_shrinks_the_resident_footprint(forest) -> None:
    sizes = {
        q: quantize_model(forest, q).nbytes for q in (GGMLType.F32, GGMLType.Q8_0, GGMLType.Q4_0)
    }
    assert sizes[GGMLType.Q4_0] < sizes[GGMLType.Q8_0] < sizes[GGMLType.F32]


def test_bin_features_rejects_wrong_width(forest) -> None:
    with pytest.raises(ValueError, match="shape"):
        quantize_model(forest, GGMLType.F32).predict_proba(np.zeros((1, 3)))


def test_optimization_report_and_selection(model_dir: Path) -> None:
    report = json.loads((model_dir / "optimization_report.json").read_text())
    assert set(report["variants"]) == {"f32", "f16", "q8_0", "q4_0"}
    selected = report["selected"]["variant"]
    assert report["variants"][selected]["passed_gate"]
    passing = [v for v in report["variants"].values() if v["passed_gate"]]
    assert report["variants"][selected]["file_bytes"] == min(v["file_bytes"] for v in passing)
    assert report["variants"]["f32"]["max_abs_logit_error"] < 1e-4
    assert (model_dir / "model.gguf").read_bytes() == (
        model_dir / report["variants"][selected]["file"]
    ).read_bytes()


def test_gate_covers_every_risk_band(reference_model, test_frame, tmp_path: Path) -> None:
    """A strict gate must reject lossy variants and change the selection accordingly."""
    joblib.dump({"model": reference_model}, tmp_path / BASELINE_FILE)
    X, y = featurize_frame(test_frame), test_frame[TARGET_COLUMN].to_numpy()
    strict = OptimizationGate(
        max_roc_auc_drop=1e-6, max_pr_auc_drop=1e-6, min_risk_band_agreement=1.0
    )
    report = optimize_model(
        reference_model,
        X,
        y,
        decision_threshold=0.2,
        review_threshold=0.03,
        output_dir=tmp_path,
        gate=strict,
    )
    q4 = report["variants"]["q4_0"]
    assert q4["risk_band_agreement"] < 1.0 and not q4["passed_gate"]
    assert report["variants"]["f32"]["passed_gate"]
    assert report["selected"]["variant"] != "q4_0"
