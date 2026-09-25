"""Hito 2 - Train the baseline fraud model and optimize it for local / edge inference.

1. Load the training data (raw CSV, or the folder written by the prep step).
2. Out-of-time split: the most recent ``valid_fraction`` of the period is the
   validation set used to pick decision thresholds and to gate quantization.
3. Train the float64 reference model (scikit-learn HistGradientBoostingClassifier).
4. ``optimize_model``: compile the ensemble into a pointer-free dense layout,
   quantize inputs (uint8 bins, lossless) and leaf values (F32 / F16 / Q8_0 /
   Q4_0 ggml blocks inside a GGUF container), measure footprint, fidelity and
   latency of every variant against the reference, and keep the smallest one
   that passes the fidelity gate.
5. Save the artifacts under ``--model_output`` (default ``outputs/model``):
   ``model.gguf`` (selected, served model), ``variants/*.gguf``,
   ``baseline_model.joblib``, ``optimization_report.json`` and
   ``training_summary.json``.

Usage:
    python src/train_and_optimize.py --train_data data/transactions.csv --model_output outputs/model
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import pandas as pd
import sklearn
from pydantic import BaseModel, ConfigDict, Field
from sklearn.ensemble import HistGradientBoostingClassifier

from fraud_detection.data_io import read_transactions, sha256_file, write_json
from fraud_detection.features import FEATURE_NAMES, featurize_frame, schema_fingerprint
from fraud_detection.metrics import (
    alert_rate_threshold,
    classification_metrics,
    fbeta_optimal_threshold,
)
from fraud_detection.quantization import GGMLType, QuantizedVector
from fraud_detection.runtime import MISSING_BIN, QuantizedGBDT
from fraud_detection.schema import TARGET_COLUMN, TIMESTAMP_COLUMN

LOG = logging.getLogger("train_and_optimize")

MODEL_FILE: Final = "model.gguf"
BASELINE_FILE: Final = "baseline_model.joblib"
VARIANTS: Final[dict[str, GGMLType]] = {
    "f32": GGMLType.F32,
    "f16": GGMLType.F16,
    "q8_0": GGMLType.Q8_0,
    "q4_0": GGMLType.Q4_0,
}
MIN_DENSE_DEPTH: Final = 5  # 2**5 = 32 leaves per tree = one whole ggml block
# Extra share of traffic sent to the "review" band (step-up authentication, e.g. 3-D Secure).
REVIEW_TRAFFIC_BUDGET: Final = 0.03
# The compiled F32 model must reproduce the reference raw scores (it only rounds
# leaf values to float32); a larger gap means a compilation bug, not quantization.
COMPILE_TOLERANCE: Final = 1e-4


class TrainParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_iter: int = Field(default=500, ge=1, le=5_000)
    learning_rate: float = Field(default=0.08, gt=0, le=1)
    max_depth: int = Field(default=6, ge=1, le=10)
    max_leaf_nodes: int = Field(default=31, ge=2, le=1_024)
    min_samples_leaf: int = Field(default=40, ge=1)
    l2_regularization: float = Field(default=1.0, ge=0)
    n_iter_no_change: int = Field(default=30, ge=1)
    random_state: int = 42


class OptimizationGate(BaseModel):
    """Maximum degradation a compressed variant may introduce vs. the reference."""

    model_config = ConfigDict(extra="forbid")

    max_roc_auc_drop: float = Field(default=0.002, ge=0)
    max_pr_auc_drop: float = Field(default=0.005, ge=0)
    # Share of transactions that must keep the reference model's risk band (approve /
    # step-up / decline): the API exposes all three, not only the binary decision.
    min_risk_band_agreement: float = Field(default=0.999, ge=0, le=1)


@dataclass(frozen=True)
class CompiledForest:
    """Tree ensemble in dense perfect-tree layout (see ``fraud_detection.runtime``)."""

    split_feature: np.ndarray  # (n_trees, 2**depth - 1) uint8
    split_bin: np.ndarray  # (n_trees, 2**depth - 1) uint8, 255 = padding (always left)
    missing_left: np.ndarray  # (n_trees, 2**depth - 1) bool
    leaf_values: np.ndarray  # (n_trees, 2**depth) float64
    bin_edges: list[np.ndarray]  # per-feature split points actually used
    baseline: float
    depth: int


# --------------------------------------------------------------------------- data


def temporal_split(frame: pd.DataFrame, valid_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Out-of-time split: train on the past, validate on the most recent period."""
    timestamps = pd.to_datetime(frame[TIMESTAMP_COLUMN], utc=True)
    ordered = frame.iloc[np.argsort(timestamps.to_numpy(), kind="stable")]
    cut = int(len(ordered) * (1 - valid_fraction))
    return ordered.iloc[:cut].reset_index(drop=True), ordered.iloc[cut:].reset_index(drop=True)


# --------------------------------------------------------------------------- training


def train_baseline(
    X: np.ndarray, y: np.ndarray, params: TrainParams
) -> HistGradientBoostingClassifier:
    clf = HistGradientBoostingClassifier(
        loss="log_loss",
        max_iter=params.max_iter,
        learning_rate=params.learning_rate,
        max_depth=params.max_depth,
        max_leaf_nodes=params.max_leaf_nodes,
        min_samples_leaf=params.min_samples_leaf,
        l2_regularization=params.l2_regularization,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=params.n_iter_no_change,
        random_state=params.random_state,
    )
    return clf.fit(X, y)


# --------------------------------------------------------------------------- compilation


def compile_hgb(clf: HistGradientBoostingClassifier) -> CompiledForest:
    """Convert a fitted binary HistGradientBoostingClassifier into a CompiledForest.

    Split thresholds are re-expressed as bin indices over the pruned set of
    split points each feature actually uses, which is exact because
    ``x <= edges[b]`` iff ``searchsorted(edges, x, "left") <= b``.
    """
    if clf.n_trees_per_iteration_ != 1:
        raise ValueError("only binary classification is supported")
    predictors = [trees[0] for trees in clf._predictors]
    if any(p.nodes["is_categorical"].any() for p in predictors):
        raise ValueError("native categorical splits are not supported")
    thresholds = clf._bin_mapper.bin_thresholds_
    n_bins_non_missing = clf._bin_mapper.n_bins_non_missing_
    if clf._bin_mapper.missing_values_bin_idx_ != MISSING_BIN:
        raise ValueError("unexpected missing-values bin (max_bins must be 255)")
    n_features = len(thresholds)
    if n_features > 256:
        raise ValueError("feature indices must fit in uint8")

    def is_nan_split(feature: int, bin_idx: int) -> bool:
        # sklearn "split on NaN": every non-missing value goes left.
        return bin_idx >= n_bins_non_missing[feature] - 1

    used: list[set[int]] = [set() for _ in range(n_features)]
    for p in predictors:
        internal = p.nodes[p.nodes["is_leaf"] == 0]
        for feature, bin_idx in zip(
            internal["feature_idx"], internal["bin_threshold"], strict=True
        ):
            if not is_nan_split(int(feature), int(bin_idx)):
                used[int(feature)].add(int(bin_idx))
    kept = [np.array(sorted(u), dtype=np.intp) for u in used]
    edges = [np.asarray(thresholds[f], dtype=np.float64)[kept[f]] for f in range(n_features)]

    depth = max(MIN_DENSE_DEPTH, *(int(p.nodes["depth"].max()) for p in predictors))
    n_internal, n_leaves, n_trees = 2**depth - 1, 2**depth, len(predictors)
    split_feature = np.zeros((n_trees, n_internal), dtype=np.uint8)
    split_bin = np.full((n_trees, n_internal), MISSING_BIN, dtype=np.uint8)  # always left
    missing_left = np.zeros((n_trees, n_internal), dtype=bool)
    leaf_values = np.zeros((n_trees, n_leaves), dtype=np.float64)

    for t, predictor in enumerate(predictors):
        nodes = predictor.nodes
        stack = [(0, 0, 0)]  # (source node, dense position, level)
        while stack:
            src, pos, level = stack.pop()
            node = nodes[src]
            if level == depth:
                assert node["is_leaf"], "tree deeper than the dense layout"
                leaf_values[t, pos - n_internal] = node["value"]
                continue
            if node["is_leaf"]:  # pad: the leaf value flows down to both children
                stack += [(src, 2 * pos + 1, level + 1), (src, 2 * pos + 2, level + 1)]
                continue
            feature, bin_idx = int(node["feature_idx"]), int(node["bin_threshold"])
            split_feature[t, pos] = feature
            split_bin[t, pos] = (
                len(kept[feature])
                if is_nan_split(feature, bin_idx)
                else np.searchsorted(kept[feature], bin_idx)
            )
            missing_left[t, pos] = bool(node["missing_go_to_left"])
            stack += [
                (int(node["left"]), 2 * pos + 1, level + 1),
                (int(node["right"]), 2 * pos + 2, level + 1),
            ]

    return CompiledForest(
        split_feature=split_feature,
        split_bin=split_bin,
        missing_left=missing_left,
        leaf_values=leaf_values,
        bin_edges=edges,
        baseline=float(np.ravel(clf._baseline_prediction)[0]),
        depth=depth,
    )


def quantize_model(
    forest: CompiledForest, qtype: GGMLType, metadata: dict[str, Any] | None = None
) -> QuantizedGBDT:
    """GGUF-style quantization: encode leaf values as ``qtype`` blocks (Q4_0/Q8_0 = 32
    values + one fp16 scale per block). Structure and inputs are already 8-bit."""
    return QuantizedGBDT(
        split_feature=forest.split_feature,
        split_bin=forest.split_bin,
        missing_left=forest.missing_left,
        leaves=QuantizedVector.encode(forest.leaf_values.reshape(-1), qtype),
        bin_edges=forest.bin_edges,
        baseline=forest.baseline,
        feature_names=FEATURE_NAMES,
        metadata=metadata,
    )


# --------------------------------------------------------------------------- optimization


def _p50_latency_ms(fn: Callable[[np.ndarray], Any], X: np.ndarray, repeats: int) -> float:
    fn(X)  # warm-up
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn(X)
        timings.append((time.perf_counter() - start) * 1_000)
    return round(float(np.median(timings)), 4)


def _sklearn_resident_bytes(clf: HistGradientBoostingClassifier) -> int:
    trees = sum(
        p.nodes.nbytes + p.binned_left_cat_bitsets.nbytes + p.raw_left_cat_bitsets.nbytes
        for trees in clf._predictors
        for p in trees
    )
    return trees + sum(t.nbytes for t in clf._bin_mapper.bin_thresholds_)


def optimize_model(
    clf: HistGradientBoostingClassifier,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    *,
    decision_threshold: float,
    review_threshold: float,
    output_dir: Path,
    gate: OptimizationGate,
    requested: str = "auto",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile + quantize the model, benchmark every variant and export the selected one.

    Returns the optimization report (also written to ``optimization_report.json``).
    """
    forest = compile_hgb(clf)
    ref_raw = clf.decision_function(X_valid)
    ref_proba = clf.predict_proba(X_valid)[:, 1]
    ref_metrics = classification_metrics(y_valid, ref_proba, decision_threshold)
    band_edges = np.array([review_threshold, decision_threshold])
    ref_band = np.digitize(ref_proba, band_edges)  # 0 low, 1 medium (review), 2 high
    one_row, batch = X_valid[:1], X_valid[:1_000]

    baseline_path = output_dir / BASELINE_FILE
    report: dict[str, Any] = {
        "reference": {
            "model": f"sklearn.HistGradientBoostingClassifier/{sklearn.__version__} (float64)",
            "n_trees": len(clf._predictors),
            "file_bytes": baseline_path.stat().st_size,
            "resident_bytes": _sklearn_resident_bytes(clf),
            "roc_auc": ref_metrics["roc_auc"],
            "pr_auc": ref_metrics["pr_auc"],
            "latency_ms": {
                "single_row_p50": _p50_latency_ms(lambda x: clf.predict_proba(x), one_row, 200),
                "batch_1000_p50": _p50_latency_ms(lambda x: clf.predict_proba(x), batch, 20),
            },
        },
        "layout": {
            "dense_depth": forest.depth,
            "internal_nodes_per_tree": 2**forest.depth - 1,
            "leaves_per_tree": 2**forest.depth,
            "kept_split_points": int(sum(len(e) for e in forest.bin_edges)),
            "input_quantization": "uint8 bins (lossless w.r.t. trained split points)",
        },
        "gate": gate.model_dump(),
        "variants": {},
    }

    variants_dir = output_dir / "variants"
    variants_dir.mkdir(parents=True, exist_ok=True)
    for name, qtype in VARIANTS.items():
        model = quantize_model(forest, qtype, metadata)
        raw = model.decision_function(X_valid)
        proba = model.predict_proba(X_valid)
        if (
            qtype is GGMLType.F32
            and (gap := float(np.abs(raw - ref_raw).max())) > COMPILE_TOLERANCE
        ):
            raise RuntimeError(f"compiled model diverges from the reference (max gap {gap:.2e})")
        metrics = classification_metrics(y_valid, proba, decision_threshold)
        path = variants_dir / f"fraud_gbdt.{name}.gguf"
        file_bytes = model.save(
            path,
            extra_metadata={
                "fraud.variant": name,
                "fraud.validation.roc_auc": metrics["roc_auc"],
                "fraud.validation.pr_auc": metrics["pr_auc"],
            },
        )
        reloaded = QuantizedGBDT.load(path)
        if not np.array_equal(reloaded.predict_proba(X_valid), proba):
            raise RuntimeError(f"{path.name}: GGUF round trip changed the predictions")
        roc_drop = ref_metrics["roc_auc"] - metrics["roc_auc"]
        pr_drop = ref_metrics["pr_auc"] - metrics["pr_auc"]
        band_agreement = float(np.mean(np.digitize(proba, band_edges) == ref_band))
        proba_error = np.abs(proba - ref_proba)
        report["variants"][name] = {
            "file": str(path.relative_to(output_dir)),
            "leaf_type": qtype.name,
            "file_bytes": file_bytes,
            "resident_bytes": reloaded.nbytes,
            "compression_vs_reference": round(
                report["reference"]["resident_bytes"] / reloaded.nbytes, 2
            ),
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "roc_auc_drop": roc_drop,
            "pr_auc_drop": pr_drop,
            "max_abs_logit_error": float(np.abs(raw - ref_raw).max()),
            "p99_abs_proba_error": float(np.quantile(proba_error, 0.99)),
            "max_abs_proba_error": float(proba_error.max()),
            "decision_agreement": float(np.mean((proba >= decision_threshold) == (ref_band == 2))),
            "risk_band_agreement": band_agreement,
            "latency_ms": {
                "single_row_p50": _p50_latency_ms(reloaded.predict_proba, one_row, 200),
                "batch_1000_p50": _p50_latency_ms(reloaded.predict_proba, batch, 20),
            },
            "passed_gate": bool(
                roc_drop <= gate.max_roc_auc_drop
                and pr_drop <= gate.max_pr_auc_drop
                and band_agreement >= gate.min_risk_band_agreement
            ),
        }
        LOG.info(
            "%-5s %7d B  roc_auc=%.5f pr_auc=%.5f risk_band_agreement=%.5f gate=%s",
            name, file_bytes, metrics["roc_auc"], metrics["pr_auc"], band_agreement,
            report["variants"][name]["passed_gate"],
        )  # fmt: skip

    variants = report["variants"]
    if requested == "auto":
        passing = [n for n in VARIANTS if variants[n]["passed_gate"]]
        selected = min(passing, key=lambda n: variants[n]["file_bytes"]) if passing else "f32"
        if not passing:
            LOG.warning("No compressed variant passed the gate; falling back to f32")
    else:
        selected = requested
        if not variants[selected]["passed_gate"]:
            LOG.warning("Requested variant %s does NOT pass the fidelity gate", selected)
    shutil.copyfile(variants_dir / f"fraud_gbdt.{selected}.gguf", output_dir / MODEL_FILE)
    report["selected"] = {
        "variant": selected,
        "file": MODEL_FILE,
        "sha256": sha256_file(output_dir / MODEL_FILE),
        "size_reduction_vs_reference_file": round(
            report["reference"]["file_bytes"] / variants[selected]["file_bytes"], 1
        ),
    }
    write_json(output_dir / "optimization_report.json", report)
    return report


# --------------------------------------------------------------------------- CLI


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train_data", type=Path, default=Path("data/transactions.csv"))
    parser.add_argument("--model_output", type=Path, default=Path("outputs/model"))
    parser.add_argument("--model_name", default="fraud-detection-gbdt")
    parser.add_argument("--valid_fraction", type=float, default=0.15)
    parser.add_argument("--quantization", choices=["auto", *VARIANTS], default="auto")
    gate, params = OptimizationGate(), TrainParams()
    parser.add_argument("--max_roc_auc_drop", type=float, default=gate.max_roc_auc_drop)
    parser.add_argument("--max_pr_auc_drop", type=float, default=gate.max_pr_auc_drop)
    parser.add_argument(
        "--min_risk_band_agreement", type=float, default=gate.min_risk_band_agreement
    )
    for field, info in TrainParams.model_fields.items():
        parser.add_argument(f"--{field}", type=type(info.default), default=getattr(params, field))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    args = parse_args(argv)
    params = TrainParams(**{f: getattr(args, f) for f in TrainParams.model_fields})
    gate = OptimizationGate(
        max_roc_auc_drop=args.max_roc_auc_drop,
        max_pr_auc_drop=args.max_pr_auc_drop,
        min_risk_band_agreement=args.min_risk_band_agreement,
    )
    output_dir: Path = args.model_output
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = read_transactions(args.train_data, preferred="train.csv")
    fit_frame, valid_frame = temporal_split(frame, args.valid_fraction)
    X_fit, y_fit = featurize_frame(fit_frame), fit_frame[TARGET_COLUMN].to_numpy()
    X_valid, y_valid = featurize_frame(valid_frame), valid_frame[TARGET_COLUMN].to_numpy()
    LOG.info(
        "Data: %d fit rows (fraud %.3f%%), %d validation rows (fraud %.3f%%), %d features",
        len(y_fit), 100 * y_fit.mean(), len(y_valid), 100 * y_valid.mean(), X_fit.shape[1],
    )  # fmt: skip

    started = time.perf_counter()
    clf = train_baseline(X_fit, y_fit, params)
    train_seconds = time.perf_counter() - started
    valid_proba = clf.predict_proba(X_valid)[:, 1]
    decision_threshold = fbeta_optimal_threshold(y_valid, valid_proba, beta=2.0)
    valid_metrics = classification_metrics(y_valid, valid_proba, decision_threshold)
    review_threshold = min(
        decision_threshold,
        alert_rate_threshold(valid_proba, valid_metrics["alert_rate"] + REVIEW_TRAFFIC_BUDGET),
    )
    LOG.info(
        "Reference model: %d trees in %.1fs | valid ROC-AUC=%.4f PR-AUC=%.4f | "
        "threshold=%.4f precision=%.3f recall=%.3f",
        clf.n_iter_, train_seconds, valid_metrics["roc_auc"], valid_metrics["pr_auc"],
        decision_threshold, valid_metrics["precision"], valid_metrics["recall"],
    )  # fmt: skip

    trained_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    joblib.dump(
        {"model": clf, "feature_names": FEATURE_NAMES, "decision_threshold": decision_threshold},
        output_dir / BASELINE_FILE,
    )
    metadata = {
        "general.name": args.model_name,
        "general.description": "Card-fraud detector: gradient-boosted trees, quantized leaves",
        "fraud.schema_fingerprint": schema_fingerprint(),
        "fraud.decision_threshold": decision_threshold,
        "fraud.review_threshold": review_threshold,
        "fraud.trained_at": trained_at,
        "fraud.source_model": f"sklearn.HistGradientBoostingClassifier/{sklearn.__version__}",
        "fraud.training_rows": len(y_fit),
    }
    report = optimize_model(
        clf,
        X_valid,
        y_valid,
        decision_threshold=decision_threshold,
        review_threshold=review_threshold,
        output_dir=output_dir,
        gate=gate,
        requested=args.quantization,
        metadata=metadata,
    )
    write_json(
        output_dir / "training_summary.json",
        {
            "model_name": args.model_name,
            "trained_at": trained_at,
            "train_seconds": round(train_seconds, 2),
            "params": params.model_dump(),
            "n_trees": int(clf.n_iter_),
            "features": list(FEATURE_NAMES),
            "schema_fingerprint": schema_fingerprint(),
            "data": {
                "source": str(args.train_data),
                "fit_rows": len(y_fit),
                "valid_rows": len(y_valid),
                "fit_fraud_rate": float(y_fit.mean()),
                "valid_fraud_rate": float(y_valid.mean()),
            },
            "thresholds": {"decision": decision_threshold, "review": review_threshold},
            "validation_metrics": valid_metrics,
            "selected_variant": report["selected"],
        },
    )
    selected = report["variants"][report["selected"]["variant"]]
    LOG.info(
        "Selected %s: %d bytes on disk (reference %d, %.1fx smaller), resident %d bytes, "
        "single-row latency %.3f ms (reference %.3f ms)",
        report["selected"]["variant"], selected["file_bytes"], report["reference"]["file_bytes"],
        report["selected"]["size_reduction_vs_reference_file"], selected["resident_bytes"],
        selected["latency_ms"]["single_row_p50"],
        report["reference"]["latency_ms"]["single_row_p50"],
    )  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
