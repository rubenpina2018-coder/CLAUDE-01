"""Pipeline step 3 - Evaluation and quality gate.

Scores the out-of-time hold-out set with the artifact that will actually be
served (``model.gguf`` on the numpy runtime) and applies the release gate:

* absolute quality: ROC-AUC, PR-AUC, recall at the decision threshold;
* operations: alert rate (analyst capacity) and single-row latency SLO;
* fidelity: PR-AUC of the served model vs. the float64 reference model;
* champion/challenger: the current champion (latest registered version) is
  re-scored on the *same* test set; the challenger must not be worse.

Writes ``evaluation.json`` (consumed by the register step) and ``evaluation_report.md``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from fraud_detection.data_io import read_transactions, sha256_file, write_json
from fraud_detection.features import featurize_frame, schema_fingerprint
from fraud_detection.metrics import classification_metrics
from fraud_detection.registry import get_registry
from fraud_detection.runtime import QuantizedGBDT
from fraud_detection.schema import SCENARIO_COLUMN, TARGET_COLUMN
from fraud_detection.tracking import log_metrics

LOG = logging.getLogger("evaluate")


class ReleaseGate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_roc_auc: float = Field(default=0.90, ge=0, le=1)
    min_pr_auc: float = Field(default=0.60, ge=0, le=1)
    min_recall: float = Field(default=0.70, ge=0, le=1)
    max_alert_rate: float = Field(default=0.05, ge=0, le=1)
    max_reference_pr_auc_drop: float = Field(default=0.01, ge=0)
    max_p99_latency_ms: float = Field(default=50.0, gt=0)
    champion_tolerance: float = Field(default=0.005, ge=0)


def _latency_ms(model: QuantizedGBDT, X: np.ndarray, n: int = 300) -> dict[str, float]:
    rows = X[np.random.default_rng(0).integers(0, len(X), n)]
    model.predict_proba(rows[:1])  # warm-up
    timings = []
    for i in range(n):
        start = time.perf_counter()
        model.predict_proba(rows[i : i + 1])
        timings.append((time.perf_counter() - start) * 1_000)
    return {
        "single_row_p50": float(np.percentile(timings, 50)),
        "single_row_p99": float(np.percentile(timings, 99)),
    }


def _slices(frame: Any, proba: np.ndarray, threshold: float) -> dict[str, Any]:
    scored = frame.assign(_flag=proba >= threshold)
    frauds = scored[scored[TARGET_COLUMN] == 1]
    by_scenario = {}
    if SCENARIO_COLUMN in frauds:
        by_scenario = {
            str(name): {"frauds": len(group), "recall": float(group["_flag"].mean())}
            for name, group in frauds.groupby(SCENARIO_COLUMN)
        }
    by_channel = {
        str(name): {
            "rows": len(group),
            "alert_rate": float(group["_flag"].mean()),
            "fraud_rate": float(group[TARGET_COLUMN].mean()),
        }
        for name, group in scored.groupby("channel")
    }
    return {"recall_by_scenario": by_scenario, "by_channel": by_channel}


def score_champion(
    model_name: str, X: np.ndarray, y: np.ndarray, threshold: float
) -> dict[str, Any] | None:
    """Re-score the latest registered version on the current test set."""
    registry = get_registry()
    champion = registry.latest(model_name)
    if champion is None:
        LOG.info(
            "No registered version of %s yet: the challenger becomes the first champion", model_name
        )
        return None
    with tempfile.TemporaryDirectory() as tmp:
        folder = registry.download(champion, Path(tmp))
        model = QuantizedGBDT.load(folder / "model.gguf")
        if model.metadata.get("fraud.schema_fingerprint") != schema_fingerprint():
            LOG.warning(
                "Champion v%s uses another feature schema; not comparable", champion.version
            )
            return {"version": champion.version, "comparable": False}
        metrics = classification_metrics(y, model.predict_proba(X), threshold)
    return {
        "version": champion.version,
        "backend": registry.backend,
        "comparable": True,
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
    }


def evaluate(
    model_dir: Path, test_path: Path, gate: ReleaseGate, model_name: str, compare_champion: bool
) -> dict[str, Any]:
    model = QuantizedGBDT.load(model_dir / "model.gguf")
    if model.metadata.get("fraud.schema_fingerprint") != schema_fingerprint():
        raise RuntimeError("model.gguf was built for a different feature schema")
    threshold = float(model.metadata["fraud.decision_threshold"])
    frame = read_transactions(test_path, preferred="test.csv")
    X, y = featurize_frame(frame), frame[TARGET_COLUMN].to_numpy()

    proba = model.predict_proba(X)
    metrics = classification_metrics(y, proba, threshold)
    reference = joblib.load(model_dir / "baseline_model.joblib")["model"]
    reference_metrics = classification_metrics(y, reference.predict_proba(X)[:, 1], threshold)
    latency = _latency_ms(model, X)
    champion = score_champion(model_name, X, y, threshold) if compare_champion else None

    checks = [
        ("roc_auc", metrics["roc_auc"], ">=", gate.min_roc_auc),
        ("pr_auc", metrics["pr_auc"], ">=", gate.min_pr_auc),
        ("recall", metrics["recall"], ">=", gate.min_recall),
        ("alert_rate", metrics["alert_rate"], "<=", gate.max_alert_rate),
        (
            "pr_auc_drop_vs_reference",
            reference_metrics["pr_auc"] - metrics["pr_auc"],
            "<=",
            gate.max_reference_pr_auc_drop,
        ),
        ("single_row_p99_latency_ms", latency["single_row_p99"], "<=", gate.max_p99_latency_ms),
    ]
    if champion and champion.get("comparable"):
        checks.append(
            (
                f"pr_auc_vs_champion_v{champion['version']}",
                metrics["pr_auc"] - champion["pr_auc"],
                ">=",
                -gate.champion_tolerance,
            )
        )
    gate_checks = [
        {
            "name": name,
            "value": float(value),
            "operator": op,
            "threshold": float(limit),
            "passed": bool(value >= limit if op == ">=" else value <= limit),
        }
        for name, value, op, limit in checks
    ]
    return {
        "model": {
            "name": model_name,
            "file": "model.gguf",
            "sha256": sha256_file(model_dir / "model.gguf"),
            "variant": model.metadata.get("fraud.variant"),
            "leaf_type": model.leaf_type.name,
            "file_bytes": (model_dir / "model.gguf").stat().st_size,
            "resident_bytes": model.nbytes,
            "decision_threshold": threshold,
            "review_threshold": float(model.metadata.get("fraud.review_threshold", threshold)),
        },
        "test_metrics": metrics,
        "reference_metrics": reference_metrics,
        "slices": _slices(frame, proba, threshold),
        "latency_ms": latency,
        "champion": champion,
        "gate": {
            "passed": all(c["passed"] for c in gate_checks),
            "checks": gate_checks,
            "config": gate.model_dump(),
        },
    }


def render_markdown(result: dict[str, Any]) -> str:
    m, model = result["test_metrics"], result["model"]
    lines = [
        f"# Evaluation — {model['name']} ({model['leaf_type']})",
        "",
        f"**Release gate: {'PASSED' if result['gate']['passed'] else 'FAILED'}**",
        "",
        "| Check | Value | Rule | Result |",
        "|---|---|---|---|",
        *(
            f"| {c['name']} | {c['value']:.4f} | {c['operator']} {c['threshold']:.4f} | "
            f"{'pass' if c['passed'] else 'FAIL'} |"
            for c in result["gate"]["checks"]
        ),
        "",
        f"Test set: {m['n_samples']} transactions, fraud rate {m['positive_rate']:.3%}. "
        f"Threshold {m['threshold']:.4f}: precision {m['precision']:.3f}, "
        f"recall {m['recall']:.3f}, alert rate {m['alert_rate']:.3%}.",
        "",
        "| Fraud scenario | Frauds | Recall |",
        "|---|---|---|",
        *(
            f"| {name} | {s['frauds']} | {s['recall']:.3f} |"
            for name, s in result["slices"]["recall_by_scenario"].items()
        ),
    ]
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model_input", type=Path, required=True)
    parser.add_argument("--test_data", type=Path, required=True)
    parser.add_argument("--evaluation_output", type=Path, required=True)
    parser.add_argument("--model_name", default="fraud-detection-gbdt")
    parser.add_argument(
        "--compare_champion", choices=["true", "false"], default="true",
        help="re-score the latest registered version on the same test set",
    )  # fmt: skip
    for field, info in ReleaseGate.model_fields.items():
        parser.add_argument(f"--{field}", type=float, default=info.default)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    args = parse_args(argv)
    gate = ReleaseGate(**{f: getattr(args, f) for f in ReleaseGate.model_fields})
    result = evaluate(
        args.model_input, args.test_data, gate, args.model_name, args.compare_champion == "true"
    )
    args.evaluation_output.mkdir(parents=True, exist_ok=True)
    write_json(args.evaluation_output / "evaluation.json", result)
    (args.evaluation_output / "evaluation_report.md").write_text(render_markdown(result))
    log_metrics(result["test_metrics"], prefix="test_")
    log_metrics(result["latency_ms"], prefix="latency_")
    for check in result["gate"]["checks"]:
        LOG.info(
            "%-32s %.4f %s %.4f -> %s", check["name"], check["value"], check["operator"],
            check["threshold"], "pass" if check["passed"] else "FAIL",
        )  # fmt: skip
    LOG.info("Release gate %s", "PASSED" if result["gate"]["passed"] else "FAILED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
