"""Pipeline step 4 - Model registration.

Registers the model package (served ``model.gguf`` + lineage/metrics reports)
only when the evaluation gate passed; otherwise records why it was skipped.
Backend: Azure ML model registry inside Azure ML jobs, local file registry
elsewhere (override with ``--registry`` or ``FRAUD_REGISTRY``).
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
from pathlib import Path

from fraud_detection.data_io import read_json, write_json
from fraud_detection.registry import get_registry

LOG = logging.getLogger("register_model")
PACKAGE_FILES = ("model.gguf", "optimization_report.json", "training_summary.json")


def build_tags(evaluation: dict) -> dict[str, str]:
    model, metrics = evaluation["model"], evaluation["test_metrics"]
    tags = {
        "variant": str(model["variant"]),
        "leaf_type": model["leaf_type"],
        "sha256": model["sha256"],
        "decision_threshold": f"{model['decision_threshold']:.6f}",
        "file_bytes": str(model["file_bytes"]),
    }
    tags |= {k: f"{metrics[k]:.6f}" for k in ("roc_auc", "pr_auc", "recall", "precision")}
    return tags


def register(
    model_dir: Path, evaluation_dir: Path, model_name: str, registry_kind: str | None
) -> dict:
    evaluation = read_json(evaluation_dir / "evaluation.json")
    if not evaluation["gate"]["passed"]:
        failed = [c["name"] for c in evaluation["gate"]["checks"] if not c["passed"]]
        LOG.warning("Release gate failed (%s): model NOT registered", ", ".join(failed))
        return {"status": "skipped", "reason": f"release gate failed: {failed}"}

    registry = get_registry(registry_kind)
    with tempfile.TemporaryDirectory(prefix="model-package-") as staging:
        package = Path(staging)
        for name in PACKAGE_FILES:
            shutil.copy2(model_dir / name, package / name)
        shutil.copy2(evaluation_dir / "evaluation.json", package / "evaluation.json")
        registered = registry.register(
            model_name,
            package,
            build_tags(evaluation),
            description="Card-fraud GBDT, quantized leaves (GGUF), numpy runtime",
        )
    LOG.info(
        "Registered %s:%s in the %s registry", registered.name, registered.version, registry.backend
    )
    return {
        "status": "registered",
        "name": registered.name,
        "version": registered.version,
        "backend": registry.backend,
        "location": registered.location,
        "tags": registered.tags,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model_input", type=Path, required=True)
    parser.add_argument("--evaluation_input", type=Path, required=True)
    parser.add_argument("--registration_output", type=Path, required=True)
    parser.add_argument("--model_name", default="fraud-detection-gbdt")
    parser.add_argument("--registry", choices=["auto", "local", "azureml"], default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    args = parse_args(argv)
    result = register(args.model_input, args.evaluation_input, args.model_name, args.registry)
    write_json(args.registration_output / "registration.json", result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
