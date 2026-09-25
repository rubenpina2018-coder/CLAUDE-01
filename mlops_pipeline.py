"""Hito 3 - Automated MLOps pipeline on Azure Machine Learning (SDK v2, ``azure-ai-ml``).

    prep_data ──► train_and_optimize ──► evaluate ──► register_model
        │                                    ▲
        └────────────── test_data ───────────┘

The pipeline is defined once (command components + ``@dsl.pipeline``) and can be:

* ``--mode azure``     validated and submitted to an Azure ML workspace. Without
  (default)            credentials/workspace it degrades to the dry run below instead
                       of failing (``--strict`` makes it exit with an error instead).
* ``--mode validate``  validated offline with the SDK's public ``jobs.validate`` and
                       "submitted" to a strict autospec of ``MLClient`` that enforces
                       the real SDK signatures. No network needed.
* ``--mode local``     executed on this machine from the *same* definition: the job is
                       serialized with ``PipelineJob.dump`` and each component command
                       is rendered with local paths and run in DAG order.

Workspace coordinates (``--mode azure``): ``--subscription-id/--resource-group/
--workspace``, or ``AZURE_SUBSCRIPTION_ID``/``AZURE_RESOURCE_GROUP``/
``AZUREML_WORKSPACE_NAME``, or a ``config.json`` downloaded from the Azure portal.
Authentication: ``DefaultAzureCredential`` (``az login``, service principal
variables, workload identity federation in CI, managed identity).
"""

# NOTE: no ``from __future__ import annotations`` here: ``@dsl.pipeline`` reads the
# parameter annotations at runtime to derive the pipeline input types.
import argparse
import inspect
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final
from unittest.mock import DEFAULT, MagicMock, create_autospec

import yaml
from azure.ai.ml import Input, MLClient, Output, command, dsl
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import AmlCompute, Environment, PipelineJob, UserIdentityConfiguration
from azure.ai.ml.operations import (
    ComputeOperations,
    EnvironmentOperations,
    JobOperations,
    ModelOperations,
)
from azure.core.exceptions import AzureError, ResourceNotFoundError
from azure.identity import CredentialUnavailableError, DefaultAzureCredential

ROOT: Final = Path(__file__).resolve().parent
CODE_DIR: Final = ROOT / "src"
OUTPUTS: Final = ROOT / "outputs"
LOG = logging.getLogger("mlops_pipeline")

# Pinned Azure ML base image (see mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04 tags).
AML_BASE_IMAGE: Final = "mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:20260920.v1"
SERVERLESS: Final = "serverless"
PIPELINE_NAME: Final = "fraud_detection_training"
ARM_SCOPE: Final = "https://management.azure.com/.default"


# --------------------------------------------------------------------------- definition


def read_requirements(path: Path) -> list[str]:
    """Flatten a pip requirements file (following ``-r`` includes)."""
    requirements: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.split(" #", 1)[0].strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r "):
            requirements += read_requirements(path.parent / line[3:].strip())
        else:
            requirements.append(line)
    return requirements


def build_environment() -> Environment:
    """Job environment: pinned Azure ML base image + pinned pip dependencies.

    Kept anonymous on purpose: Azure ML content-addresses anonymous environments,
    so the image is rebuilt only when the dependency spec actually changes.
    """
    return Environment(
        image=AML_BASE_IMAGE,
        conda_file={
            "name": "fraud-detection-train",
            "channels": ["conda-forge"],
            "dependencies": [
                "python=3.11",
                "pip",
                {"pip": read_requirements(ROOT / "requirements" / "azureml.txt")},
            ],
        },
        description="Runtime for the fraud-detection training pipeline",
    )


def build_components(environment: Environment | str) -> dict[str, Any]:
    """Command components. Each one runs a script of ``src/`` (uploaded as code snapshot)."""
    common = {"code": str(CODE_DIR), "environment": environment}
    return {
        "prep_data": command(
            name="fraud_prep_data",
            display_name="Prepare data",
            description="Data-contract validation, de-duplication and out-of-time split.",
            inputs={
                "raw_data": Input(type=AssetTypes.URI_FILE, description="Raw transactions CSV"),
                "test_fraction": Input(type="number", default=0.2),
                "max_invalid_fraction": Input(type="number", default=0.01),
            },
            outputs={
                "train_data": Output(type=AssetTypes.URI_FOLDER),
                "test_data": Output(type=AssetTypes.URI_FOLDER),
            },
            command=(
                "python prep_data.py --raw_data ${{inputs.raw_data}}"
                " --test_fraction ${{inputs.test_fraction}}"
                " --max_invalid_fraction ${{inputs.max_invalid_fraction}}"
                " --train_data ${{outputs.train_data}} --test_data ${{outputs.test_data}}"
            ),
            **common,
        ),
        "train_and_optimize": command(
            name="fraud_train_and_optimize",
            display_name="Train and optimize (quantize)",
            description="Trains the GBDT reference model and exports a quantized GGUF artifact.",
            inputs={
                "train_data": Input(type=AssetTypes.URI_FOLDER),
                "model_name": Input(type="string"),
                "valid_fraction": Input(type="number", default=0.15),
                "quantization": Input(type="string", default="auto"),
            },
            outputs={"model_output": Output(type=AssetTypes.URI_FOLDER)},
            command=(
                "python train_and_optimize.py --train_data ${{inputs.train_data}}"
                " --model_name ${{inputs.model_name}} --valid_fraction ${{inputs.valid_fraction}}"
                " --quantization ${{inputs.quantization}} --model_output ${{outputs.model_output}}"
            ),
            **common,
        ),
        "evaluate": command(
            name="fraud_evaluate",
            display_name="Evaluate (release gate)",
            description="Hold-out metrics, fidelity vs. reference, latency, champion/challenger.",
            inputs={
                "model_input": Input(type=AssetTypes.URI_FOLDER),
                "test_data": Input(type=AssetTypes.URI_FOLDER),
                "model_name": Input(type="string"),
                "min_roc_auc": Input(type="number", default=0.90),
                "min_pr_auc": Input(type="number", default=0.60),
                "compare_champion": Input(type="string", default="true"),
            },
            outputs={"evaluation_output": Output(type=AssetTypes.URI_FOLDER)},
            command=(
                "python evaluate.py --model_input ${{inputs.model_input}}"
                " --test_data ${{inputs.test_data}} --model_name ${{inputs.model_name}}"
                " --min_roc_auc ${{inputs.min_roc_auc}} --min_pr_auc ${{inputs.min_pr_auc}}"
                " --compare_champion ${{inputs.compare_champion}}"
                " --evaluation_output ${{outputs.evaluation_output}}"
            ),
            # Reads the model registry (champion): its result must never be reused from cache.
            is_deterministic=False,
            **common,
        ),
        "register_model": command(
            name="fraud_register_model",
            display_name="Register model",
            description="Registers the model package when the release gate passed.",
            inputs={
                "model_input": Input(type=AssetTypes.URI_FOLDER),
                "evaluation_input": Input(type=AssetTypes.URI_FOLDER),
                "model_name": Input(type="string"),
            },
            outputs={"registration_output": Output(type=AssetTypes.URI_FOLDER)},
            command=(
                "python register_model.py --model_input ${{inputs.model_input}}"
                " --evaluation_input ${{inputs.evaluation_input}}"
                " --model_name ${{inputs.model_name}}"
                " --registration_output ${{outputs.registration_output}}"
            ),
            is_deterministic=False,
            **common,
        ),
    }


@dataclass(frozen=True)
class PipelineParams:
    model_name: str = "fraud-detection-gbdt"
    quantization: str = "auto"
    test_fraction: float = 0.2
    valid_fraction: float = 0.15
    min_roc_auc: float = 0.90
    min_pr_auc: float = 0.60


def build_pipeline(
    components: dict[str, Any], raw_data: Input, compute: str, params: PipelineParams
) -> PipelineJob:
    @dsl.pipeline(
        name=PIPELINE_NAME,
        display_name="Fraud detection: prep → train+quantize → evaluate → register",
        description="Trains, quantizes (GGUF Q4_0/Q8_0), gates and registers the fraud model.",
        compute=compute,
        tags={"project": "fraud-detection", "framework": "scikit-learn", "artifact": "gguf"},
    )
    def fraud_detection_pipeline(
        raw_data: Input,
        model_name: str,
        quantization: str,
        test_fraction: float,
        valid_fraction: float,
        min_roc_auc: float,
        min_pr_auc: float,
    ) -> dict[str, Any]:
        prep_data = components["prep_data"](raw_data=raw_data, test_fraction=test_fraction)
        train_and_optimize = components["train_and_optimize"](
            train_data=prep_data.outputs.train_data,
            model_name=model_name,
            valid_fraction=valid_fraction,
            quantization=quantization,
        )
        evaluate = components["evaluate"](
            model_input=train_and_optimize.outputs.model_output,
            test_data=prep_data.outputs.test_data,
            model_name=model_name,
            min_roc_auc=min_roc_auc,
            min_pr_auc=min_pr_auc,
        )
        register_model = components["register_model"](
            model_input=train_and_optimize.outputs.model_output,
            evaluation_input=evaluate.outputs.evaluation_output,
            model_name=model_name,
        )
        # Registry access runs as the submitting user (on-behalf-of token in the job).
        evaluate.identity = UserIdentityConfiguration()
        register_model.identity = UserIdentityConfiguration()
        return {
            "model": train_and_optimize.outputs.model_output,
            "evaluation": evaluate.outputs.evaluation_output,
            "registration": register_model.outputs.registration_output,
        }

    job = fraud_detection_pipeline(
        raw_data=raw_data,
        model_name=params.model_name,
        quantization=params.quantization,
        test_fraction=params.test_fraction,
        valid_fraction=params.valid_fraction,
        min_roc_auc=params.min_roc_auc,
        min_pr_auc=params.min_pr_auc,
    )
    job.settings.continue_on_step_failure = False
    return job


# --------------------------------------------------------------------------- validation


class OfflineCredential:
    """TokenCredential that never authenticates: lets MLClient run offline validation."""

    def get_token(self, *scopes: str, **kwargs: Any) -> Any:
        raise CredentialUnavailableError(message="offline validation: no Azure credential")


def offline_ml_client() -> MLClient:
    return MLClient(
        credential=OfflineCredential(),
        subscription_id="00000000-0000-0000-0000-000000000000",
        resource_group_name="offline-validation",
        workspace_name="offline-validation",
    )


@dataclass
class ValidationReport:
    passed: bool
    errors: dict[str, str] = field(default_factory=dict)
    deferred: dict[str, str] = field(default_factory=dict)


def validate_pipeline(job: PipelineJob, client: MLClient, offline: bool) -> ValidationReport:
    """Validation through the SDK's public APIs, both usable without a workspace.

    ``components.validate`` checks every component (command data bindings, I/O
    declarations); ``jobs.validate`` checks the pipeline graph (schema, required
    inputs, bindings between steps). Offline, a named compute cannot be resolved:
    that single check is deferred to submission instead of reported as a failure.
    """
    errors: dict[str, str] = {}
    for name, node in job.jobs.items():
        result = client.components.validate(node.component)
        errors |= {
            f"jobs.{name}.component.{k}": v for k, v in (result.error_messages or {}).items()
        }
    errors |= dict(client.jobs.validate(job).error_messages or {})
    deferred = {k: v for k, v in errors.items() if offline and k.split(".")[-1] == "compute"}
    remaining = {k: v for k, v in errors.items() if k not in deferred}
    return ValidationReport(passed=not remaining, errors=remaining, deferred=deferred)


# --------------------------------------------------------------------------- submission


def strict_autospec(spec: type) -> Any:
    """``create_autospec`` that also rejects keyword arguments the real method does not
    declare (SDK methods accept ``**kwargs``, which would hide typos)."""
    mock = create_autospec(spec, instance=True)
    for name, member in inspect.getmembers(spec, inspect.isfunction):
        if name.startswith("_"):
            continue
        signature = inspect.signature(member)
        explicit = signature.replace(
            parameters=[
                p
                for p in signature.parameters.values()
                if p.kind is not p.VAR_KEYWORD and p.name != "self"
            ]
        )

        def check(*args: Any, _sig: inspect.Signature = explicit, **kwargs: Any) -> Any:
            _sig.bind(*args, **kwargs)  # TypeError on anything the real SDK would reject
            return DEFAULT

        getattr(mock, name).side_effect = check
    return mock


def dry_run_ml_client() -> MagicMock:
    """MLClient stand-in with the real SDK signatures; records every call it receives."""
    client = MagicMock(spec=MLClient, name="MLClient(dry-run)")
    client.jobs = strict_autospec(JobOperations)
    client.compute = strict_autospec(ComputeOperations)
    client.environments = strict_autospec(EnvironmentOperations)
    client.models = strict_autospec(ModelOperations)
    validate_signature = client.jobs.create_or_update.side_effect

    def submit(job: PipelineJob, **kwargs: Any) -> SimpleNamespace:
        validate_signature(job, **kwargs)
        return SimpleNamespace(
            name=f"dryrun-{uuid.uuid4().hex[:10]}",
            display_name=job.display_name,
            experiment_name=kwargs.get("experiment_name"),
            status="NotSubmitted",
            studio_url=None,
        )

    client.jobs.create_or_update.side_effect = submit
    return client


def ensure_compute(client: MLClient, name: str, size: str, max_instances: int) -> None:
    try:
        client.compute.get(name)
    except ResourceNotFoundError:
        LOG.info("Creating compute cluster %s (%s, 0..%d nodes)", name, size, max_instances)
        cluster = AmlCompute(
            name=name,
            size=size,
            min_instances=0,
            max_instances=max_instances,
            idle_time_before_scale_down=900,
            tier="Dedicated",
        )
        client.compute.begin_create_or_update(cluster).result()


def submit_pipeline(
    client: MLClient, job: PipelineJob, *, experiment: str, compute: str, stream: bool
) -> Any:
    if compute != SERVERLESS:
        ensure_compute(client, compute, size="Standard_DS3_v2", max_instances=4)
    submitted = client.jobs.create_or_update(job, experiment_name=experiment)
    LOG.info(
        "Pipeline job %s created in experiment %s (status: %s)",
        submitted.name, experiment, submitted.status,
    )  # fmt: skip
    if submitted.studio_url:
        LOG.info("Azure ML Studio: %s", submitted.studio_url)
    if stream:
        client.jobs.stream(submitted.name)
    return submitted


@dataclass
class WorkspaceSettings:
    subscription_id: str | None = None
    resource_group: str | None = None
    workspace: str | None = None
    config_path: Path | None = None

    @classmethod
    def resolve(cls, args: argparse.Namespace) -> "WorkspaceSettings":
        return cls(
            subscription_id=args.subscription_id or os.environ.get("AZURE_SUBSCRIPTION_ID"),
            resource_group=args.resource_group or os.environ.get("AZURE_RESOURCE_GROUP"),
            workspace=args.workspace or os.environ.get("AZUREML_WORKSPACE_NAME"),
            config_path=args.config,
        )

    def has_coordinates(self) -> bool:
        return bool(self.subscription_id and self.resource_group and self.workspace)

    def config_file(self) -> Path | None:
        candidates = [self.config_path] if self.config_path else [
            ROOT / "config.json", ROOT / ".azureml" / "config.json",
        ]  # fmt: skip
        return next((p for p in candidates if p and p.is_file()), None)


def connect(settings: WorkspaceSettings) -> MLClient | None:
    """Connect to the workspace, or return None (with the reason logged) if impossible."""
    if not settings.has_coordinates() and settings.config_file() is None:
        LOG.warning(
            "No Azure ML workspace configured (set AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP "
            "and AZUREML_WORKSPACE_NAME, pass --subscription-id/--resource-group/--workspace, "
            "or provide config.json)"
        )
        return None
    try:
        credential = DefaultAzureCredential(exclude_interactive_browser_credential=True)
        credential.get_token(ARM_SCOPE)  # fail fast on missing/expired credentials
        if settings.has_coordinates():
            client = MLClient(
                credential,
                subscription_id=settings.subscription_id,
                resource_group_name=settings.resource_group,
                workspace_name=settings.workspace,
            )
        else:
            client = MLClient.from_config(credential=credential, path=settings.config_file())
        client.workspaces.get(client.workspace_name)  # verifies RBAC access
        LOG.info("Connected to workspace %s", client.workspace_name)
        return client
    except (AzureError, CredentialUnavailableError, ValueError, OSError) as exc:
        LOG.warning("Azure ML workspace unreachable (%s): %s", type(exc).__name__, exc)
        return None


# --------------------------------------------------------------------------- local executor

_PARENT_INPUT = re.compile(r"^\$\{\{parent\.inputs\.(\w+)\}\}$")
_PARENT_OUTPUT = re.compile(r"^\$\{\{parent\.jobs\.(\w+)\.outputs\.(\w+)\}\}$")
_PLACEHOLDER = re.compile(r"\$\{\{(inputs|outputs)\.(\w+)\}\}")


def serialize(job: PipelineJob, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    job.dump(destination)
    return yaml.safe_load(destination.read_text())


def _binding(value: Any) -> Any:
    return value.get("path", value.get("value")) if isinstance(value, dict) else value


def execution_order(jobs: dict[str, dict[str, Any]]) -> list[str]:
    """Topological order of the pipeline nodes, from their ``${{parent.jobs...}}`` bindings."""
    upstream = {
        name: {
            m.group(1)
            for value in job.get("inputs", {}).values()
            if isinstance(bound := _binding(value), str) and (m := _PARENT_OUTPUT.match(bound))
        }
        for name, job in jobs.items()
    }
    order: list[str] = []
    while upstream:
        ready = sorted(name for name, deps in upstream.items() if deps <= set(order))
        if not ready:
            raise RuntimeError(f"cycle in pipeline graph: {upstream}")
        order += ready
        for name in ready:
            del upstream[name]
    return order


def render_command(template: str, inputs: dict[str, str], outputs: dict[str, Path]) -> str:
    """Substitute ``${{inputs.x}}`` / ``${{outputs.y}}`` like Azure ML does (shell-quoted)."""

    def substitute(match: re.Match[str]) -> str:
        kind, key = match.groups()
        return shlex.quote(inputs[key] if kind == "inputs" else str(outputs[key]))

    return _PLACEHOLDER.sub(substitute, template)


def run_local(
    spec: dict[str, Any], data_inputs: dict[str, Path], run_dir: Path, registry_dir: Path
) -> dict[str, Any]:
    """Execute a serialized Azure ML pipeline job on this machine.

    Component commands are rendered exactly as Azure ML would (``${{inputs.x}}`` /
    ``${{outputs.y}}`` replaced by local paths/values) and run in dependency order
    with the component's code folder as working directory.
    """
    pipeline_inputs = {name: value for name, value in spec.get("inputs", {}).items()}
    pipeline_inputs |= {name: str(path) for name, path in data_inputs.items()}
    jobs: dict[str, dict[str, Any]] = spec["jobs"]

    order = execution_order(jobs)

    env = os.environ | {
        "FRAUD_REGISTRY": "local",
        "FRAUD_LOCAL_REGISTRY_DIR": str(registry_dir),
        "PYTHONUNBUFFERED": "1",
    }
    output_paths: dict[tuple[str, str], Path] = {}
    steps: list[dict[str, Any]] = []
    for name in order:
        job, component = jobs[name], jobs[name]["component"]
        values: dict[str, str] = {}
        for input_name, declaration in component.get("inputs", {}).items():
            bound = _binding(job.get("inputs", {}).get(input_name))
            if bound is None:
                bound = (declaration or {}).get("default")
            if isinstance(bound, str) and (m := _PARENT_INPUT.match(bound)):
                bound = _binding(pipeline_inputs[m.group(1)])
            elif isinstance(bound, str) and (m := _PARENT_OUTPUT.match(bound)):
                bound = str(output_paths[(m.group(1), m.group(2))])
            if bound is None:
                raise ValueError(f"{name}: input {input_name!r} has no value")
            values[input_name] = str(bound)
        for output_name in component.get("outputs", {}):
            path = run_dir / name / output_name
            path.mkdir(parents=True, exist_ok=True)
            output_paths[(name, output_name)] = path

        outputs = {o: output_paths[(name, o)] for o in component.get("outputs", {})}
        rendered = render_command(component["command"], values, outputs)
        argv = shlex.split(rendered)
        if argv[0] == "python":
            argv[0] = sys.executable
        LOG.info("▶ %s: %s", name, rendered)
        started = time.perf_counter()
        log_path = run_dir / name / "step.log"
        with log_path.open("w") as log_file:
            process = subprocess.Popen(
                argv,
                cwd=component.get("code", CODE_DIR),
                env=env | {str(k): str(v) for k, v in job.get("environment_variables", {}).items()},
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            assert process.stdout is not None
            for line in process.stdout:
                log_file.write(line)
                print(f"    [{name}] {line}", end="", flush=True)
            returncode = process.wait()
        steps.append(
            {
                "step": name,
                "status": "Completed" if returncode == 0 else "Failed",
                "seconds": round(time.perf_counter() - started, 2),
                "log": str(log_path),
            }
        )
        if returncode != 0:
            LOG.error("Step %s failed (exit code %d); see %s", name, returncode, log_path)
            break
    completed = len(steps) == len(order) and all(s["status"] == "Completed" for s in steps)
    return {
        "status": "Completed" if completed else "Failed",
        "order": order,
        "steps": steps,
        "outputs": {f"{job}.{out}": str(path) for (job, out), path in output_paths.items()},
    }


# --------------------------------------------------------------------------- CLI


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Azure ML (SDK v2) fraud-detection pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mode", choices=["azure", "validate", "local"], default="azure")
    parser.add_argument(
        "--raw-data",
        default=str(ROOT / "data" / "transactions.csv"),
        help="local CSV (generated if missing), azureml:<data-asset>:<version> or azureml:// URI",
    )
    parser.add_argument("--compute", default=SERVERLESS, help="'serverless' or a cluster name")
    parser.add_argument("--experiment", default="fraud-detection")
    parser.add_argument(
        "--environment", default=None,
        help="registered environment (e.g. azureml:fraud-train@latest) instead of the inline spec",
    )  # fmt: skip
    parser.add_argument("--model-name", default=PipelineParams.model_name)
    parser.add_argument(
        "--quantization", default="auto", choices=["auto", "f32", "f16", "q8_0", "q4_0"]
    )
    parser.add_argument("--subscription-id")
    parser.add_argument("--resource-group")
    parser.add_argument("--workspace")
    parser.add_argument("--config", type=Path, help="Azure ML workspace config.json")
    parser.add_argument("--stream", action="store_true", help="stream the Azure ML run logs")
    parser.add_argument(
        "--strict", action="store_true", help="fail instead of dry-running without Azure access"
    )
    return parser.parse_args(argv)


def _display(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _ensure_raw_data(raw_data: str) -> Input:
    if raw_data.startswith(("azureml:", "https://", "abfss://", "wasbs://")):
        return Input(type=AssetTypes.URI_FILE, path=raw_data)
    path = Path(raw_data).resolve()
    if not path.is_file():
        LOG.info("%s not found: generating the synthetic dataset first", path)
        subprocess.run(
            [sys.executable, str(CODE_DIR / "generate_data.py"), "--output", str(path)],
            check=True,
        )
    return Input(type=AssetTypes.URI_FILE, path=str(path))


def _dry_run(job: PipelineJob, args: argparse.Namespace) -> int:
    client = dry_run_ml_client()
    submitted = submit_pipeline(
        client, job, experiment=args.experiment, compute=args.compute, stream=args.stream
    )
    recorded = [
        f"MLClient.{operations}.{call[0]}"
        for operations in ("compute", "environments", "jobs", "models")
        for call in getattr(client, operations).method_calls
    ]
    LOG.info(
        "Dry run OK: %s would be submitted. SDK calls: %s", submitted.name, ", ".join(recorded)
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    for noisy in ("azure", "azure.ai.ml", "azure.identity", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    args = parse_args(argv)

    raw_data = _ensure_raw_data(args.raw_data)
    params = PipelineParams(model_name=args.model_name, quantization=args.quantization)
    components = build_components(args.environment or build_environment())
    job = build_pipeline(components, raw_data, args.compute, params)

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    definition = OUTPUTS / "pipeline" / "pipeline_job.yml"
    spec = serialize(job, definition)
    LOG.info(
        "Pipeline %s: %s (definition: %s)",
        PIPELINE_NAME, " → ".join(execution_order(spec["jobs"])), _display(definition),
    )  # fmt: skip

    client: MLClient | None = (
        connect(WorkspaceSettings.resolve(args)) if args.mode == "azure" else None
    )
    offline = client is None
    report = validate_pipeline(job, client or offline_ml_client(), offline=offline)
    for path, message in report.deferred.items():
        LOG.warning("Validation of %s deferred to submission: %s", path, message)
    if not report.passed:
        for path, message in report.errors.items():
            LOG.error("Validation error at %s: %s", path, message)
        return 1
    LOG.info("SDK validation passed (%s)", "offline" if offline else "workspace")

    if args.mode == "local":
        run_dir = OUTPUTS / "pipeline_runs" / run_id
        summary = run_local(
            spec,
            data_inputs={"raw_data": Path(raw_data.path)},
            run_dir=run_dir,
            registry_dir=OUTPUTS / "registry",
        )
        registration = run_dir / "register_model" / "registration_output" / "registration.json"
        if registration.is_file():
            summary["registration"] = json.loads(registration.read_text())
        (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        for step in summary["steps"]:
            LOG.info("  %-20s %-9s %6.1fs", step["step"], step["status"], step["seconds"])
        LOG.info("Local run %s: %s (%s)", run_id, summary["status"], _display(run_dir))
        if registration_info := summary.get("registration"):
            keys = ("status", "name", "version", "location", "reason")
            LOG.info(
                "Registration: %s",
                {k: registration_info[k] for k in keys if k in registration_info},
            )
        return 0 if summary["status"] == "Completed" else 1

    if client is None:
        if args.mode == "azure" and args.strict:
            LOG.error("--strict: no Azure ML access, nothing submitted")
            return 2
        if args.mode == "azure":
            LOG.warning("Falling back to a dry run against a strict MLClient autospec")
        return _dry_run(job, args)

    try:
        submit_pipeline(
            client, job, experiment=args.experiment, compute=args.compute, stream=args.stream
        )
    except (AzureError, CredentialUnavailableError) as exc:
        LOG.error("Submission failed (%s): %s", type(exc).__name__, exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
