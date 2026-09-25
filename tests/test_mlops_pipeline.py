"""Azure ML SDK v2 pipeline: definition, offline validation, dry run and local execution."""

# No ``from __future__ import annotations``: ``@dsl.pipeline`` reads annotations at runtime.
import json
from pathlib import Path

import pytest
import yaml
from azure.ai.ml import Input, Output, command, dsl
from azure.ai.ml.constants import AssetTypes

import mlops_pipeline as mp

STEPS = ["prep_data", "train_and_optimize", "evaluate", "register_model"]


@pytest.fixture(scope="module")
def pipeline_job(dataset_csv: Path):
    components = mp.build_components(mp.build_environment())
    raw = Input(type=AssetTypes.URI_FILE, path=str(dataset_csv))
    return mp.build_pipeline(components, raw, mp.SERVERLESS, mp.PipelineParams())


@pytest.fixture(scope="module")
def spec(pipeline_job, tmp_path_factory) -> dict:
    return mp.serialize(pipeline_job, tmp_path_factory.mktemp("pipeline") / "pipeline_job.yml")


def test_pipeline_graph_and_bindings(spec: dict) -> None:
    jobs = spec["jobs"]
    assert mp.execution_order(jobs) == STEPS
    assert jobs["train_and_optimize"]["inputs"]["train_data"]["path"] == (
        "${{parent.jobs.prep_data.outputs.train_data}}"
    )
    assert (
        jobs["evaluate"]["inputs"]["test_data"]["path"]
        == "${{parent.jobs.prep_data.outputs.test_data}}"
    )
    assert jobs["register_model"]["inputs"]["evaluation_input"]["path"] == (
        "${{parent.jobs.evaluate.outputs.evaluation_output}}"
    )
    for step in ("evaluate", "register_model"):  # registry access: no cache reuse, user identity
        assert jobs[step]["component"]["is_deterministic"] is False
        assert jobs[step]["identity"]["type"] == "user_identity"
    assert spec["compute"].endswith(mp.SERVERLESS)


def test_environment_is_pinned(spec: dict) -> None:
    environment = spec["jobs"]["train_and_optimize"]["component"]["environment"]
    assert environment["image"] == mp.AML_BASE_IMAGE and ":latest" not in environment["image"]
    pip = next(d["pip"] for d in environment["conda_file"]["dependencies"] if isinstance(d, dict))
    assert all("==" in requirement for requirement in pip)
    assert any(r.startswith("azure-ai-ml==") for r in pip) and any(
        r.startswith("azureml-mlflow==") for r in pip
    )


def test_offline_validation_with_the_public_sdk_api(pipeline_job) -> None:
    report = mp.validate_pipeline(pipeline_job, mp.offline_ml_client(), offline=True)
    assert report.passed and not report.errors and not report.deferred


def test_named_compute_validation_is_deferred_offline(dataset_csv: Path) -> None:
    raw = Input(type=AssetTypes.URI_FILE, path=str(dataset_csv))
    job = mp.build_pipeline(
        mp.build_components("azureml:fraud-env@latest"), raw, "cpu-cluster", mp.PipelineParams()
    )
    report = mp.validate_pipeline(job, mp.offline_ml_client(), offline=True)
    assert report.passed and report.deferred


def test_component_binding_errors_are_caught(dataset_csv: Path) -> None:
    components = mp.build_components(mp.build_environment())
    components["train_and_optimize"] = command(
        name="broken_train",
        code=str(mp.CODE_DIR),
        environment=mp.build_environment(),
        command="python train.py --x ${{inputs.not_declared}} --o ${{outputs.model_output}}",
        inputs={
            name: Input(type="string")
            for name in ("train_data", "model_name", "valid_fraction", "quantization")
        },
        outputs={"model_output": Output(type=AssetTypes.URI_FOLDER)},
    )
    raw = Input(type=AssetTypes.URI_FILE, path=str(dataset_csv))
    job = mp.build_pipeline(components, raw, mp.SERVERLESS, mp.PipelineParams())
    report = mp.validate_pipeline(job, mp.offline_ml_client(), offline=True)
    assert not report.passed
    assert any("not_declared" in message for message in report.errors.values())


def test_graph_errors_are_caught(dataset_csv: Path) -> None:
    components = mp.build_components(mp.build_environment())

    @dsl.pipeline(compute=mp.SERVERLESS)
    def incomplete(raw_data: Input):
        prep = components["prep_data"](raw_data=raw_data)
        components["evaluate"](
            model_input=prep.outputs.train_data, test_data=prep.outputs.test_data
        )

    job = incomplete(raw_data=Input(type=AssetTypes.URI_FILE, path=str(dataset_csv)))
    report = mp.validate_pipeline(job, mp.offline_ml_client(), offline=True)
    assert not report.passed and any("model_name" in path for path in report.errors)


def test_dry_run_client_enforces_real_sdk_signatures(pipeline_job) -> None:
    client = mp.dry_run_ml_client()
    submitted = mp.submit_pipeline(
        client, pipeline_job, experiment="exp", compute="cpu-cluster", stream=True
    )
    assert submitted.name.startswith("dryrun-")
    client.compute.get.assert_called_once_with("cpu-cluster")
    client.jobs.create_or_update.assert_called_once_with(pipeline_job, experiment_name="exp")
    client.jobs.stream.assert_called_once_with(submitted.name)
    with pytest.raises(TypeError):
        client.jobs.create_or_update(pipeline_job, experiment="typo")  # not a real parameter
    with pytest.raises(AttributeError):
        client.jobz  # noqa: B018


def test_requirements_are_flattened() -> None:
    requirements = mp.read_requirements(mp.ROOT / "requirements" / "azureml.txt")
    assert "scikit-learn==1.9.1" in requirements and not any(
        r.startswith("-r") for r in requirements
    )


def test_main_without_azure_access(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, dataset_csv: Path
) -> None:
    for variable in ("AZURE_SUBSCRIPTION_ID", "AZURE_RESOURCE_GROUP", "AZUREML_WORKSPACE_NAME"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(mp, "OUTPUTS", tmp_path)
    args = ["--raw-data", str(dataset_csv)]
    assert mp.main(args) == 0  # degrades to a dry run
    assert mp.main([*args, "--strict"]) == 2
    assert mp.main([*args, "--mode", "validate"]) == 0
    assert (
        yaml.safe_load((tmp_path / "pipeline" / "pipeline_job.yml").read_text())["type"]
        == "pipeline"
    )


def test_unreachable_workspace_is_handled(monkeypatch: pytest.MonkeyPatch) -> None:
    for variable in ("AZURE_CLIENT_ID", "AZURE_TENANT_ID", "AZURE_CLIENT_SECRET"):
        monkeypatch.delenv(variable, raising=False)
    settings = mp.WorkspaceSettings("00000000-0000-0000-0000-000000000000", "rg", "ws")
    assert mp.connect(settings) is None


def test_local_execution_of_the_pipeline_definition(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, dataset_csv: Path
) -> None:
    monkeypatch.setattr(mp, "OUTPUTS", tmp_path)
    assert mp.main(["--mode", "local", "--raw-data", str(dataset_csv)]) == 0
    (run_dir,) = (tmp_path / "pipeline_runs").iterdir()
    summary = json.loads((run_dir / "run_summary.json").read_text())
    assert summary["status"] == "Completed" and summary["order"] == STEPS
    assert summary["registration"]["status"] == "registered"
    assert (tmp_path / "registry" / "fraud-detection-gbdt" / "1" / "model.gguf").is_file()
