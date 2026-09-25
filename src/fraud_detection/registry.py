"""Model registry abstraction: Azure ML model registry or a local file registry.

Both backends expose the same semantics (auto-incremented integer versions,
string tags), so the evaluate/register steps run unchanged locally and in Azure.
Azure SDK imports are lazy: the local path needs no Azure packages.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from .data_io import read_json, sha256_file, write_json

REGISTRY_ENV = "FRAUD_REGISTRY"  # auto | local | azureml
LOCAL_REGISTRY_ENV = "FRAUD_LOCAL_REGISTRY_DIR"
DEFAULT_LOCAL_REGISTRY = "outputs/registry"


@dataclass(frozen=True)
class RegisteredModel:
    name: str
    version: str
    tags: dict[str, str] = field(default_factory=dict)
    location: str | None = None


class ModelRegistry(Protocol):
    backend: str

    def latest(self, name: str) -> RegisteredModel | None: ...

    def download(self, model: RegisteredModel, destination: Path) -> Path: ...

    def register(
        self, name: str, package_dir: Path, tags: dict[str, str], description: str
    ) -> RegisteredModel: ...


class LocalModelRegistry:
    """``<root>/<name>/<version>/`` folders holding the package and a ``model.json``."""

    backend = "local"

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _versions(self, name: str) -> list[int]:
        folder = self.root / name
        if not folder.is_dir():
            return []
        return sorted(
            int(p.name)
            for p in folder.iterdir()
            if p.name.isdigit() and (p / "model.json").is_file()
        )

    def latest(self, name: str) -> RegisteredModel | None:
        versions = self._versions(name)
        if not versions:
            return None
        folder = self.root / name / str(versions[-1])
        info = read_json(folder / "model.json")
        return RegisteredModel(name, str(versions[-1]), info.get("tags", {}), str(folder))

    def download(self, model: RegisteredModel, destination: Path) -> Path:
        return Path(model.location or self.root / model.name / model.version)

    def register(
        self, name: str, package_dir: Path, tags: dict[str, str], description: str
    ) -> RegisteredModel:
        version = str((self._versions(name) or [0])[-1] + 1)
        target = self.root / name / version
        shutil.copytree(package_dir, target)
        write_json(
            target / "model.json",
            {
                "name": name,
                "version": version,
                "description": description,
                "tags": tags,
                "registered_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "files": {p.name: sha256_file(p) for p in sorted(target.iterdir()) if p.is_file()},
            },
        )
        return RegisteredModel(name, version, tags, str(target))


class AzureMLModelRegistry:
    """Azure ML workspace model registry (``custom_model`` assets)."""

    backend = "azureml"

    def __init__(self, ml_client: Any) -> None:
        self.client = ml_client

    def latest(self, name: str) -> RegisteredModel | None:
        from azure.core.exceptions import ResourceNotFoundError

        try:
            model = self.client.models.get(name=name, label="latest")
        except ResourceNotFoundError:
            return None
        return RegisteredModel(model.name, str(model.version), dict(model.tags or {}), model.path)

    def download(self, model: RegisteredModel, destination: Path) -> Path:
        self.client.models.download(
            name=model.name, version=model.version, download_path=str(destination)
        )
        # Downloads land in <destination>/<name>/<artifact folder>/...
        found = sorted(Path(destination).rglob("model.gguf"))
        if not found:
            raise FileNotFoundError(f"model.gguf not found in {model.name}:{model.version}")
        return found[0].parent

    def register(
        self, name: str, package_dir: Path, tags: dict[str, str], description: str
    ) -> RegisteredModel:
        from azure.ai.ml.constants import AssetTypes
        from azure.ai.ml.entities import Model

        created = self.client.models.create_or_update(
            Model(
                name=name,
                path=str(package_dir),
                type=AssetTypes.CUSTOM_MODEL,
                description=description,
                tags=tags,
            )
        )
        return RegisteredModel(created.name, str(created.version), dict(created.tags or {}))


def running_in_azureml_job() -> bool:
    return "AZUREML_RUN_ID" in os.environ


def ml_client_from_environment() -> Any:
    """MLClient for code running *inside* an Azure ML job (or with AZURE_* variables).

    Credentials, in order: the submitting user's identity (job ``identity:
    user_identity``), the compute's user-assigned managed identity, then the
    ``DefaultAzureCredential`` chain (environment, workload identity, az login...).
    """
    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential, ManagedIdentityCredential

    if os.environ.get("OBO_ENDPOINT"):
        from azure.ai.ml.identity import AzureMLOnBehalfOfCredential

        credential: Any = AzureMLOnBehalfOfCredential()
    elif client_id := os.environ.get("DEFAULT_IDENTITY_CLIENT_ID"):
        credential = ManagedIdentityCredential(client_id=client_id)
    else:
        credential = DefaultAzureCredential()

    def env(*names: str) -> str:
        for name in names:
            if value := os.environ.get(name):
                return value
        raise KeyError(f"none of {names} is set")

    return MLClient(
        credential=credential,
        subscription_id=env("AZUREML_ARM_SUBSCRIPTION", "AZURE_SUBSCRIPTION_ID"),
        resource_group_name=env("AZUREML_ARM_RESOURCEGROUP", "AZURE_RESOURCE_GROUP"),
        workspace_name=env("AZUREML_ARM_WORKSPACE_NAME", "AZUREML_WORKSPACE_NAME"),
    )


def get_registry(kind: str | None = None, local_dir: str | Path | None = None) -> ModelRegistry:
    kind = kind or os.environ.get(REGISTRY_ENV, "auto")
    if kind == "auto":
        kind = "azureml" if running_in_azureml_job() else "local"
    if kind == "azureml":
        return AzureMLModelRegistry(ml_client_from_environment())
    if kind == "local":
        root = local_dir or os.environ.get(LOCAL_REGISTRY_ENV, DEFAULT_LOCAL_REGISTRY)
        return LocalModelRegistry(Path(root))
    raise ValueError(f"unknown registry backend: {kind!r}")
