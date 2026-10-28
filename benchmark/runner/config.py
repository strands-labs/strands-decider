from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Environment:
    path: Path
    name: str
    models: tuple[str, ...]
    endpoint: str
    sagemaker: dict[str, Any] | None
    timeout_s: float
    lifecycle: dict[str, Any]
    pricing: dict[str, Any]
    raw: dict[str, Any]
    fingerprint: str


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def load_environment(value: str) -> Environment:
    candidate = Path(value).expanduser()
    if candidate.is_file():
        path = candidate
    elif candidate.is_dir():
        path = candidate / "environment.yaml"
    else:
        path = Path(__file__).resolve().parents[1] / "environments" / value / "environment.yaml"
    if not path.is_file():
        raise ValueError(f"environment configuration not found: {path}")

    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    name = str(raw.get("name") or path.parent.name)
    configured_models = raw.get("models") or []
    if not isinstance(configured_models, list) or not all(
        isinstance(model, str) and model for model in configured_models
    ):
        raise ValueError(f"{path}: models must be a list of model identifiers")
    endpoint = raw.get("endpoint")
    sagemaker = raw.get("sagemaker")
    timeout_s = float(raw.get("timeout_s", 120))
    lifecycle = raw.get("lifecycle") or {"type": "external"}
    pricing = raw.get("pricing") or {}
    if sagemaker is not None:
        if not isinstance(sagemaker, dict):
            raise ValueError(f"{path}: sagemaker must be a mapping")
        region = sagemaker.get("region")
        if not isinstance(region, str) or not region:
            raise ValueError(f"{path}: sagemaker.region is required")
        instance_type = sagemaker.get("instance_type")
        if not isinstance(instance_type, str) or not instance_type:
            raise ValueError(f"{path}: sagemaker.instance_type is required")
        endpoint_name = sagemaker.get("endpoint_name")
        endpoints = sagemaker.get("endpoints")
        if endpoint_name is not None and (not isinstance(endpoint_name, str) or not endpoint_name):
            raise ValueError(f"{path}: sagemaker.endpoint_name must be a model endpoint name")
        if endpoints is not None and (
            not isinstance(endpoints, dict)
            or not all(
                isinstance(model, str) and model and isinstance(name, str) and name
                for model, name in endpoints.items()
            )
        ):
            raise ValueError(
                f"{path}: sagemaker.endpoints must map model identifiers to endpoint names"
            )
        if endpoint_name and endpoints:
            raise ValueError(
                f"{path}: configure either sagemaker.endpoint_name or sagemaker.endpoints"
            )
        if not endpoint_name and not endpoints:
            raise ValueError(f"{path}: sagemaker.endpoint_name or sagemaker.endpoints is required")
        if endpoints:
            missing = [model for model in configured_models if model not in endpoints]
            if missing:
                raise ValueError(
                    f"{path}: sagemaker.endpoints is missing configured models: "
                    + ", ".join(missing)
                )
        elif len(configured_models) > 1:
            raise ValueError(f"{path}: multiple SageMaker models require sagemaker.endpoints")
        if not isinstance(endpoint, str) or not endpoint:
            endpoint = f"https://runtime.sagemaker.{region}.amazonaws.com"
    elif not isinstance(endpoint, str) or not endpoint:
        raise ValueError(f"{path}: endpoint is required")
    if timeout_s <= 0:
        raise ValueError(f"{path}: timeout_s must be positive")
    if not isinstance(lifecycle, dict):
        raise ValueError(f"{path}: lifecycle must be a mapping")
    if lifecycle.get("type", "external") not in {"external", "managed"}:
        raise ValueError(f"{path}: lifecycle.type must be external or managed")

    execution_config = {key: value for key, value in raw.items() if key != "models"}
    fingerprint = hashlib.sha256(_canonical(execution_config).encode()).hexdigest()
    return Environment(
        path=path.resolve(),
        name=name,
        models=tuple(configured_models),
        endpoint=endpoint,
        sagemaker=sagemaker,
        timeout_s=timeout_s,
        lifecycle=lifecycle,
        pricing=pricing,
        raw=raw,
        fingerprint=fingerprint,
    )


def model_slug(model: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model.strip("/"))
    slug = slug.strip("-") or "model"
    if len(slug) <= 80:
        return slug
    suffix = hashlib.sha256(model.encode()).hexdigest()[:10]
    return f"{slug[:69]}-{suffix}"


def environment_runs_dir(environment: Environment) -> Path:
    return environment.path.parent / "runs"


def accepted_baseline_dir(environment: Environment, model: str) -> Path:
    return environment_runs_dir(environment) / model_slug(model)
