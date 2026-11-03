from __future__ import annotations

import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import boto3
import numpy as np
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError, ConnectTimeoutError, ReadTimeoutError
from transformers import AutoTokenizer

from benchmark.runner.config import Environment
from benchmark.runner.workload import Request, Shape, Workload


def percentile(values: list[float], q: float) -> float | None:
    return float(np.quantile(values, q, method="inverted_cdf")) if values else None


def median(values: list[float | int | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return float(np.median(present)) if present else None


def round_or_none(value: float | None, digits: int = 3) -> float | None:
    return round(value, digits) if value is not None else None


@dataclass
class Invocation:
    status: int
    latency_ms: float
    error: str | None = None
    timeout: bool = False
    server_ms: float | None = None
    input_tokens: int | None = None
    answer_count: int = 0

    @property
    def success(self) -> bool:
        return self.status == 200 and self.error is None

    @property
    def rejected(self) -> bool:
        return 400 <= self.status < 500


def parse_invocation(status: int, body: bytes, latency_ms: float) -> Invocation:
    try:
        parsed = json.loads(body)
    except ValueError:
        parsed = None
    if status != 200:
        detail = (
            parsed.get("detail", parsed)
            if isinstance(parsed, dict)
            else body.decode(errors="replace")
        )
        return Invocation(
            status=status,
            latency_ms=latency_ms,
            error=str(detail)[:500],
        )
    if not isinstance(parsed, dict):
        return Invocation(
            status=status,
            latency_ms=latency_ms,
            error="response was not a JSON object",
        )
    usage = parsed.get("usage") or {}
    answers = parsed.get("answers") or {}
    return Invocation(
        status=status,
        latency_ms=latency_ms,
        server_ms=parsed.get("latency_ms"),
        input_tokens=usage.get("input_tokens"),
        answer_count=len(answers),
    )


class HTTPClient:
    """Small, retry-free HTTP client for benchmark requests."""

    def __init__(self, base_url: str, timeout_s: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        parsed = urllib.parse.urlsplit(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"invalid HTTP base URL: {base_url}")

    def invoke(self, payload: bytes) -> Invocation:
        started = time.perf_counter()
        request = urllib.request.Request(
            f"{self.base_url}/v1/systemone",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                status = response.status
                body = response.read()
        except urllib.error.HTTPError as exc:
            status = exc.code
            body = exc.read()
        except (TimeoutError, urllib.error.URLError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            return Invocation(
                status=0,
                latency_ms=(time.perf_counter() - started) * 1000,
                error=f"{type(exc).__name__}: {exc}"[:500],
                timeout=isinstance(reason, TimeoutError),
            )

        return parse_invocation(status, body, (time.perf_counter() - started) * 1000)

    def health(self) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(
                f"{self.base_url}/health",
                timeout=min(self.timeout_s, 10),
            ) as response:
                parsed = json.loads(response.read())
                return parsed if isinstance(parsed, dict) else {}
        except (OSError, ValueError, urllib.error.URLError):
            return {}


class SageMakerClient:
    """Small, retry-free SageMaker Runtime client using the AWS credential chain."""

    def __init__(
        self,
        endpoint_name: str,
        region: str,
        model: str,
        *,
        timeout_s: float,
        profile: str | None,
        max_pool_connections: int,
    ):
        session = boto3.Session(profile_name=profile, region_name=region)
        control = session.client("sagemaker")
        endpoint = control.describe_endpoint(EndpointName=endpoint_name)
        endpoint_config = control.describe_endpoint_config(
            EndpointConfigName=endpoint["EndpointConfigName"]
        )
        live_variants = {
            variant["VariantName"]: variant for variant in endpoint.get("ProductionVariants", [])
        }
        variants = [
            {
                "name": variant["VariantName"],
                "model_name": variant["ModelName"],
                "instance_type": variant.get("InstanceType"),
                "instance_count": live_variants.get(variant["VariantName"], {}).get(
                    "CurrentInstanceCount", variant.get("InitialInstanceCount")
                ),
                "desired_instance_count": live_variants.get(variant["VariantName"], {}).get(
                    "DesiredInstanceCount", variant.get("InitialInstanceCount")
                ),
            }
            for variant in endpoint_config.get("ProductionVariants", [])
        ]
        self.client = session.client(
            "sagemaker-runtime",
            config=Config(
                connect_timeout=timeout_s,
                read_timeout=timeout_s,
                max_pool_connections=max_pool_connections,
                retries={"max_attempts": 0},
            ),
        )
        self.endpoint_name = endpoint_name
        self.model = model
        self.region = region
        self.metadata = {
            "endpoint_name": endpoint_name,
            "endpoint_arn": endpoint.get("EndpointArn"),
            "endpoint_config_name": endpoint["EndpointConfigName"],
            "instance_count": sum(int(variant.get("instance_count") or 0) for variant in variants),
            "variants": variants,
        }

    def invoke(self, payload: bytes) -> Invocation:
        started = time.perf_counter()
        try:
            response = self.client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType="application/json",
                Body=payload,
            )
            status = int(response["ResponseMetadata"]["HTTPStatusCode"])
            body = response["Body"].read()
        except (BotoCoreError, ClientError) as exc:
            response = getattr(exc, "response", {}) or {}
            metadata = response.get("ResponseMetadata") or {}
            status = int(response.get("OriginalStatusCode") or metadata.get("HTTPStatusCode") or 0)
            return Invocation(
                status=status,
                latency_ms=(time.perf_counter() - started) * 1000,
                error=f"{type(exc).__name__}: {exc}"[:500],
                timeout=isinstance(exc, (ConnectTimeoutError, ReadTimeoutError)),
            )
        return parse_invocation(status, body, (time.perf_counter() - started) * 1000)

    def health(self) -> dict[str, Any]:
        return {
            "checkpoint": self.model,
            "backend": "sagemaker",
            "region": self.region,
            **self.metadata,
        }


@dataclass
class Sample:
    request: Request
    invocation: Invocation

    @property
    def success(self) -> bool:
        return (
            self.invocation.success and self.invocation.answer_count == self.request.shape.questions
        )

    @property
    def decisions(self) -> int:
        return self.request.shape.questions if self.success else 0


class ManagedServer:
    def __init__(self, environment: Environment, model: str):
        self.environment = environment
        self.model = model
        self.process: subprocess.Popen[bytes] | None = None
        self.log_path: Path | None = None
        self.log_file: Any = None
        self.ready_seconds: float | None = None
        self.base_url: str | None = None

    @staticmethod
    def _free_port() -> int:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def start(self) -> str | None:
        lifecycle = self.environment.lifecycle
        if lifecycle.get("type", "external") != "managed":
            return self.environment.endpoint
        command = lifecycle.get("command")
        if not isinstance(command, list) or not command:
            raise ValueError("managed lifecycle requires a command list")
        port = self._free_port()
        values = {"model": self.model, "port": str(port)}
        rendered = [str(part).format(**values) for part in command]
        self.base_url = self.environment.endpoint.format(**values)
        self.log_path = Path(os.environ.get("TMPDIR", "/tmp")) / (
            f"strands-decider-benchmark-{os.getpid()}-{port}.log"
        )
        self.log_file = self.log_path.open("wb")
        started = time.monotonic()
        self.process = subprocess.Popen(
            rendered,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        timeout_s = float(lifecycle.get("health_timeout_s", 1800))
        health = HTTPClient(
            self.base_url,
            timeout_s=min(self.environment.timeout_s, 10),
        )
        while time.monotonic() - started < timeout_s:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"managed server exited with {self.process.returncode}; log: {self.log_path}"
                )
            if health.health():
                self.ready_seconds = time.monotonic() - started
                return self.base_url
            time.sleep(1)
        raise TimeoutError(f"server did not become ready within {timeout_s}s; log: {self.log_path}")

    def alive(self) -> bool:
        return self.process is None or self.process.poll() is None

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=10)
        if self.log_file is not None:
            self.log_file.close()


def summarize_samples(
    samples: list[Sample],
    elapsed_s: float,
) -> dict[str, Any]:
    successful = [sample for sample in samples if sample.success]
    latencies = [sample.invocation.latency_ms for sample in successful]
    server = [
        float(sample.invocation.server_ms)
        for sample in successful
        if sample.invocation.server_ms is not None
    ]
    decisions = sum(sample.decisions for sample in successful)
    failures = len(samples) - len(successful)
    ooms = sum(
        "out of memory" in (sample.invocation.error or "").lower()
        or "oom" in (sample.invocation.error or "").lower().split()
        for sample in samples
        if not sample.success
    )
    errors: dict[str, int] = {}
    for sample in samples:
        if sample.success:
            continue
        if (
            sample.invocation.success
            and sample.invocation.answer_count != sample.request.shape.questions
        ):
            key = "incomplete answer set"
        else:
            key = sample.invocation.error or f"status {sample.invocation.status}"
        errors[key] = errors.get(key, 0) + 1

    by_shape: dict[str, Any] = {}
    for shape in sorted({sample.request.shape for sample in samples}, key=lambda item: item.key):
        selected = [sample for sample in samples if sample.request.shape == shape]
        selected_ok = [sample for sample in selected if sample.success]
        shape_latencies = [sample.invocation.latency_ms for sample in selected_ok]
        reported_tokens = [
            sample.invocation.input_tokens
            for sample in selected_ok
            if sample.invocation.input_tokens is not None
        ]
        shape_failures = len(selected) - len(selected_ok)
        shape_decisions = sum(sample.decisions for sample in selected_ok)
        by_shape[shape.key] = {
            "requested_input_tokens": shape.input_tokens,
            "input_tokens": (
                round(float(np.median(reported_tokens))) if reported_tokens else shape.input_tokens
            ),
            "questions": shape.questions,
            "requests": len(selected),
            "successes": len(selected_ok),
            "failures": shape_failures,
            "failure_rate": round(shape_failures / len(selected), 6) if selected else 1.0,
            "decisions": shape_decisions,
            "decisions_per_second": round(shape_decisions / elapsed_s, 4) if elapsed_s else 0,
            "p50_ms": round_or_none(percentile(shape_latencies, 0.50), 2),
            "p95_ms": round_or_none(percentile(shape_latencies, 0.95), 2),
            "p99_ms": round_or_none(percentile(shape_latencies, 0.99), 2),
        }

    return {
        "elapsed_seconds": round(elapsed_s, 3),
        "requests": len(samples),
        "successes": len(successful),
        "failures": failures,
        "failure_rate": round(failures / len(samples), 6) if samples else 1.0,
        "timeouts": sum(sample.invocation.timeout for sample in samples),
        "rejections": sum(sample.invocation.rejected for sample in samples),
        "ooms": ooms,
        "decisions": decisions,
        "requests_per_second": round(len(successful) / elapsed_s, 4) if elapsed_s else 0,
        "decisions_per_second": round(decisions / elapsed_s, 4) if elapsed_s else 0,
        "latency": {
            "min_ms": round_or_none(min(latencies) if latencies else None, 2),
            "mean_ms": round_or_none(float(np.mean(latencies)) if latencies else None, 2),
            "p50_ms": round_or_none(percentile(latencies, 0.50), 2),
            "p95_ms": round_or_none(percentile(latencies, 0.95), 2),
            "p99_ms": round_or_none(percentile(latencies, 0.99), 2),
            "max_ms": round_or_none(max(latencies) if latencies else None, 2),
        },
        "server_latency": {
            "mean_ms": round_or_none(float(np.mean(server)) if server else None, 2),
            "p50_ms": round_or_none(percentile(server, 0.50), 2),
            "p99_ms": round_or_none(percentile(server, 0.99), 2),
        },
        "repeated_state_fraction": round(
            sum(sample.request.repeated_state for sample in samples) / len(samples), 4
        )
        if samples
        else None,
        "errors": dict(sorted(errors.items(), key=lambda item: (-item[1], item[0]))[:20]),
        "shapes": by_shape,
    }


def combine_repetitions(
    mode: str,
    load: float,
    repetitions: list[dict[str, Any]],
) -> dict[str, Any]:
    latency = {
        name: round_or_none(median([rep["latency"].get(name) for rep in repetitions]), 2)
        for name in ("min_ms", "mean_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms")
    }
    point = {
        "mode": mode,
        "load": load,
        "repetitions": repetitions,
        "requests_per_second": round_or_none(
            median([rep["requests_per_second"] for rep in repetitions]), 4
        ),
        "decisions_per_second": round_or_none(
            median([rep["decisions_per_second"] for rep in repetitions]), 4
        ),
        "failure_rate": round_or_none(median([rep["failure_rate"] for rep in repetitions]), 6),
        "failure_rate_max": round_or_none(
            max((rep["failure_rate"] for rep in repetitions), default=1.0), 6
        ),
        "ooms": sum(rep["ooms"] for rep in repetitions),
        "latency": latency,
        "server_latency": {
            name: round_or_none(median([rep["server_latency"].get(name) for rep in repetitions]), 2)
            for name in ("mean_ms", "p50_ms", "p99_ms")
        },
    }
    return point


class BenchmarkRunner:
    def __init__(
        self,
        environment: Environment,
        model: str,
        *,
        requests_per_point: int,
        repetitions: int,
        concurrency_levels: list[int],
        seed: int,
        progress: Callable[[str], None] = print,
        advance: Callable[[int], None] | None = None,
    ):
        self.environment = environment
        self.model = model
        self.requests_per_point = requests_per_point
        self.repetitions = repetitions
        self.concurrency_levels = concurrency_levels
        if not concurrency_levels or min(concurrency_levels) <= 0:
            raise ValueError("concurrency levels must be positive")
        if requests_per_point % 8:
            raise ValueError("requests per point must be divisible by the 8-request workload cycle")
        self.seed = seed
        self.progress = progress
        self.advance = advance or (lambda _: None)
        self.server = ManagedServer(environment, model)
        self.client: HTTPClient | SageMakerClient | None = None
        self.workload: Workload | None = None
        self._point_index = 0

    def _invoke(self, request: Request) -> Sample:
        assert self.client is not None
        invocation = self.client.invoke(request.body)
        return Sample(request=request, invocation=invocation)

    def _measure_input_tokens(self, request: Request) -> int | None:
        assert self.client is not None
        invocation = self.client.invoke(request.body)
        if invocation.status in {413, 422}:
            return None
        if not invocation.success:
            raise RuntimeError(
                "input-length calibration failed: "
                f"status={invocation.status}, error={invocation.error}"
            )
        if invocation.input_tokens is None:
            raise RuntimeError("server did not report usage.input_tokens during calibration")
        return int(invocation.input_tokens)

    def _closed_repetition(self, concurrency: int, offset: int) -> dict[str, Any]:
        assert self.workload is not None
        warmups = self.workload.throughput_requests(concurrency, sequence=offset + 900_000_000)
        request_count = max(self.requests_per_point, 2 * concurrency)
        requests = self.workload.throughput_requests(request_count, sequence=offset)
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            warmup_samples = []
            for sample in pool.map(self._invoke, warmups):
                warmup_samples.append(sample)
                self.advance(1)
            started = time.monotonic()
            samples = []
            for sample in pool.map(self._invoke, requests):
                samples.append(sample)
                self.advance(1)
        elapsed = time.monotonic() - started
        summary = summarize_samples(samples, elapsed)
        summary["warmup_requests"] = len(warmup_samples)
        summary["warmup_failures"] = sum(not sample.success for sample in warmup_samples)
        return summary

    def _latency_repetition(self, offset: int) -> dict[str, Any]:
        assert self.workload is not None
        warmups = self.workload.latency_warmups(sequence=offset + 900_000_000)
        requests = self.workload.requests(self.requests_per_point, sequence=offset)
        with ThreadPoolExecutor(max_workers=1) as pool:
            warmup_samples = []
            for sample in pool.map(self._invoke, warmups):
                warmup_samples.append(sample)
                self.advance(1)
            started = time.monotonic()
            samples = []
            for sample in pool.map(self._invoke, requests):
                samples.append(sample)
                self.advance(1)
        elapsed = time.monotonic() - started
        summary = summarize_samples(samples, elapsed)
        summary["warmup_requests"] = len(warmup_samples)
        summary["warmup_failures"] = sum(not sample.success for sample in warmup_samples)
        return summary

    def _latency_sweep(self) -> dict[str, Any]:
        repetitions = []
        for repetition in range(self.repetitions):
            offset = 500_000_000 + repetition * 1_000_000
            repetitions.append(self._latency_repetition(offset))
        point = combine_repetitions("input-length", 1, repetitions)
        self.progress(
            f"serial latency: p50 {point['latency']['p50_ms']} ms, "
            f"failures {100 * (point['failure_rate'] or 0):.2f}%"
        )
        return point

    def _run_point(self, concurrency: int) -> dict[str, Any]:
        reps = []
        self._point_index += 1
        for repetition in range(self.repetitions):
            # These multiples ensure nominally unique states do not become
            # cross-point cache hits.
            offset = self._point_index * 10_000_000 + repetition * 1_000_000
            reps.append(self._closed_repetition(concurrency, offset))
        point = combine_repetitions("concurrency", concurrency, reps)
        self.progress(
            f"concurrency {concurrency}: {point['requests_per_second']:.3f} requests/s, "
            f"p50 {point['latency']['p50_ms']} ms, failures {100 * (point['failure_rate'] or 0):.2f}%"
        )
        return point

    def _closed_sweep(self) -> list[dict[str, Any]]:
        return [self._run_point(concurrency) for concurrency in self.concurrency_levels]

    def _pricing(self, point: dict[str, Any]) -> dict[str, Any]:
        pricing = self.environment.pricing
        dps = point.get("decisions_per_second")
        if not pricing or not dps:
            return {"cost_per_1m_decisions_usd": None, "reason": "pricing unavailable"}
        if pricing.get("hourly_usd") is not None:
            cost = float(pricing["hourly_usd"]) / (3600 * dps) * 1_000_000
            return {
                "cost_per_1m_decisions_usd": round(cost, 4),
                "method": "hourly",
                "hourly_usd": float(pricing["hourly_usd"]),
            }
        request_usd = float(pricing.get("request_usd", 0))
        gb_second_usd = pricing.get("gb_second_usd")
        memory_gb = pricing.get("memory_gb")
        server_mean_ms = point["server_latency"].get("mean_ms")
        duration_ms = server_mean_ms or point["latency"].get("mean_ms")
        if request_usd or (
            gb_second_usd is not None and memory_gb is not None and duration_ms is not None
        ):
            requests_per_second = point.get("requests_per_second") or 0
            decisions_per_request = dps / requests_per_second if requests_per_second else 0
            if not decisions_per_request:
                return {
                    "cost_per_1m_decisions_usd": None,
                    "reason": "no successful decisions",
                }
            per_request = request_usd
            if gb_second_usd is not None and memory_gb is not None and duration_ms is not None:
                per_request += float(gb_second_usd) * float(memory_gb) * float(duration_ms) / 1000
            return {
                "cost_per_1m_decisions_usd": round(
                    per_request / decisions_per_request * 1_000_000, 4
                ),
                "method": "per-request",
                "duration_source": (
                    "server-reported" if server_mean_ms is not None else "client-round-trip"
                ),
            }
        return {
            "cost_per_1m_decisions_usd": None,
            "reason": "unsupported pricing configuration",
        }

    def run(self) -> tuple[dict[str, Any], dict[str, Any]]:
        base_url = self.server.start()
        if not base_url:
            raise ValueError("HTTP environment requires endpoint")
        if self.environment.sagemaker:
            endpoint_name = self.environment.sagemaker.get("endpoint_name")
            endpoints = self.environment.sagemaker.get("endpoints") or {}
            endpoint_name = endpoint_name or endpoints.get(self.model)
            if not endpoint_name:
                raise ValueError(f"no SageMaker endpoint configured for model {self.model}")
            self.client = SageMakerClient(
                endpoint_name,
                self.environment.sagemaker["region"],
                self.model,
                timeout_s=self.environment.timeout_s,
                profile=self.environment.sagemaker.get("profile"),
                max_pool_connections=max(self.concurrency_levels),
            )
        else:
            self.client = HTTPClient(base_url, timeout_s=self.environment.timeout_s)
        try:
            health = self.client.health()
            if self.environment.sagemaker:
                expected_instance_type = self.environment.sagemaker["instance_type"]
                actual_instance_types = {
                    variant.get("instance_type")
                    for variant in health.get("variants", [])
                    if variant.get("instance_type")
                }
                if actual_instance_types != {expected_instance_type}:
                    raise ValueError(
                        f"SageMaker endpoint for {self.model} uses "
                        f"{sorted(actual_instance_types) or ['unknown']}, expected "
                        f"{expected_instance_type}"
                    )
            tokenizer_id = health.get("checkpoint") or self.model
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
            self.workload = Workload(tokenizer, self.seed)
            self.progress("calibrating input lengths against server-reported tokens")
            self.workload.calibrate(self._measure_input_tokens)

            first_shape = Shape(input_tokens=256)
            first = self._invoke(self.workload.request(first_shape, repeated=True, sequence=0))
            self.advance(1)
            first_request = {
                "latency_ms": round(first.invocation.latency_ms, 2),
                "status": first.invocation.status,
                "success": first.success,
                "input_tokens_reported": first.invocation.input_tokens,
                "requested_input_tokens": first_shape.input_tokens,
                "questions": first_shape.questions,
            }
            if not first.success:
                raise RuntimeError(
                    "first benchmark request failed: "
                    f"status={first.invocation.status}, error={first.invocation.error}"
                )

            latency_sweep = self._latency_sweep()
            closed = self._closed_sweep()
            saturation = max(
                closed,
                key=lambda point: point.get("requests_per_second") or 0,
            )
            if not saturation.get("requests_per_second"):
                raise RuntimeError("closed-loop sweep produced no successful requests")
            measured_points = [latency_sweep, *closed]
            failure_rate = max(float(point["failure_rate_max"] or 0) for point in measured_points)
            ooms = sum(int(point["ooms"]) for point in measured_points)
            headline = {
                "max_requests_per_second": saturation["requests_per_second"],
                "reference_concurrency": 1,
                "reference_latency": latency_sweep["latency"],
                "failure_rate": round(failure_rate, 6),
                "zero_oom_or_crash": ooms == 0 and self.server.alive(),
                **self._pricing(saturation),
            }
            summary = {
                "headline": headline,
                "lifecycle": {
                    "time_to_ready_seconds": round_or_none(self.server.ready_seconds, 3),
                    "first_request": first_request,
                },
                "peak": {
                    "concurrency": saturation["load"],
                    "requests_per_second": saturation["requests_per_second"],
                },
                "latency_sweep": latency_sweep,
                "closed_loop": closed,
                "health": health,
            }
            return summary, self.workload.manifest()
        finally:
            self.server.stop()


def machine_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    if shutil.which("nvidia-smi"):
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        metadata["gpus"] = result.stdout.strip().splitlines()
    return metadata


def git_metadata(root: Path) -> dict[str, Any]:
    def command(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(root), *args],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    revision = command("rev-parse", "HEAD")
    branch = command("branch", "--show-current")
    status = command("status", "--porcelain")

    return {
        "revision": revision or None,
        "branch": branch or None,
        "dirty": bool(status) if status is not None else None,
    }
