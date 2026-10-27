"""Serving on AWS Inferentia2 and Trainium through torch-neuronx: the torso as compiled graphs.

neuronx-cc compiles a forward ahead of time for one input shape, so the torso runs as one graph
per sequence bucket: a call is right-padded to the smallest bucket that fits its row, run, and
sliced back. Right padding makes that exact: every layer is causal, so no real position reads a
pad. Full attention is plain causal SDPA with no padding mask; the Gated DeltaNet layers get the
padding mask, as transformers builds it. Their two kernels are replaced by static-shape versions
for Neuron tensors (`neuron_kernels.py`).

Only the torso forward moves, as in `mlx_engine.py`. `NeuronEngine` subclasses
`SystemOneEngine`, so rendering, tokenisation, truncation, option positions, temperatures and the
fp32 head are the torch engine's own code, run on the host CPU. The shared-prefix path is off: it
forwards a KV cache, which a compiled graph does not take, so every question forwards its whole
prompt.

The LoRA adapter is merged at load, in fp32, and the torso is then cast to the checkpoint's dtype
(bf16). A NeuronCore runs one graph at a time, so the engine keeps a copy of the torso on every
visible NeuronCore (`NEURON_RT_VISIBLE_CORES`) and each forward takes whichever is free: an
inf2.xlarge's two cores serve two requests at once from one process.

Compiling every bucket takes about 35 minutes for a 2B torso on inf2.xlarge.
`strands-decider compile-neuron` does it once into a directory, with a manifest of the library
versions; `serve --device neuron --neuron-cache DIR` loads the graphs from it in a few minutes and
refuses a directory compiled by other versions, which would compile everything again.
"""

from __future__ import annotations

import copy
import json
import os
import platform
import queue
import statistics
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from importlib import metadata
from typing import Any

import torch
from torch import nn

from .infer import EngineConfig, SystemOneEngine
from .modeling import StrandsDeciderModel

# Measured on inf2.xlarge, bf16, one row. A graph's time depends on the compiler's choices for its
# shape as well as on its length, so the set comes from measurement (docs/inference.md).
DEFAULT_BUCKETS = (128, 256, 512, 1024, 1536, 2048, 2816, 3584, 4096)
MANIFEST = "manifest.json"
# The compile cache key covers these versions and the paths of the traced Python, so a graph is
# only reused by the same install.
VERSIONED = ("torch", "torch-neuronx", "neuronx-cc", "transformers", "peft", "strands-decider")
# torch-neuronx reads these when it is imported.
_NKI_SDPA = "TORCH_NEURONX_ENABLE_NKI_SDPA"
_NEFF_CACHE = "TORCH_NEURONX_NEFF_CACHE_DIR"

Runner = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
# Builds the runner for NeuronCore `core` from a torso graph of its own.
RunnerFactory = Callable[[nn.Module, int], Runner]


class TorsoGraph(nn.Module):
    """The torso forward a graph is compiled from: (ids [B, L], mask [B, L]) -> hidden [B, L, d]."""

    def __init__(self, torso: nn.Module) -> None:
        super().__init__()
        self.torso = torso

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        linear = attention_mask.to(self.torso.embed_tokens.weight.dtype)
        out = self.torso(
            input_ids=input_ids,
            # Causal SDPA without a padding mask is exact for right-padded rows.
            attention_mask={"full_attention": None, "linear_attention": linear},
            use_cache=False,
            return_dict=True,
        )
        hidden: torch.Tensor = out.last_hidden_state
        return hidden


@dataclass
class _Output:
    last_hidden_state: torch.Tensor


class BucketedTorso(nn.Module):
    """Stands in for `model.torso`: runs each row through the graph of the smallest bucket that
    holds it. Positions past a row's length come back as zeros; nothing reads them.

    `runners` are interchangeable, one per NeuronCore. A forward takes a free one for all its rows
    and waits while every one is busy, so concurrent requests spread over the cores."""

    def __init__(
        self, runners: Sequence[Runner], buckets: Sequence[int], pad_id: int, config: Any
    ) -> None:
        super().__init__()
        if not runners:
            raise ValueError("no runners")
        self.runners = list(runners)
        self._free: queue.SimpleQueue[Runner] = queue.SimpleQueue()
        for runner in self.runners:
            self._free.put(runner)
        self.buckets = sorted(int(b) for b in buckets)
        self.pad_id = pad_id
        self.config = config  # StrandsDeciderModel.hidden_size reads torso.config

    def bucket_for(self, length: int) -> int:
        for bucket in self.buckets:
            if bucket >= length:
                return bucket
        raise ValueError(f"{length} tokens exceeds the largest bucket, {self.buckets[-1]}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Any = None,
        **kwargs: Any,
    ) -> _Output:
        if past_key_values is not None:
            raise RuntimeError("a compiled torso takes no KV cache; serve without the prefix cache")
        rows, width = input_ids.shape
        lengths = [int(n) for n in attention_mask.sum(dim=1).tolist()]
        for length in lengths:
            self.bucket_for(length)  # refuse before taking a core
        hidden: torch.Tensor | None = None
        runner = self._free.get()
        try:
            for row, length in enumerate(lengths):
                bucket = self.bucket_for(length)
                ids = torch.full((1, bucket), self.pad_id, dtype=torch.long)
                mask = torch.zeros((1, bucket), dtype=torch.long)
                ids[0, :length] = input_ids[row, :length].cpu()
                mask[0, :length] = 1
                out = runner(ids, mask)[0, :length].to(torch.float32)
                if hidden is None:
                    hidden = torch.zeros(rows, width, out.shape[-1], dtype=torch.float32)
                hidden[row, :length] = out
        finally:
            self._free.put(runner)
        assert hidden is not None
        return _Output(hidden)


class CpuRunner:
    """The graph run eagerly on the host: what the engine computes, without a Neuron device."""

    def __init__(self, graph: nn.Module, core: int = 0) -> None:
        self.graph = graph
        self.core = core

    def __call__(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            out: torch.Tensor = self.graph(ids, mask)
            return out


class NeuronRunner:
    """The graph on NeuronCore `core` under torch.compile(backend="neuron"): one compiled graph
    per bucket shape. Every core's graphs are the same, so the cores after the first load them
    from the cache.

    Dynamo guards on inference mode, so every call, warm-up included, runs under
    torch.inference_mode(); a call outside it would trace and compile again. torch-neuronx runs
    a compiled graph on the calling thread's current device, not on its inputs' device, so a call
    selects its core first.
    """

    def __init__(self, graph: nn.Module, core: int = 0) -> None:
        import torch._dynamo as dynamo
        import torch_neuronx  # registers the "neuron" device and backend

        # A compiled entry per bucket shape, and each core's graph is another module.
        dynamo.config.cache_size_limit = max(dynamo.config.cache_size_limit, 64 * (core + 1))
        self.core = core
        self.device = f"neuron:{core}"
        self._select = torch_neuronx.set_device
        self._select(core)
        self.graph = graph.to(self.device)
        device = next(self.graph.parameters()).device.type
        if device != "neuron":  # neuron_kernels dispatches on this name
            raise RuntimeError(f"torch-neuronx placed the torso on {device!r}, not 'neuron'")
        self.compiled = torch.compile(self.graph, backend="neuron", fullgraph=True, dynamic=False)

    def __call__(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        self._select(self.core)  # per thread
        with torch.inference_mode():
            out: torch.Tensor = self.compiled(ids.to(self.device), mask.to(self.device))
            return out.to("cpu")


def neuron_cores() -> int:
    """How many NeuronCores this process sees (`NEURON_RT_VISIBLE_CORES`)."""
    import torch_neuronx

    count: int = torch_neuronx.device_count()
    if count < 1:
        raise RuntimeError("torch-neuronx sees no NeuronCore")
    return count


def build_runners(graph: nn.Module, make_runner: RunnerFactory, cores: int) -> list[Runner]:
    """One runner per core, each with its own copy of `graph` (moving a module to a device moves
    it in place). The copies are made one at a time and the last core takes `graph` itself, so the
    host holds at most two torsos while loading."""
    if cores < 1:
        raise ValueError(f"cores must be at least 1, not {cores}")
    return [
        make_runner(graph if core == cores - 1 else copy.deepcopy(graph), core)
        for core in range(cores)
    ]


class NeuronEngine(SystemOneEngine):
    """`SystemOneEngine` with the torso forward on a NeuronCore. Build one with
    `load_neuron_engine`."""

    def __init__(
        self,
        model: StrandsDeciderModel,
        config: EngineConfig,
        runners: Sequence[Runner],
        buckets: Sequence[int] = DEFAULT_BUCKETS,
    ) -> None:
        # Not super().__init__(): that moves the torso to a torch device, and on "cpu" upcasts it.
        # The head and the small index and temperature tensors stay on the host CPU.
        self.cfg = replace(config, device="neuron", use_prefix_cache=False)
        self.model = model.eval()
        self.tok = model.tokenizer
        self.device = "cpu"
        if max(buckets) < model.config.max_length:
            raise ValueError(
                f"the largest bucket, {max(buckets)}, is shorter than the checkpoint's "
                f"max_length, {model.config.max_length}"
            )
        pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        model.torso = BucketedTorso(runners, buckets, pad_id, model.torso.config)
        # No engine lock: SystemOneEngine.evaluate is safe to run concurrently, and the torso
        # hands each forward a free core.
        self._prefix_cache_unforkable = False
        self._tok_lock = threading.Lock()

    @property
    def torso(self) -> BucketedTorso:
        torso: BucketedTorso = self.model.torso
        return torso

    def warm(self, log: Callable[[dict[str, Any]], None] | None = None) -> dict[int, float]:
        """Compile, or load from the cache, every bucket's graph on every core, before serving.
        Returns each bucket's first-call seconds on the first core, where a compile happens; the
        other cores load what it compiled. Inputs are built under inference mode, as a request's
        are."""
        times: dict[int, float] = {}
        for index, runner in enumerate(self.torso.runners):
            for bucket in self.torso.buckets:
                started = time.perf_counter()
                with torch.inference_mode():
                    ids = torch.zeros((1, bucket), dtype=torch.long)
                    mask = torch.ones((1, bucket), dtype=torch.long)
                    runner(ids, mask)
                seconds = round(time.perf_counter() - started, 2)
                times.setdefault(bucket, seconds)
                if log:
                    log({"core": index, "bucket": bucket, "first_call_s": seconds})
        return times

    def time_buckets(self, reps: int = 8) -> dict[int, float]:
        """Median milliseconds per call of each bucket's graph on the first core, once warm."""
        runner = self.torso.runners[0]
        medians: dict[int, float] = {}
        with torch.inference_mode():
            for bucket in self.torso.buckets:
                ids = torch.randint(0, 1000, (1, bucket))
                mask = torch.ones((1, bucket), dtype=torch.long)
                seconds = []
                for _ in range(reps):
                    started = time.perf_counter()
                    runner(ids, mask)
                    seconds.append(time.perf_counter() - started)
                medians[bucket] = round(1000 * statistics.median(seconds), 1)
        return medians


def versions() -> dict[str, str | None]:
    """What a compiled graph depends on: library versions, Python, and this package's path."""
    out: dict[str, str | None] = {
        "python": platform.python_version(),
        "strands_decider_path": os.path.dirname(os.path.abspath(__file__)),
    }
    for package in VERSIONED:
        try:
            out[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            out[package] = None
    return out


def read_manifest(cache_dir: str) -> dict[str, Any]:
    """`cache_dir`'s manifest, refused unless it was compiled by this install."""
    path = os.path.join(cache_dir, MANIFEST)
    with open(path, encoding="utf-8") as fh:
        manifest: dict[str, Any] = json.load(fh)
    want, have = manifest.get("versions", {}), versions()
    differ = {
        k: (want.get(k), have.get(k)) for k in set(want) | set(have) if want.get(k) != have.get(k)
    }
    if differ:
        raise RuntimeError(
            f"{path}: compiled by another install, (compiled, running) = {differ}. Serving it "
            "would compile every bucket again; run `strands-decider compile-neuron` here."
        )
    return manifest


def _set_neuron_env(cache_dir: str | None) -> None:
    """Environment torch-neuronx reads at import: no NKI flash attention, and the graph cache."""
    imported = "torch_neuronx" in sys.modules
    # At L % 2048 == 0 torch-neuronx routes SDPA to an NKI flash-attention kernel, which fails to
    # compile for NeuronCore-v2 (inf2, trn1).
    if os.environ.setdefault(_NKI_SDPA, "0") != "0":
        raise RuntimeError(f"{_NKI_SDPA} must be 0 for this torso")
    wanted = {} if cache_dir is None else {_NEFF_CACHE: os.path.abspath(cache_dir)}
    for name, value in wanted.items():
        if os.environ.get(name) not in (None, value) or (imported and name not in os.environ):
            raise RuntimeError(f"torch-neuronx is already imported; export {name}={value}")
        os.environ[name] = value


def merge_for_neuron(model: StrandsDeciderModel, dtype: torch.dtype) -> StrandsDeciderModel:
    """Fold the LoRA adapter into the base weights in fp32, then cast the torso to `dtype`.
    A compiled graph takes plain linear layers; the head stays fp32."""
    torso = model.torso
    if getattr(torso.config, "model_type", "") not in ("qwen3_5_text", "qwen3_5"):
        raise ValueError(
            f"--device neuron supports Qwen3.5 torsos, not {torso.config.model_type!r}"
        )
    with torch.no_grad():
        torso.to(torch.float32)
        if hasattr(torso, "merge_and_unload"):
            torso = torso.merge_and_unload()
        torso.to(dtype)
    model.torso = torso.eval()
    model.head.to(torch.float32).eval()
    return model


def load_neuron_engine(
    checkpoint: str,
    config: EngineConfig | None = None,
    *,
    cache_dir: str | None = None,
    cores: int | None = None,
    runner: RunnerFactory | None = None,
) -> NeuronEngine:
    """The Neuron counterpart of `infer.load_engine`. `cache_dir` is a directory that
    `compile_neuron` wrote: the graphs load from it, with its buckets. Without one, every bucket
    compiles on first use. `cores` is how many NeuronCores to serve from (default: every visible
    one). `runner` builds a core's graph runner (default `NeuronRunner`; `CpuRunner` runs the
    same graph on the host, with one core unless `cores` says otherwise)."""
    buckets: Sequence[int] = DEFAULT_BUCKETS
    if cache_dir is not None:
        buckets = read_manifest(cache_dir)["buckets"]
        cache_dir = writable_cache(cache_dir)
    return _load(checkpoint, config, cache_dir, buckets, runner, cores)


def writable_cache(cache_dir: str) -> str:
    """`cache_dir`, or a copy of it in a temporary directory if it is read-only (SageMaker's copy
    of the model data is). The runtime writes to the cache as it loads graphs; its path is not
    part of the cache key, so a copy loads the same graphs."""
    if os.access(cache_dir, os.W_OK):
        return cache_dir
    import shutil
    import tempfile

    copy = os.path.join(tempfile.mkdtemp(prefix="neuron-cache-"), "cache")
    shutil.copytree(cache_dir, copy)
    return copy


def _load(
    checkpoint: str,
    config: EngineConfig | None,
    cache_dir: str | None,
    buckets: Sequence[int],
    runner: RunnerFactory | None,
    cores: int | None = None,
) -> NeuronEngine:
    if runner is None:
        _set_neuron_env(cache_dir)
        cores = cores or neuron_cores()
    model = StrandsDeciderModel.load(checkpoint, attn_implementation="sdpa")
    model = merge_for_neuron(model, getattr(torch, model.config.torch_dtype))
    from .neuron_kernels import install

    install()
    graph = TorsoGraph(model.torso).eval()
    runners = build_runners(graph, runner or NeuronRunner, cores or 1)
    return NeuronEngine(model, config or EngineConfig(), runners, buckets)


def compile_neuron(
    checkpoint: str,
    cache_dir: str,
    *,
    buckets: Sequence[int] = DEFAULT_BUCKETS,
    reps: int = 8,
    log: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Compile every bucket's graph into `cache_dir`, time each, and write the manifest that
    `load_neuron_engine` checks. Returns the manifest."""
    os.makedirs(cache_dir, exist_ok=True)
    started = time.perf_counter()
    engine = _load(checkpoint, None, cache_dir, buckets, None, cores=1)  # compiles once
    manifest: dict[str, Any] = {
        "versions": versions(),
        "checkpoint": checkpoint,
        "dtype": engine.model.config.torch_dtype,
        "buckets": list(engine.torso.buckets),
        "load_s": round(time.perf_counter() - started, 1),
    }
    manifest["first_call_s"] = engine.warm(log)
    manifest["exec_ms_median"] = engine.time_buckets(reps)
    with open(os.path.join(cache_dir, MANIFEST), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    return manifest
