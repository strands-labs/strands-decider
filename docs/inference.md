# Inference server

Strands Decider ships with a small HTTP server so your applications can keep
one model loaded and ask it many questions over the network. The server runs
the same engine as the CLI, so the answers match `strands-decider ask`.

## Install and serve

```bash
pip install strands-decider
strands-decider serve StrandsAgents/strands-decider-E2B-gemma4-v1-2610 --port 8000
```

On first start the engine downloads the checkpoint and warms every request
shape so later requests do not pay a one-time compile cost.

## HTTP API

The server exposes three endpoints:

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/v1/systemone` | Ask one or more questions about a shared state. |
| `GET` | `/health` | Liveness, plus device and vision flags. |
| `GET` | `/ping` | Readiness without touching the engine. |

A request picks a question type for each question. The server encodes the
shared state once and evaluates every question against that encoding:

```bash
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Refund request from customer in EU for a damaged item.",
  "questions": {
    "queue":  {"type": "choice", "instructions": "Which queue handles this?",
               "criteria": {"refunds": "refund request", "returns": "return request",
                            "complaints": "complaint", "general": "other"}},
    "urgent": {"type": "noul",   "instructions": "Is this time-sensitive?"},
    "sev":    {"type": "score",  "instructions": "How severe is the complaint?",
               "criteria": ["minor", "moderate", "major"]}
  }
}'
```

The response names the selected option for each `choice`, the probability for
each `noul`, and the expected level with its distribution for each `score`.
Every answer includes a calibrated confidence.

## Devices

The server picks a device automatically — `cuda`, then `mps`, then `cpu`. Pass
`--device` to override. On Apple silicon, `--device mlx` runs the torso on
Metal through mlx-lm. `--device neuron` runs on AWS Trainium2 and is never
picked automatically.

| Flag | Install | Target |
| --- | --- | --- |
| `--device cuda` | `pip install "strands-decider[cuda]"` | NVIDIA GPU on Linux or WSL2. |
| `--device mps` | `pip install strands-decider` | Apple silicon GPU. |
| `--device mlx` | `pip install "strands-decider[mlx]"` | Apple silicon GPU through mlx-lm. |
| `--device cpu` | `pip install strands-decider` | CPU-only hosts. |
| `--device neuron` | Inside the Neuron container, see [below](#aws-trainium2---device-neuron) | AWS Trainium2 through TorchNeuron. |

### AWS Trainium2 (`--device neuron`)

The Neuron backend splits a v2 model checkpoint's text decoder over several
NeuronCores by tensor parallelism, one worker process per core. It is text
only (a request with images gets 422), runs bfloat16, always encodes the state
once per pass, and serves prompts up to the checkpoint's validated length
(32,768 tokens for the v2 models). The checkpoints it has been checked on are
listed with their measurements under [Checked checkpoints](#checked-checkpoints).

**Requirements.** A Trainium2 instance (tested on `trn2.48xlarge`), and the
Neuron deep learning container with PyTorch 2.14 and TorchNeuron from Neuron
SDK 2.33.1
(`public.ecr.aws/neuron/torch-neuronx:2.14.0-neuronx-py314-sdk2.33.1-ubuntu26.04`).

**Getting a Trainium2 instance.** The AWS Neuron documentation covers each way
to get one ([Deploy on AWS](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/deploy/index.html)):

- **Amazon EC2:** launch a trn2 instance, for example from a
  [Neuron Deep Learning AMI](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/deploy/environments/dlami.html),
  which has the Neuron driver, and run the Neuron PyTorch container on it
  ([PyTorch DLC setup](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/setup/pytorch/dlc.html)).
- **Amazon EKS:** add trn2 nodes and install the Neuron device plugin
  ([Kubernetes getting started](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/deploy/eks/setup.html)).
- **Amazon SageMaker:** see
  [Neuron on SageMaker](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/deploy/sagemaker/index.html),
  and check that the SageMaker feature you use offers trn2 instances.

Use the container's `torch` and `torch-neuronx`: a `pip install` that replaces
them breaks the backend. Install into a virtual environment that sees the
container's packages, with the Neuron stack pinned:

```bash
python3 -m venv --system-site-packages .venv && source .venv/bin/activate
pip list | awk '$1 ~ /^(torch|torch-neuronx|neuronx-cc|nki|numpy)$/ {print $1"=="$2}' > neuron.txt
pip install -c neuron.txt strands-decider
strands-decider serve "$CHECKPOINT" --device neuron --port 8000   # a checked checkpoint
```

`torchvision` is a dependency only for the 12B's image processor. If pip can't
find a `torchvision` that matches the container's `torch`, install
strands-decider with `--no-deps` and its other dependencies under the same
constraints file. The Neuron backend doesn't need `torchvision`.

**Start-up.** Each worker loads only its slice of the weights, then compiles
the graphs for the common request shapes, or loads them from the compile cache.
With the cache warm, start-up is the weight load and a cache read per shape. A request whose
shape bucket hasn't been compiled yet pays that compile once, from about 40
seconds for a short prompt to several minutes at 32K tokens. Keep the cache on
persistent storage with TorchNeuron's `TORCH_NEURONX_NEFF_CACHE_DIR` and
`TORCH_NEURONX_HLO_CACHE_DIR`. Put `TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR` on a
local disk.

#### Checked checkpoints

Each was measured on a `trn2.48xlarge` at LNC=2, in bfloat16, with Neuron SDK
2.33.1, against its own references with `tools/parity.py`. Conformance means at
least 1,000 questions, top-1 agreement of at least 0.98 and a mean absolute
probability difference of at most 0.02 (`tests/core_conformance.py`).

| Checkpoint | NeuronCores | Typed Decisions request, p50 | One question at 4K / 32K tokens | Conformance |
| --- | --- | --- | --- | --- |

**Environment variables.**

| Variable | Default | Effect |
| --- | --- | --- |
| `STRANDS_NEURON_TP` | `16` | NeuronCores (and worker processes) the model is split over. |
| `STRANDS_NEURON_CORES` | first `TP` cores | Which NeuronCores to use, such as `16-31`; sets `NEURON_RT_VISIBLE_CORES`. Lets several servers share an instance. |
| `STRANDS_NEURON_WARM` | `default` | Shapes compiled at start-up: `default`, `none`, or `state:question tokens:questions,...`, such as `512:640:5`. |
| `STRANDS_NEURON_EAGER` | `0` | `1` runs op by op without `torch.compile` (slow; for debugging). |
| `STRANDS_NEURON_CC_OVERLAP` | `0` | `1` passes neuronx-cc's collective/compute overlap option: about 4% faster, but it fails to compile the graphs for states over 4,096 tokens. Use it only when prompts stay below that. Ignored when `NEURON_CC_FLAGS` is set. |
| `STRANDS_NEURON_JOINT_MAX` | `4096` | Longest state, in tokens, packed jointly. Longer states use separate buckets and the NKI flash-attention kernel. |
| `STRANDS_NEURON_NKI_STATE` | `1` | NKI Library `attention_cte` for the state block of long prompts. `0` uses plain PyTorch, which runs out of device memory past about 20K tokens. |
| `STRANDS_NEURON_LAYERS_PER_GRAPH` | `6` | Decoder layers per compiled graph for joint buckets. |

## Flags worth knowing

- `--vision` loads the vision tower on the v1 Qwen3.5 checkpoint so requests
  may carry base64 `images`. v2 Gemma 4 models accept images without this
  flag. See [designs/images](../designs/images.md).
- `--max-concurrency N` caps in-flight requests. Each request runs to
  completion before another picks it up on the engine.
- `--max-batch N` sets the per-request question budget.
- `--device-map auto` splits a model across multiple GPUs when one is not
  enough.

## Shared-prefix cache

Asking several questions about the same state is nearly free because the
server encodes the state once and reuses the hidden representation for every
question in the request. For low-latency decisions, pack related questions
into a single POST rather than opening several.

## Deeper reading

- [designs/inference](../designs/inference.md) — full operator reference,
  calibration notes, and Mac/MLX kernel details.
- [Deployment](deployment/local.md) — local, SageMaker, Lambda, and other hosting targets.
