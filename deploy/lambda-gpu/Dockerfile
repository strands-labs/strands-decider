# AWS Lambda Managed Instances GPU image for strands-decider (README.md): torch + CUDA, the
# strands-decider package, and the model baked in. `cdk deploy` builds it; by hand, from the
# repository root:
#   docker build --platform linux/amd64 -f deploy/lambda-gpu/Dockerfile --build-arg MODEL=<Hub id> .
#
# The base image: NVIDIA's PyTorch runtime with CUDA 12.6 and cuDNN 9, same as the SageMaker
# image's. Lambda Managed Instances also publishes GPU base images with awslambdaric preinstalled
# (per the preview docs); using the public pytorch image + explicit awslambdaric install lets the
# same Dockerfile work before the preview base image is generally available. To switch to a
# Lambda-published GPU base image, change FROM and drop the awslambdaric install.
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime AS base

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates git git-lfs \
    && git lfs install --system \
    && rm -rf /var/lib/apt/lists/*

# The Lambda Runtime Interface Client turns the container into a Lambda target: it reads
# AWS_LAMBDA_RUNTIME_API and routes invocations to the Python handler named in CMD.
RUN pip install awslambdaric==3.1.0

# ---- Download the model into the image ---------------------------------------------------------
FROM base AS model
WORKDIR /opt/strands-decider
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
ARG TRANSFORMERS_VERSION=5.19.0
# [cuda] pulls flash-linear-attention and other GPU extras; `.` installs the package.
RUN SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0+lambda-gpu \
    pip install ".[cuda]" "transformers==${TRANSFORMERS_VERSION}"
COPY deploy/lambda-gpu/scripts/prepare_model.py ./
ARG MODEL
ARG MODEL_REVISION=main
RUN python prepare_model.py "${MODEL}" "${MODEL_REVISION}" /tmp/out \
    && mkdir -p /model && mv /tmp/out/checkpoint /model/checkpoint \
    && du -sh /model/checkpoint /opt/hf

# ---- Final runtime image -----------------------------------------------------------------------
FROM base
WORKDIR /var/task
# The strands-decider package and its [cuda] extras, in the Lambda task root.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
ARG TRANSFORMERS_VERSION=5.19.0
RUN SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0+lambda-gpu \
    pip install ".[cuda]" "transformers==${TRANSFORMERS_VERSION}"
# causal-conv1d (same prebuilt wheel as the SageMaker image).
ARG CAUSAL_CONV1D_VERSION=1.7.0
RUN abi=$(python -c "import torch; print(str(torch._C._GLIBCXX_USE_CXX11_ABI).upper())") && \
    pip install "https://github.com/Dao-AILab/causal-conv1d/releases/download/v${CAUSAL_CONV1D_VERSION}/causal_conv1d-${CAUSAL_CONV1D_VERSION}+cu12torch2.7cxx11abi${abi}-cp311-cp311-linux_x86_64.whl"

COPY --from=model /model /opt/model
# prepare_model.py also materialized the base model into HF_HOME so transformers.AutoConfig /
# AutoModelForCausalLM resolve the base model offline. Copy the cache into the final image.
COPY --from=model /opt/hf /opt/hf
COPY deploy/lambda-gpu/function/handler.py ./

ENV DECIDER_CHECKPOINT=/opt/model/checkpoint \
    DECIDER_DEVICE=cuda \
    DECIDER_SERVER_PORT=8000 \
    DECIDER_SERVER_READY_TIMEOUT=600 \
    HF_HUB_OFFLINE=1

# One inference server per execution environment, started before Lambda forks workers so one
# copy of the model serves every concurrent invocation (lambda-managed-instances-gpu-inference).
# Each worker process runs handler.py, which forwards to the server over 127.0.0.1. Loading the
# model once from the handler would fail at PerExecutionEnvironmentMaxConcurrency > 1, because
# N workers would load N copies into GPU memory.
#
# Lambda Managed Instances GPU validates ImageConfig.EntryPoint[0] as an absolute path to a
# regular executable file (no symlinks, no PATH), so the bootstrap script lives at a known path.
RUN cat > /var/task/bootstrap <<'BOOT' && chmod 755 /var/task/bootstrap
#!/bin/sh
set -eu
PY=/opt/conda/bin/python
PORT="${DECIDER_SERVER_PORT:-8000}"
# Start the strands-decider inference server. --no-warmup keeps the server startup under
# control: handler.py runs the warm-up itself on first forward through the server.
/opt/conda/bin/strands-decider serve \
    "${DECIDER_CHECKPOINT}" \
    --host 127.0.0.1 --port "$PORT" \
    --device "${DECIDER_DEVICE:-cuda}" \
    ${DECIDER_MAX_CONCURRENCY:+--max-concurrency "$DECIDER_MAX_CONCURRENCY"} &
SERVER_PID=$!
# Block until /health answers OK, or the init times out. Lambda marks the version Active only
# when `python -m awslambdaric` is ready for events, so this wait is inside the init window.
"$PY" - <<'PY' "$PORT" "${DECIDER_SERVER_READY_TIMEOUT:-600}" || { echo "server never became ready; strands-decider serve pid=$SERVER_PID still running? $(kill -0 "$SERVER_PID" 2>/dev/null && echo yes || echo no)"; exit 1; }
import json, sys, time, urllib.error, urllib.request
port, timeout = int(sys.argv[1]), int(sys.argv[2])
deadline = time.monotonic() + timeout
while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
            if r.status == 200:
                print("decider server ready:", r.read(200).decode())
                sys.exit(0)
    except (urllib.error.URLError, ConnectionError, TimeoutError):
        pass
    time.sleep(1)
sys.exit(1)
PY
exec "$PY" -m awslambdaric "$@"
BOOT
ENTRYPOINT ["/var/task/bootstrap"]
CMD ["handler.handler"]
