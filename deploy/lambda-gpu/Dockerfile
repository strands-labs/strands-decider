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
    HF_HUB_OFFLINE=1

# Lambda Managed Instances GPU validates ImageConfig.EntryPoint[0] is an absolute path to an
# executable regular file. The pytorch/pytorch base image's python is at /opt/conda/bin/python
# (a symlink to python3.X), and Lambda rejects the symlink target chase. A tiny wrapper script
# at a fixed absolute path gets around it.
RUN printf '#!/bin/sh\nexec /opt/conda/bin/python -m awslambdaric "$@"\n' > /var/task/bootstrap \
    && chmod 755 /var/task/bootstrap
ENTRYPOINT ["/var/task/bootstrap"]
CMD ["handler.handler"]
