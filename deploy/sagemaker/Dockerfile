# SageMaker inference image for strands-decider: the package, without model weights (README.md).
# `cdk deploy` builds it; by hand, from the repository root:
#   docker build --platform linux/amd64 -f deploy/sagemaker/Dockerfile -t strands-decider-sagemaker .
# DEVICE=cuda (the default) is for GPU instances. DEVICE=cpu leaves out flash-linear-attention,
# whose Triton kernels need a GPU, for CPU instances; the CUDA build of torch also runs on CPU.
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

# Triton compiles a small C launcher at run time (training/README.md, Setup).
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf

WORKDIR /opt/strands-decider
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
# No .git in the build context, so setuptools-scm takes the version from here.
ARG PACKAGE_VERSION=0.0.0+sagemaker
ARG DEVICE=cuda
# transformers is pinned to the last release tested with this image's torch 2.7.1: 5.20.0 imports
# torch.distributed.checkpoint.hf_storage, which torch 2.7 does not have, and Qwen3.5 fails to load.
ARG TRANSFORMERS_VERSION=5.19.0
RUN SETUPTOOLS_SCM_PRETEND_VERSION=${PACKAGE_VERSION} \
    pip install ".[${DEVICE}]" "transformers==${TRANSFORMERS_VERSION}"
# On CUDA, causal-conv1d is the fused kernel for the Gated DeltaNet conv layer; without it
# transformers runs a slower PyTorch path. Its PyPI release only builds from source against the
# CUDA toolkit, which this runtime image lacks, so install the prebuilt wheel for this image's
# CUDA 12, torch 2.7, Python 3.11 and C++ ABI.
ARG CAUSAL_CONV1D_VERSION=1.7.0
RUN if [ "${DEVICE}" = cuda ]; then \
      abi=$(python -c "import torch; print(str(torch._C._GLIBCXX_USE_CXX11_ABI).upper())") && \
      pip install "https://github.com/Dao-AILab/causal-conv1d/releases/download/v${CAUSAL_CONV1D_VERSION}/causal_conv1d-${CAUSAL_CONV1D_VERSION}+cu12torch2.7cxx11abi${abi}-cp311-cp311-linux_x86_64.whl"; \
    fi

# The model is not in the image: SageMaker copies it from S3 to /opt/ml/model before the
# container starts (the stack's ModelDataSource, scripts/fetch-model.ts), and the stack sets
# DECIDER_CHECKPOINT to its folder there. DECIDER_ARGS adds `strands-decider serve` options, such
# as "--max-batch 8".
ENV DECIDER_CHECKPOINT=/opt/ml/model/checkpoint \
    DECIDER_DEVICE=${DEVICE} \
    DECIDER_ARGS="" \
    HF_HUB_OFFLINE=1

EXPOSE 8080
# SageMaker runs `docker run IMAGE serve`; that trailing `serve` becomes $0 and is ignored.
ENTRYPOINT ["sh", "-c", "exec strands-decider serve \"$DECIDER_CHECKPOINT\" --host 0.0.0.0 --port 8080 --device \"$DECIDER_DEVICE\" $DECIDER_ARGS"]
