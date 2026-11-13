# AWS Lambda image for a decision checkpoint (../README.md, Decision checkpoints): the decoder as a
# Q8_0 GGUF and the vision tower as an mmproj GGUF that llama.cpp runs, behind the torch-free core
# (strands_decider.core) and its llama.cpp backend (strands_decider.backends.llama_cpp).
# `cdk deploy -c decision=true` builds it; by hand, from the repository root:
#   docker build --platform linux/arm64 -f deploy/lambda/decision/Dockerfile \
#     --build-arg CHECKPOINT=<Hub id, or s3:// URI with --secret for its credentials> .
# Every stage is the Lambda base image, pinned by digest, so what is built links against the
# libraries it runs on.
ARG BASE=public.ecr.aws/lambda/python:3.12@sha256:21c21867e3c999a5b9471c16e089bb44a4da0e4e4104a26056799dacadeac278

# llama.cpp at the commit in LLAMA_CPP_REV, fetched by its hash, which git checks the source
# against, with patches/ applied, each checked first, so the build fails at a patch that does not
# apply; static, with mtmd, its multimodal library; and the shim linked against it into one
# library, for every Graviton Lambda runs: the build fails if the library is not arm64 or holds an
# instruction Graviton2 lacks (scripts/build_native.sh).
FROM ${BASE} AS llama
RUN dnf -y install cmake gcc gcc-c++ git binutils && dnf clean all
COPY deploy/lambda/decision/ /opt/decision/
RUN sh /opt/decision/scripts/build_native.sh /llama /build --lambda && rm -rf /llama/.git

# The model: the checkpoint converted by llama.cpp's own converter (scripts/build_artifact.py),
# which needs torch, on CPU. Before it, scripts/shim_check.py runs the image's own library against
# transformers on a tiny random checkpoint: the build fails if the mask, state reuse, image rows or
# answers are not the reference's.
FROM ${BASE} AS model
# llama-quantize, like ggml's CPU backend everywhere, uses OpenMP; the base image has no libgomp.
RUN dnf -y install libgomp && dnf clean all
ENV PIP_NO_CACHE_DIR=1 HF_HOME=/tmp/hf
# The versions this image was tested with; torch from its CPU index, which has no CUDA libraries.
RUN pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install numpy==2.5.4 safetensors==0.8.0 sentencepiece==0.2.1 transformers==5.20.2 \
      "protobuf>=4.21,<5" pyyaml==6.0.3 tqdm==4.70.1 tokenizers==0.23.2 pillow==13.0.0 \
      boto3==1.43.127
COPY --from=llama /llama /llama
COPY --from=llama /build/bin/llama-quantize /usr/local/bin/
COPY --from=llama /build/libdecision.so /opt/libdecision.so
COPY src /opt/strands-decider/src
COPY deploy/lambda/decision/scripts /opt/strands-decider/deploy/lambda/decision/scripts
RUN python /opt/strands-decider/deploy/lambda/decision/scripts/shim_check.py \
      --llama-cpp /llama --lib /opt/libdecision.so --work /tmp/shim-check \
    && rm -rf /tmp/shim-check
ARG CHECKPOINT
ARG CHECKPOINT_REVISION=main
# The sha256 of the checkpoint's manifest (README.md, Decision checkpoints), to refuse any other.
ARG CHECKPOINT_DIGEST=
# The checkpoint goes to the container's disk and the BF16 GGUF to memory (/tmp/gguf): written to
# Docker's storage in CodeBuild, the converter's write fails with "Not enough free space" with
# hundreds of GB free. Both are gone when the step ends. An s3:// checkpoint is read with the
# credentials the secret's endpoint gives (the CodeBuild project's role), so no layer holds them.
RUN --mount=type=secret,id=aws_container_credentials_uri --mount=type=tmpfs,target=/tmp/gguf \
    if [ -s /run/secrets/aws_container_credentials_uri ]; then \
      export AWS_CONTAINER_CREDENTIALS_RELATIVE_URI="$(cat /run/secrets/aws_container_credentials_uri)"; fi \
    && python /opt/strands-decider/deploy/lambda/decision/scripts/build_artifact.py "${CHECKPOINT}" /model \
      --revision "${CHECKPOINT_REVISION}" --expect-digest "${CHECKPOINT_DIGEST}" --llama-cpp /llama \
      --quantize /usr/local/bin/llama-quantize --work /tmp/build --gguf-tmp /tmp/gguf \
    && rm -rf /tmp/build /tmp/hf \
    && chmod -R a+rX /model  # Lambda runs the function as a user other than root

FROM ${BASE}
RUN dnf -y install libgomp && dnf clean all
COPY --from=model /model /opt/model
COPY pyproject.toml README.md LICENSE /tmp/strands-decider/
COPY src /tmp/strands-decider/src
# What the core and the backend import (Pillow decodes images), then the package without its
# dependencies, which include torch.
ENV PIP_NO_CACHE_DIR=1
RUN pip install --target "${LAMBDA_TASK_ROOT}" numpy==2.5.4 tokenizers==0.23.2 pillow==13.0.0 \
    && SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0+lambda pip install --target "${LAMBDA_TASK_ROOT}" \
      --no-deps /tmp/strands-decider \
    && rm -rf /tmp/strands-decider
COPY --from=llama /build/libdecision.so ${LAMBDA_TASK_ROOT}/
COPY deploy/lambda/decision/handler.py ${LAMBDA_TASK_ROOT}/
# numpy's OpenBLAS would start a thread per core beside llama's OpenMP threads.
ENV OPENBLAS_NUM_THREADS=1
CMD ["handler.handler"]
