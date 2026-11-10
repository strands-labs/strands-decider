# AWS Lambda image for a decision checkpoint (../README.md, Decision checkpoints): the decoder as a
# Q8_0 GGUF and the vision tower as an mmproj GGUF that llama.cpp runs, behind the torch-free core
# (strands_decider.core) and its llama.cpp backend (strands_decider.backends.llama_cpp).
# `cdk deploy -c decision=true` builds it; by hand, from the repository root:
#   docker build --platform linux/arm64 -f deploy/lambda/decision/Dockerfile \
#     --build-arg CHECKPOINT=<Hub id, or s3:// URI with --secret for its credentials> .
# Every stage is the Lambda base image, so what is built links against the libraries it runs on.
ARG BASE=public.ecr.aws/lambda/python:3.12

# llama.cpp at the commit this image was tested with, fetched by its hash, which git checks the
# source against, with patches/ applied; static, with mtmd, its multimodal library; and the shim
# linked against it into one library.
FROM ${BASE} AS llama
RUN dnf -y install cmake gcc gcc-c++ git && dnf clean all
ARG LLAMA_CPP_REV=26d52b0f9543f441fc2bfa383557c992a4fbf4b9
WORKDIR /llama
COPY deploy/lambda/decision/patches/ /tmp/patches/
RUN git init -q . && git fetch -q --depth 1 https://github.com/ggml-org/llama.cpp "${LLAMA_CPP_REV}" \
    && git checkout -q FETCH_HEAD && test "$(git rev-parse HEAD)" = "${LLAMA_CPP_REV}" \
    && for p in /tmp/patches/*.patch; do git apply "$p"; done && rm -rf .git
# armv8.2-a+fp16+dotprod, without i8mm: Lambda runs arm64 functions on Graviton2, 3 and 4, a
# SnapStart snapshot taken on one restores on any, and an i8mm instruction kills the process on
# Graviton2. The build fails if the library has one.
RUN cmake -S . -B /build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DGGML_NATIVE=OFF \
      -DGGML_CPU_ARM_ARCH=armv8.2-a+fp16+dotprod -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF \
      -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF \
    && cmake --build /build --target llama llama-quantize mtmd -j"$(nproc)"
COPY deploy/lambda/decision/shim.c /build/
# --no-undefined: a symbol no archive defines fails the build, not the function's first load.
RUN gcc -O3 -fPIC -shared -o /build/libdecision.so /build/shim.c \
      -Iinclude -Iggml/include -Itools/mtmd -Wl,--no-undefined \
      -Wl,--start-group $(find /build -name '*.a') -Wl,--end-group \
      -fopenmp -lstdc++ -lm -lpthread -ldl \
    && objdump -d /build/libdecision.so > /build/libdecision.s \
    && if grep -q smmla /build/libdecision.s; then echo "i8mm in libdecision.so" >&2; exit 1; fi

# The model: the checkpoint converted by llama.cpp's own converter (scripts/build_artifact.py),
# which needs torch, on CPU.
FROM ${BASE} AS model
# llama-quantize, like ggml's CPU backend everywhere, uses OpenMP; the base image has no libgomp.
RUN dnf -y install libgomp && dnf clean all
ENV PIP_NO_CACHE_DIR=1 HF_HOME=/tmp/hf
# The versions this image was tested with; torch from its CPU index, which has no CUDA libraries.
RUN pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install numpy==2.5.4 safetensors==0.8.0 sentencepiece==0.2.1 transformers==5.20.2 \
      "protobuf>=4.21,<5" pyyaml tqdm tokenizers==0.23.2 boto3
COPY --from=llama /llama /llama
COPY --from=llama /build/bin/llama-quantize /usr/local/bin/
COPY src /opt/strands-decider/src
COPY deploy/lambda/decision/scripts /opt/strands-decider/deploy/lambda/decision/scripts
ARG CHECKPOINT
ARG CHECKPOINT_REVISION=main
ARG LLAMA_CPP_REV=26d52b0f9543f441fc2bfa383557c992a4fbf4b9
# The vision tower's type. q8_0 matches the reference as closely as f16 and encodes an image about
# three times as fast on Graviton, where ggml's f16 matrix products are slow.
ARG MMPROJ_TYPE=q8_0
# The checkpoint goes to the container's disk and the BF16 GGUF to memory (/tmp/gguf): written to
# Docker's storage in CodeBuild, the converter's write fails with "Not enough free space" with
# hundreds of GB free. Both are gone when the step ends. An s3:// checkpoint is read with the
# credentials the secret's endpoint gives (the CodeBuild project's role), so no layer holds them.
RUN --mount=type=secret,id=aws_container_credentials_uri --mount=type=tmpfs,target=/tmp/gguf \
    if [ -s /run/secrets/aws_container_credentials_uri ]; then \
      export AWS_CONTAINER_CREDENTIALS_RELATIVE_URI="$(cat /run/secrets/aws_container_credentials_uri)"; fi \
    && export LLAMA_CPP_REV \
    && python /opt/strands-decider/deploy/lambda/decision/scripts/build_artifact.py "${CHECKPOINT}" /model \
      --revision "${CHECKPOINT_REVISION}" --llama-cpp /llama --quantize /usr/local/bin/llama-quantize \
      --mmproj-type "${MMPROJ_TYPE}" --work /tmp/build --gguf-tmp /tmp/gguf \
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
