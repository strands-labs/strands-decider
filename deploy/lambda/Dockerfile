# AWS Lambda image for strands-decider (README.md): the torso as a GGUF that llama.cpp runs,
# and the rest in the package's torch-free HiddenStateEngine. `cdk deploy` builds it; by hand, from
# the repository root:
#   docker build --platform linux/arm64 -f deploy/lambda/Dockerfile --build-arg MODEL=<Hub id> .
# Every stage is the Lambda base image, so what is built links against the libraries it runs on.
ARG BASE=public.ecr.aws/lambda/python:3.12

# llama.cpp, static, and the shim linked against it into one library, with mtmd, llama.cpp's
# multimodal library, for an image built with vision on.
FROM ${BASE} AS llama
RUN dnf -y install cmake gcc gcc-c++ git && dnf clean all
# The llama.cpp commit this image was tested with, fetched by its hash, which git checks the
# source against.
ARG LLAMA_CPP_REV=6184e92c57dcd34de8a3e381d7641a5e75250d5f
WORKDIR /llama
RUN git init -q . && git fetch -q --depth 1 https://github.com/ggml-org/llama.cpp "${LLAMA_CPP_REV}" \
    && git checkout -q FETCH_HEAD && test "$(git rev-parse HEAD)" = "${LLAMA_CPP_REV}" && rm -rf .git
# armv8.2-a+fp16+dotprod, without i8mm: Lambda runs arm64 functions on Graviton2, 3 and 4, a
# SnapStart snapshot taken on one restores on any, and an i8mm instruction kills the process on
# Graviton2. The build fails if the library has one.
RUN cmake -S . -B /build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DGGML_NATIVE=OFF \
      -DGGML_CPU_ARM_ARCH=armv8.2-a+fp16+dotprod -DLLAMA_CURL=OFF \
    && cmake --build /build --target llama llama-quantize mtmd -j"$(nproc)"
COPY deploy/lambda/function/shim.c /build/
# --no-undefined: a symbol no archive defines fails the build, not the function's first load.
RUN gcc -O3 -fPIC -shared -o /build/libdecider.so /build/shim.c \
      -Iinclude -Iggml/include -Itools/mtmd -Wl,--no-undefined \
      -Wl,--start-group /build/tools/mtmd/libmtmd.a /build/vendor/hash/libvendor-hash.a \
      /build/src/libllama.a /build/ggml/src/libggml*.a -Wl,--end-group \
      -fopenmp -lstdc++ -lm \
    && objdump -d /build/libdecider.so > /build/libdecider.s \
    && if grep -q smmla /build/libdecider.s; then echo "i8mm in libdecider.so" >&2; exit 1; fi

# The model: the LoRA folded into the torso (torch, on CPU), converted to GGUF and quantized.
FROM ${BASE} AS model
# llama-quantize, like ggml's CPU backend everywhere, uses OpenMP; the base image has no libgomp.
RUN dnf -y install libgomp && dnf clean all
ENV PIP_NO_CACHE_DIR=1 HF_HOME=/tmp/hf
WORKDIR /opt/strands-decider
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
# The versions this image was tested with; torch from its CPU index, which has no CUDA libraries.
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
    && SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0+lambda pip install . transformers==5.20.2
# llama.cpp's source, for its converter, and its quantizer.
COPY --from=llama /llama /llama
COPY --from=llama /build/bin/llama-quantize /usr/local/bin/
COPY deploy/lambda/scripts/prepare_model.py ./
ARG MODEL
ARG MODEL_REVISION=main
# true adds the base model's vision tower, as an mmproj GGUF, for requests with images.
ARG VISION=false
# The weights' precision (README.md, Precision): the decoder q8_0 or f16, the vision tower f16 or f32.
ARG DECODER_PRECISION=q8_0
ARG VISION_PRECISION=f16
# Each step deletes what the next does not read: the downloads and the merged torso are about
# 4.5 and 3.8 GB. The GGUFs are written to memory (/tmp/gguf): written to Docker's storage in
# CodeBuild, the converter's write fails with "Not enough free space" with hundreds of GB free.
RUN --mount=type=tmpfs,target=/tmp/gguf \
    case "${DECODER_PRECISION}/${VISION_PRECISION}" in q8_0/f16|q8_0/f32|f16/f16|f16/f32) ;; \
      *) echo "DECODER_PRECISION q8_0 or f16, VISION_PRECISION f16 or f32" >&2; exit 1;; esac \
    && python prepare_model.py "${MODEL}" "${MODEL_REVISION}" /tmp/out "${VISION}" "${DECODER_PRECISION}" \
    && mkdir /model \
    && if [ "${VISION}" = true ]; then \
      python /llama/convert_hf_to_gguf.py /tmp/out/vision --mmproj --outtype "${VISION_PRECISION}" \
        --outfile /tmp/gguf/vision.gguf && mv /tmp/gguf/vision.gguf /model/; fi \
    && rm -rf /tmp/hf \
    && python /llama/convert_hf_to_gguf.py /tmp/out/torso --outtype f16 --outfile /tmp/gguf/torso.gguf \
    && rm -rf /tmp/out/torso && mv /tmp/out/assets /model/ \
    && if [ "${DECODER_PRECISION}" = q8_0 ]; then \
      llama-quantize /tmp/gguf/torso.gguf /model/decider.gguf Q8_0; \
    else mv /tmp/gguf/torso.gguf /model/decider.gguf; fi \
    && rm -rf /tmp/out \
    && chmod -R a+rX /model  # Lambda runs the function as a user other than root

FROM ${BASE}
RUN dnf -y install libgomp && dnf clean all
COPY --from=model /model /opt/model
COPY pyproject.toml README.md LICENSE /tmp/strands-decider/
COPY src /tmp/strands-decider/src
# What HiddenStateEngine imports (Pillow decodes images), then the package without its
# dependencies, which include torch.
ENV PIP_NO_CACHE_DIR=1
RUN pip install --target "${LAMBDA_TASK_ROOT}" \
      pydantic==2.14.0 numpy==2.5.3 tokenizers==0.23.2 safetensors==0.8.0 pillow==12.0.0 \
    && SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0+lambda pip install --target "${LAMBDA_TASK_ROOT}" \
      --no-deps /tmp/strands-decider \
    && rm -rf /tmp/strands-decider
COPY --from=llama /build/libdecider.so ${LAMBDA_TASK_ROOT}/
COPY deploy/lambda/function/handler.py deploy/lambda/function/backend.py ${LAMBDA_TASK_ROOT}/
# numpy's OpenBLAS would start a thread per core beside llama's OpenMP threads.
ENV OPENBLAS_NUM_THREADS=1
CMD ["handler.handler"]
