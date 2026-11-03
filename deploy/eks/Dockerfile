# EKS inference image for strands-decider on Graviton (arm64, CPU): the package, without model
# weights (README.md). `cdk deploy` builds it on an arm64 CodeBuild host; by hand, from the
# repository root:
#   docker build --platform linux/arm64 -f deploy/eks/Dockerfile -t strands-decider-eks .
FROM public.ecr.aws/docker/library/python:3.12.15-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

WORKDIR /opt/strands-decider
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
# No .git in the build context, so setuptools-scm takes the version from here.
ARG PACKAGE_VERSION=0.0.0+eks
# The CPU build of torch (the PyPI arm64 wheel also carries CUDA), and the transformers release
# tested with it.
ARG TORCH_VERSION=2.14.1
ARG TRANSFORMERS_VERSION=5.20.2
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH_VERSION}" \
    && SETUPTOOLS_SCM_PRETEND_VERSION=${PACKAGE_VERSION} \
       pip install . "transformers==${TRANSFORMERS_VERSION}"

# The model is not in the image: an init container copies it from S3 to /opt/ml/model, in the
# layout ../sagemaker/scripts/fetch-model.ts writes, and the stack sets DECIDER_CHECKPOINT to its
# folder there. DECIDER_ARGS adds `strands-decider serve` options.
ENV DECIDER_CHECKPOINT=/opt/ml/model/checkpoint \
    DECIDER_ARGS="" \
    HF_HUB_OFFLINE=1

# The server needs no root. The pod's init container writes the model as root, readable by all.
RUN useradd --create-home --uid 1000 decider
USER 1000

EXPOSE 8080
ENTRYPOINT ["sh", "-c", "exec strands-decider serve \"$DECIDER_CHECKPOINT\" --host 0.0.0.0 --port 8080 --device cpu $DECIDER_ARGS"]
