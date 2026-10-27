#!/bin/bash
# User data for the temporary compile host (Neuron DLAMI, Ubuntu 24.04) that handler.py starts.
#
# It copies the model data from __SOURCE__, runs `strands-decider compile-neuron` in the serving
# image with the Neuron device passed in, and uploads the model data with the compiled graphs
# (neuron-cache/) to __TARGET__, which the SageMaker model serves. Then it writes __TARGET___SUCCESS
# or __TARGET___FAILED and shuts down, which terminates the instance. It also shuts down after a
# fixed time whatever happens.
#
# The container has no network (--network none) and no AWS credentials: the instance metadata hop
# limit of 1 keeps it from the instance role. Its output goes to CloudWatch Logs, __LOG_GROUP__.
# handler.py fills in the placeholder values below.
set -euo pipefail
exec > >(tee -a /var/log/neuron-compile.log) 2>&1
shutdown -h +150

REGION=__REGION__
SOURCE=__SOURCE__
TARGET=__TARGET__
IMAGE=__IMAGE__
CHECKPOINT=__CHECKPOINT__
LOG_GROUP=__LOG_GROUP__
MODEL=/opt/ml/model

finish() {
  local rc=$? marker=_FAILED
  set +e
  if [ $rc -eq 0 ]; then
    echo "uploading to $TARGET"
    aws s3 sync --region "$REGION" --only-show-errors "$MODEL/" "$TARGET" && marker=_SUCCESS
  fi
  aws s3 cp --region "$REGION" --only-show-errors /var/log/neuron-compile.log "${TARGET}compile.log"
  echo "exit code $rc" | aws s3 cp --region "$REGION" --only-show-errors - "${TARGET}${marker}"
  shutdown -h now
}
trap finish EXIT

command -v aws > /dev/null || snap install aws-cli --classic
if ! command -v docker > /dev/null; then
  apt-get update -q && apt-get install -y -q docker.io
fi
mkdir -p "$MODEL"
aws s3 sync --region "$REGION" --only-show-errors "$SOURCE" "$MODEL/"
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "${IMAGE%%/*}"
docker pull -q "$IMAGE"
# /opt/ml/model inside the container, as on SageMaker: the compile cache records the paths it ran at
docker run --rm --network none --device=/dev/neuron0 -v "$MODEL:/opt/ml/model" \
  --log-driver=awslogs --log-opt awslogs-region="$REGION" --log-opt awslogs-group="$LOG_GROUP" \
  --entrypoint strands-decider "$IMAGE" \
  compile-neuron "/opt/ml/model/$CHECKPOINT" /opt/ml/model/neuron-cache
