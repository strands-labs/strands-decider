#!/bin/bash
# Upload the GGUF and the function zip, then deploy the stack.
#
#   BUCKET=my-bucket examples/lambda/deploy.sh
#
# Uses the AWS CLI's usual credentials and region (AWS_PROFILE, AWS_REGION). The
# bucket must exist in that region. Optional: STACK (default strands-decider-lambda),
# PREFIX (S3 key prefix, default strands-decider-lambda), SNAPSTART (on or off, default on).
#
#   SNAPSTART=on   a new execution environment restores from a snapshot in about 4 s, and the
#                  snapshot cache is billed for as long as the version exists, about $40 a
#                  month at 10 GB
#   SNAPSTART=off  nothing is billed while idle, and a new execution environment downloads and
#                  loads the model before its first request, about 30 s
set -euo pipefail
cd "$(dirname "$0")"
: "${BUCKET:?set BUCKET to an S3 bucket in the deploy region}"
STACK=${STACK:-strands-decider-lambda}
PREFIX=${PREFIX:-strands-decider-lambda}
case "${SNAPSTART:-on}" in
  on) SNAPSTART_APPLY_ON=PublishedVersions ;;
  off) SNAPSTART_APPLY_ON=None ;;
  *) echo "SNAPSTART must be on or off, not '$SNAPSTART'"; exit 1 ;;
esac

GGUF=build/model/decider-q8_0.gguf
ZIP=build/function.zip
for f in "$GGUF" "$ZIP"; do [ -f "$f" ] || { echo "missing $f -- run build.sh"; exit 1; }; done

MODEL_KEY=$PREFIX/model/decider-q8_0.gguf
# Key the zip by its hash, so a changed zip publishes a new version and an unchanged one does not.
CODE_KEY=$PREFIX/code/function-$(shasum -a 256 "$ZIP" | cut -c1-12).zip

LOCAL_SIZE=$(stat -f%z "$GGUF" 2>/dev/null || stat -c%s "$GGUF")
REMOTE_SIZE=$(aws s3api head-object --bucket "$BUCKET" --key "$MODEL_KEY" \
  --query ContentLength --output text 2>/dev/null || echo 0)
if [ "$REMOTE_SIZE" = "$LOCAL_SIZE" ]; then
  echo ">> s3://$BUCKET/$MODEL_KEY is already there, skipping the model upload"
else
  echo ">> uploading the model ($(du -h "$GGUF" | cut -f1)) -> s3://$BUCKET/$MODEL_KEY"
  aws s3 cp "$GGUF" "s3://$BUCKET/$MODEL_KEY" --only-show-errors
fi
echo ">> uploading the code -> s3://$BUCKET/$CODE_KEY"
aws s3 cp "$ZIP" "s3://$BUCKET/$CODE_KEY" --only-show-errors

echo ">> deploying $STACK, SnapStart ${SNAPSTART:-on} (publishing a SnapStart version runs the full init once; a few minutes)"
aws cloudformation deploy --stack-name "$STACK" --template-file template.yaml \
  --capabilities CAPABILITY_IAM CAPABILITY_AUTO_EXPAND \
  --parameter-overrides ModelBucket="$BUCKET" ModelKey="$MODEL_KEY" CodeKey="$CODE_KEY" \
    SnapStart="$SNAPSTART_APPLY_ON"

# Every deploy publishes a version, and Lambda bills each SnapStart version's snapshot
# cache for as long as the version exists (Python snapshots do not expire). Delete the
# versions the alias has moved off; the Function URL serves the alias.
LIVE=$(aws lambda get-alias --function-name "$STACK" --name live --query FunctionVersion --output text)
for v in $(aws lambda list-versions-by-function --function-name "$STACK" \
  --query "Versions[?Version!='\$LATEST' && Version!='$LIVE'].Version" --output text); do
  echo ">> deleting superseded version $v (live is $LIVE)"
  aws lambda delete-function --function-name "$STACK" --qualifier "$v"
done
aws cloudformation describe-stacks --stack-name "$STACK" --query 'Stacks[0].Outputs' --output table
echo "Try it: python examples/lambda/client.py --stack $STACK ask"
