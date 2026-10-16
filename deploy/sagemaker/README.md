# Serving on Amazon SageMaker

This guide outlines how to host a strands decider model on Amazon SageMaker in your AWS
account. One AWS CDK stack builds the model image and runs it on a SageMaker real-time
endpoint. The endpoint takes the same request body as `POST /v1/systemone`
([docs/inference.md, Serve](../../docs/inference.md#serve)) and returns the same response.

To follow it you need:

- An AWS account, and credentials that can deploy CloudFormation stacks that create IAM roles.
- Node.js 22 or later. It runs the AWS CDK through `npx`, so you do not install the CDK
  separately.
- [awscurl](https://github.com/okigan/awscurl) (`pip install awscurl`), to test the endpoint,
  and the AWS CLI v2, to read its logs.
- Quota for one `ml.g6.xlarge for endpoint usage` in the region (Service Quotas, SageMaker),
  or for the instance type you choose ([Choosing an instance](#choosing-an-instance)).

Install the CDK app's dependencies, and bootstrap CDK once for each account and region:

```bash
cd deploy/sagemaker
npm install
npx cdk bootstrap aws://<account>/<region>
```

## 1. Deploy

```bash
npx cdk deploy
```

This copies the model to S3 and builds the image in AWS CodeBuild at the same time, then creates
the endpoint.

## 2. Test it

```bash
awscurl --service sagemaker --region <region> -X POST -H 'content-type: application/json' \
  -d '{"state": "Help! My payouts have been failing for 3 days!",
       "questions": {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}}}' \
  https://runtime.sagemaker.<region>.amazonaws.com/endpoints/strands-decider-2b-qwen3-5-v1-2610/invocations
```

```json
{"model":"strands-decider-2B-qwen3.5-v1-2610","answers":{"is_urgent":{"type":"noul","noul":0.8581}},"usage":{"input_tokens":86,"output_tokens":1},"latency_ms":1013.13}
```

## Debugging

The container's output, including any error while the model loads, is in CloudWatch Logs:

```bash
aws logs tail /aws/sagemaker/Endpoints/strands-decider-2b-qwen3-5-v1-2610 --region <region> --since 1h --follow
```

The endpoint's state, and why it failed if it did:

```bash
aws sagemaker describe-endpoint --endpoint-name strands-decider-2b-qwen3-5-v1-2610 --region <region> \
  --query '[EndpointStatus, FailureReason]'
```

If the endpoint cannot start, CloudFormation rolls the stack back and deletes what it created,
so the endpoint is gone after a failed deploy. The log group stays.

## What `cdk deploy` creates

One CloudFormation stack per model, named after it like its endpoint
([Deploying several models](#deploying-several-models)): `strands-decider-2b-qwen3-5-v1-2610` for the default.
It holds:

1. A private S3 bucket with the model. A CodeBuild job (`scripts/fetch-model.ts`) downloads the model
   from the Hugging Face Hub, and the base model it loads at the revision the model records
   (`Qwen/Qwen3.5-2B-Base` for the default, about 4.5 GB), and copies both to
   `s3://<bucket>/<endpoint>/<model_revision>/`. It runs again when `model` or `model_revision`
   changes.
2. The image, built from `Dockerfile` for `linux/amd64` by a CodeBuild project
   ([deploy-time-build](https://github.com/cdklabs/deploy-time-build)) and published to an ECR
   repository in the stack. It holds the package but no model weights, and runs
   `strands-decider serve` on port 8080. The server answers SageMaker's `GET /ping` and
   `POST /invocations` as `/health` and `/v1/systemone`.
3. The SageMaker model, endpoint config and endpoint, with an execution role that can read the
   bucket, pull the image and write logs. SageMaker copies the model from the bucket to
   `/opt/ml/model` on each instance before the container starts (`ModelDataSource`), and the
   container loads it with `HF_HUB_OFFLINE=1`, so the endpoint needs no internet access.
   On GPU instances the endpoint config selects the Amazon Linux 2023 host image
   (`al2023-ami-sagemaker-inference-gpu-4-1`: NVIDIA driver 580, CUDA 13.0, which runs the
   image's CUDA 12.6). SageMaker has no Amazon Linux 2023 option for CPU instances, so they
   use its default host image.

`cdk deploy` waits until the endpoint is `InService`, and prints its name, its device, the S3
location of its model and its invoke URL.

## Configuration

Every setting is a CDK context value. Change it in `cdk.json`, or pass `-c key=value` to
`cdk deploy`:

```bash
npx cdk deploy -c model=StrandsAgents/strands-decider-E4B-gemma4-v1-2610 -c instance_type=ml.g6.xlarge
```

| Setting | Default | Meaning |
| --- | --- | --- |
| `model` | `StrandsAgents/strands-decider-2B-qwen3.5-v1-2610` | The strands-decider model, a Hugging Face Hub id ([Choosing a model](#choosing-a-model)). It also names the stack and the endpoint. |
| `model_revision` | `main` | The model's revision. Use a release's date tag (`YYMMDD`) to pin it ([docs/inference.md, Model artifact](../../docs/inference.md#model-artifact)). |
| `instance_type` | `ml.g6.xlarge` | The SageMaker instance type ([Choosing an instance](#choosing-an-instance)) |
| `instance_count` | `1` | Instances behind the endpoint |
| `serve_args` | empty | More `strands-decider serve` options, such as `"--max-batch 8 --strict-window"` |

A change to a setting updates the endpoint in place: SageMaker moves traffic to the new
endpoint config when it is ready.

## Deploying several models

Each model gets its own stack and endpoint, named after it: the part of the Hub id after the
`/`, lowercased, with characters other than letters, digits and hyphens turned into hyphens.
So you can deploy several models side by side in one account and region:

```bash
npx cdk deploy                                                       # strands-decider-2b-qwen3-5-v1-2610
npx cdk deploy -c model=StrandsAgents/strands-decider-E4B-gemma4-v1-2610   # strands-decider-e4b-gemma4-v1-2610
```

Deploying the same model again updates its endpoint, for example with a new `instance_type`.
Each endpoint runs, and is billed for, its own instances.

## Choosing a model

| `model` | Weights in bf16 (GPU) | Weights in fp32 (CPU) | Instance |
| --- | --- | --- | --- |
| `StrandsAgents/strands-decider-2B-qwen3.5-v1-2610` (default) | about 4.5 GB | about 9 GB | `ml.g6.xlarge`, or `ml.m5.2xlarge` on CPU |
| `StrandsAgents/strands-decider-E2B-gemma4-v1-2610` | about 9 GB | about 19 GB | `ml.g6.xlarge`, or `ml.m5.4xlarge` on CPU |
| `StrandsAgents/strands-decider-E4B-gemma4-v1-2610` | about 15 GB | about 30 GB | `ml.g6.xlarge`, or `ml.m5.4xlarge` on CPU |
| `StrandsAgents/strands-decider-12B-gemma4-v1-2610` | about 23 GB | about 46 GB | `ml.g6e.xlarge`, or `ml.r5.2xlarge` on CPU |
| `StrandsAgents/strands-decider-26B-A4B-gemma4-v1-2610` | about 51 GB | about 101 GB | A GPU with more than 51 GB, such as `ml.p5.4xlarge` (one H100), or `ml.r5.4xlarge` on CPU |
| `StrandsAgents/strands-decider-2B-hobson-v21`, `-hobson-v19` | about 4.5 GB | about 9 GB | As for the 2B Qwen3.5 model |

The main [README](../../README.md#gemma-4-models) compares the models' accuracy. Only the
default has been deployed on SageMaker with this guide; the memory figures for the others are
from [docs/inference.md](../../docs/inference.md#gemma-4-models), and activations come on top.

## Choosing an instance

The stack picks the device from the instance type: `cuda` on NVIDIA GPU instances (`ml.g*`,
`ml.p*`), `cpu` on the other x86 instances. Graviton (arm64), Inferentia and Trainium
instances are not supported, and `cdk deploy` stops with an error for them.

The recommended setup is a GPU:

| Instance | GPU | GPU memory |
| --- | --- | --- |
| `ml.g6.xlarge` (default) | one NVIDIA L4 | 24 GB |
| `ml.g5.xlarge` | one NVIDIA A10G | 24 GB |
| `ml.g6e.xlarge` | one NVIDIA L40S | 48 GB |
| `ml.p5.4xlarge` | one NVIDIA H100 | 80 GB |

A quota does not reserve capacity, and GPU instances are often in short supply. If the deploy
fails with `InsufficientInstanceCapacity`, SageMaker found no free instance of that type in
the region; no instance ran and nothing was billed.
Try another GPU type or region, or deploy on a CPU instance:

| Instance | vCPUs | Memory |
| --- | --- | --- |
| `ml.m5.2xlarge` | 8 | 32 GB |
| `ml.m5.4xlarge` | 16 | 64 GB |
| `ml.c6i.2xlarge` | 8 | 16 GB |
| `ml.r5.2xlarge` | 8 | 64 GB |

```bash
npx cdk deploy -c instance_type=ml.m5.2xlarge
```

On CPU the model runs in fp32, so pick an instance with more memory than the fp32 weights
([Choosing a model](#choosing-a-model)). The default model on `ml.m5.2xlarge` answers a
request with three questions in about 2.5 s.

## Calling it from your code

Send the `/v1/systemone` body with `ContentType="application/json"`:

```python
import json, boto3

runtime = boto3.client("sagemaker-runtime")
response = runtime.invoke_endpoint(
    EndpointName="strands-decider-2b-qwen3-5-v1-2610",
    ContentType="application/json",
    Body=json.dumps({
        "state": "Help! My payouts have been failing for 3 days!",
        "questions": {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}},
    }),
)
print(json.loads(response["Body"].read())["answers"])
```

Callers need IAM permission `sagemaker:InvokeEndpoint` on the endpoint. The container has no
authentication of its own.

## Deleting it

An endpoint costs money for every hour it runs, whether or not it gets requests. When you
no longer need it:

```bash
npx cdk destroy                    # the default model
npx cdk destroy -c model=<Hub id>  # any other model you deployed
```

This deletes everything that model's `cdk deploy` created: the endpoint, the endpoint config,
the SageMaker model, the S3 bucket with the model, the ECR repository with the image, the
CodeBuild projects and the roles.
