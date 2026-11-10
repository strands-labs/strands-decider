# Serving on Amazon SageMaker

This guide outlines how to host a strands decider model on Amazon SageMaker in your AWS
account. One AWS CDK stack runs the model on a SageMaker real-time endpoint, using the
[strands-decider AWS Deep Learning Container](https://gallery.ecr.aws/deep-learning-containers/strands-decider)
(DLC) on A10G, A100, L4, L40S, H100 and H200 instances, or an image built from this folder's `Dockerfile` on
other instances (CPU, older GPUs). The endpoint takes the same request body as `POST /v1/systemone`
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

This copies the model to S3, then creates the endpoint. On the default `ml.g6.xlarge` the endpoint
runs the published DLC image, so nothing is built; on other instances (or with `-c image=build`) the
stack also builds the image from `Dockerfile` in AWS CodeBuild ([Image](#image)).

> If you deployed a GPU endpoint with an earlier version of this stack, the next `cdk deploy` moves
> it to the DLC image and removes the image build from the stack. Add `-c image=build` to keep
> serving the image built from your checkout. If CloudFormation can't delete the stack's ECR
> repository because it still holds images, delete the repository yourself.

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
2. On `ml.g5`, `ml.g6`, `ml.g6e`, `ml.p4d(e)` and `ml.p5(e/en)` instances, nothing: the endpoint pulls the DLC image
   `763104351884.dkr.ecr.<region>.amazonaws.com/strands-decider:serve-sagemaker-cuda-v1` (the
   account differs in some regions; the stack picks it). With `-c image=build`, or on other
   instance types, the image built from `Dockerfile` for `linux/amd64` by a CodeBuild project
   ([deploy-time-build](https://github.com/cdklabs/deploy-time-build)) and published to an ECR
   repository in the stack. Either image holds the package but no model weights, and runs
   `strands-decider serve` on port 8080. The server answers SageMaker's `GET /ping` and
   `POST /invocations` as `/health` and `/v1/systemone`.
3. The SageMaker model, endpoint config and endpoint, with an execution role that can read the
   bucket, pull the image and write logs. SageMaker copies the model from the bucket to
   `/opt/ml/model` on each instance before the container starts (`ModelDataSource`), and the
   container loads it with `HF_HUB_OFFLINE=1`, so the endpoint needs no internet access.
   On GPU instances the endpoint config selects the Amazon Linux 2023 host image
   (`al2023-ami-sagemaker-inference-gpu-4-1`: NVIDIA driver 580, which runs the DLC's CUDA 12.8
   and the `Dockerfile` image's CUDA 12.6). SageMaker has no Amazon Linux 2023 option for CPU instances, so they
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
| `fallback_instance_types` | empty | Up to four more instance types, comma-separated, that SageMaker tries in order when it has no capacity for `instance_type` (instance pools), e.g. `ml.g6e.xlarge,ml.g5.xlarge`. They must use the same device and image as `instance_type`, and each must hold the model ([Choosing a model](#choosing-a-model)): the endpoint uses the same settings (such as `device_map`) on all of them. |
| `image` | empty | Where the endpoint's image comes from: `dlc` (the published Deep Learning Container) or `build` (build `Dockerfile` in CodeBuild). Empty picks `dlc` on the instance families the DLC supports and `build` on the others ([Image](#image)). |
| `dlc_image_tag` | `serve-sagemaker-cuda-v1` | The DLC tag. `-v1` points at the latest compatible release (rebuilt weekly with security patches) when an endpoint is created; a running endpoint keeps the image it started with until its SageMaker model is replaced. Pin a full version such as `serve-sagemaker-cuda-v1.0.0` to choose one image. |
| `max_concurrency` | empty | Requests that run on the model at once; the others wait (`serve --max-concurrency`). Empty means 1 on GPU: a request's questions are already batched on the GPU, and without a cap a burst of large requests can run the GPU out of memory. On CPU, empty means no cap. `0` removes the cap (up to 40 at once). |
| `device_map` | empty | `auto` splits a model larger than one GPU across the instance's GPUs (`serve --device-map`), e.g. the 26B-A4B on `ml.g6e.12xlarge` |
| `inference_ami_version` | empty | A SageMaker [inference AMI version](https://docs.aws.amazon.com/sagemaker/latest/dg/instance-types-ami.html) to pin the host AMI to, for GPU families outside the default AL2023 GPU AMI's compatibility list. Empty keeps the default (AL2023 on GPU, SageMaker default on CPU). |
| `serve_args` | empty | More `strands-decider serve` options, such as `"--max-batch 8"` |

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
| `StrandsAgents/strands-decider-26B-A4B-gemma4-v1-2610` | about 51 GB | about 101 GB | `ml.g6e.12xlarge` (four L40S) with `-c device_map=auto`, a GPU with more than 51 GB such as `ml.p5.4xlarge` (one H100), or `ml.r5.4xlarge` on CPU |
| `StrandsAgents/strands-decider-2B-hobson-v21`, `-hobson-v19` | about 4.5 GB | about 9 GB | As for the 2B Qwen3.5 model |

The main [README](../../README.md#gemma-4-models) compares the models' accuracy. The DLC is
tested with every v1 model on these GPUs (2B, E2B and E4B on one L4, 12B on one L40S, 26B-A4B
split across L40S GPUs), and the default model on a SageMaker endpoint. The memory figures are
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
the region; no instance ran and nothing was billed. Give SageMaker fallbacks to try in order
(each needs its own quota, and must hold the model). An endpoint with several instances can end up
on a mix of these types, with different speeds:

```bash
npx cdk deploy -c fallback_instance_types=ml.g6e.xlarge,ml.g5.xlarge
```

Or try another region, or deploy on a CPU instance:

| Instance | vCPUs | Memory |
| --- | --- | --- |
| `ml.m5.2xlarge` | 8 | 32 GB |
| `ml.m5.4xlarge` | 16 | 64 GB |
| `ml.c6i.2xlarge` | 8 | 16 GB |
| `ml.r5.2xlarge` | 8 | 64 GB |

```bash
npx cdk deploy -c instance_type=ml.m5.2xlarge
```

The DLC image is GPU-only, so a CPU instance type builds the image from `Dockerfile` ([Image](#image)).
CPU endpoints run requests in parallel by default (no `max_concurrency` cap).

On CPU the model runs in fp32, so pick an instance with more memory than the fp32 weights
([Choosing a model](#choosing-a-model)). The default model on `ml.m5.2xlarge` answers a
request with three questions in about 2.5 s.

## Image

| `image` | What runs | When |
| --- | --- | --- |
| `dlc` (default on `ml.g5`, `ml.g6`, `ml.g6e`, `ml.p4d(e)`, `ml.p5(e/en)`) | The [strands-decider Deep Learning Container](https://gallery.ecr.aws/deep-learning-containers/strands-decider): Amazon Linux 2023, PyTorch 2.7.1 with CUDA 12.8, published in every region and rebuilt weekly with security patches | GPU deployments on those families (its kernels are built for A10G, A100, L4, L40S, H100 and H200), unless you need a change to the package that isn't released in the DLC yet |
| `build` (default elsewhere) | `Dockerfile` in this folder, built from this checkout in CodeBuild during `cdk deploy` | CPU instances, older GPUs (`ml.g4dn`, `ml.p3`), or serving code from your checkout |

```bash
npx cdk deploy -c image=build   # build Dockerfile even on a GPU instance
```

The DLC maps environment variables to `strands-decider serve` options (`DECIDER_MAX_CONCURRENCY`,
`DECIDER_DEVICE_MAP`, `DECIDER_ARGS`, ...); the stack sets them from the settings above, and passes
the same settings as `serve` options to the `Dockerfile` image. One difference: the DLC splits
`serve_args` like a shell command line (`--model-name "fraud triage"` works), while the `Dockerfile`
image splits it on spaces, so values with spaces need the DLC.

## Calling it from your code

Send the `/v1/systemone` body with `ContentType="application/json"`:

```python
import json, boto3

runtime = boto3.client("sagemaker-runtime")
response = runtime.invoke_endpoint(
    EndpointName="strands-decider-2b-qwen3-5-v1-2610",
    ContentType="application/json",
    Body=json.dumps(
        {
            "state": "Help! My payouts have been failing for 3 days!",
            "questions": {
                "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}
            },
        }
    ),
)
print(json.loads(response["Body"].read())["answers"])
```

Callers need IAM permission `sagemaker:InvokeEndpoint` on the endpoint. The container has no
authentication of its own.

The notebook [samples/ask-the-endpoint.ipynb](samples/ask-the-endpoint.ipynb) calls the endpoint
step by step: each question type, the confidence, a check of an agent's tool call, and the error
for a request that is not valid.

## Deleting it

An endpoint costs money for every hour it runs, whether or not it gets requests. When you
no longer need it:

```bash
npx cdk destroy                    # the default model
npx cdk destroy -c model=<Hub id>  # any other model you deployed
```

This deletes everything that model's `cdk deploy` created: the endpoint, the endpoint config,
the SageMaker model, the S3 bucket with the model, the ECR repository with the image (when it
built one), the CodeBuild projects and the roles.
