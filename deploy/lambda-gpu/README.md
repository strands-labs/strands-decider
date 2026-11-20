# Serving on Lambda Managed Instances GPU (preview)

**Preview**. [Lambda Managed Instances GPU](https://preview.docs.aws.dev/preview/ngzhang/IAD/AWSLambdaDocs/lmi-gpu-overview-page/lambda/latest/dg/lambda-managed-instances-gpu.html)
runs Lambda Event Functions on GPU-accelerated EC2 instances (G4dn/G5/G6/G6e/G6f/Gr6/Gr6f).
Lambda provisions the instances, installs and patches the NVIDIA driver, and assigns one GPU to
each execution environment. At the time of writing this is a preview service: its CloudFormation
resource types (`AWS::Lambda::CapacityProvider`, and `AcceleratorConfig` on
`AWS::Lambda::Function`) are only reachable from accounts allowlisted to the preview, and
`cdk deploy` will fail with a resource-type error otherwise. This directory is a scaffold: it
gets `cdk synth` to CloudFormation, but a successful `cdk deploy` depends on preview access and
an EC2 GPU quota increase (see below).

Unlike `deploy/lambda` (arm64 CPU, llama.cpp), this deploy keeps the model on torch + CUDA — the
same code path as `deploy/sagemaker` — so the engine sees the model's full weights and the GPU
does the forward pass. GPU-accelerated LMI functions are **Event Functions only**: they handle
direct `Invoke` calls and event-source records (SQS, SNS, S3, EventBridge, ...). They have no
Function URL and no synchronous HTTP path.

**Concurrency.** The image follows the pattern the preview docs recommend
([`lambda-managed-instances-gpu-inference.html`](https://docs.aws.amazon.com/lambda/latest/dg/lambda-managed-instances-gpu-inference.html#lambda-managed-instances-gpu-inference-custom-handler)):
the bootstrap script starts `strands-decider serve` in the background during initialization,
before Lambda forks worker processes. Each worker then imports `handler.py`, which is a thin
HTTP forwarder to `http://127.0.0.1:8000/v1/systemone`. One copy of the model serves every
concurrent invocation in the execution environment, so `PerExecutionEnvironmentMaxConcurrency`
can go up to 48 without duplicating the model in GPU memory. The server queues on its own GPU
lock and (optionally) on `--max-concurrency`.

To follow this guide you need:

- An AWS account with preview access to Lambda Managed Instances GPU, and credentials that can
  deploy CloudFormation stacks that create IAM roles.
- An increase to the EC2 **Running On-Demand G and VT instances** quota in your region. The
  default is 0 vCPUs; request enough vCPUs for your `instance_types` × desired concurrency in the
  [EC2 Service Quotas console](https://console.aws.amazon.com/servicequotas/home/services/ec2/quotas/).
- Node.js 22 or later. The CDK runs through `npx`, so you do not install it separately.
- The AWS CLI v2, to invoke the function and read its logs.

Install dependencies and bootstrap CDK once for each account and region:

```bash
cd deploy/lambda-gpu
npm install
npx cdk bootstrap aws://<account>/<region>
```

## 1. Deploy

```bash
npx cdk deploy
```

This builds the GPU image in AWS CodeBuild, publishes it to an ECR repository in the stack,
creates a Lambda capacity provider over the configured GPU instance types, and creates the
function with `AcceleratorConfig` attached. About 15 minutes: most of it is torch + CUDA layers
and the model download.

## 2. Test it

```bash
aws lambda invoke --function-name <FunctionArn> --cli-binary-format raw-in-base64-out \
  --payload '{"state": "Help! My payouts have been failing for 3 days!",
              "questions": {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}}}' \
  answer.json
cat answer.json
```

```json
{"model": "strands-decider-2B-qwen3.5-v1-2610", "answers": {"is_urgent": {"type": "noul", "noul": 0.86}}, "usage": {"input_tokens": 86, "output_tokens": 1}, "latency_ms": 42.5}
```

Event-source invocations work too: send the same JSON as the body of an SQS message, or as the
`detail` of an EventBridge event, and the handler dispatches it to the same engine
(`function/handler.py`, `_request_from`).

## What `cdk deploy` creates

One CloudFormation stack per model, `<name>-lambda-gpu`, holding:

1. **The image**, built from `Dockerfile` for `linux/amd64` by a CodeBuild project
   ([deploy-time-build](https://github.com/cdklabs/deploy-time-build)) and published to an ECR
   repository in the stack. The build installs torch 2.7.1 + CUDA 12.6 + strands-decider with
   the `[cuda]` extras, downloads the model and its base model from the Hugging Face Hub, and
   bakes them into the image. LMI GPU allows container images up to 32 GB.
2. **The capacity provider** (`AWS::Lambda::CapacityProvider`): the GPU instance types Lambda may
   launch for this function, and an optional `MaxVCpus` cap.
3. **The function**: x86_64, `MemorySize` from `memory_size_mb`, 15-minute timeout,
   `AcceleratorConfig.AcceleratorMemorySize` from `accelerator_memory_gb`, and
   `PerExecutionEnvironmentMaxConcurrency` from `per_env_max_concurrency`.
4. **The alias** `live`, pointing at the latest version. Each deploy that changes the function
   publishes a new version and deletes the one before.

## Configuration

Every setting is a CDK context value. Change it in `cdk.json`, or pass `-c key=value` to
`cdk deploy`:

| Setting | Default | Meaning |
| --- | --- | --- |
| `model` | `StrandsAgents/strands-decider-2B-qwen3.5-v1-2610` | The strands-decider model, a Hugging Face Hub id. Names the stack. |
| `model_revision` | `main` | The model's revision (a branch, tag, or commit). |
| `instance_types` | `g4dn.xlarge,g4dn.2xlarge,g4dn.4xlarge` | Comma list of GPU EC2 instance types Lambda may launch. Must be in `g4dn / g5 / g6 / gr6 / g6e / g6f / gr6f`. g4dn (T4) is the default because it has the widest availability; switch to g6 (L4, 24 GB) or g5 (A10G, 24 GB) with `-c instance_types=g6.xlarge,g6.2xlarge -c accelerator_memory_gb=24`. |
| `accelerator_memory_gb` | `16` | GPU memory per execution environment. One of `3, 6, 12, 16, 24, 48`. Lambda picks the smallest qualifying instance from `instance_types`, and when several types have the same memory, the latest generation. |
| `max_vcpus` | `32` | Cap on vCPUs the capacity provider may launch. `0` for no cap (still bounded by your EC2 quota). |
| `per_env_max_concurrency` | `4` | Invocations in-flight per execution environment (max 48). Safe to raise because the image uses one shared server per execution environment; workers only forward over `127.0.0.1`. |
| `server_max_concurrency` | *unset* | Passed to `strands-decider serve --max-concurrency`, which caps simultaneous forwards on the GPU. Unset means no cap beyond the server's default thread pool. |
| `memory_size_mb` | `4096` | Lambda host memory. LMI GPU allows 2048–32768 MB. |
| `ephemeral_storage_mb` | `4096` | `/tmp` size. LMI GPU allows 512–32768 MB. |
| `capacity_provider_service_role_arn` | *unset* | Pre-existing service role for the capacity provider. If unset, the stack creates one and attaches `AWSLambdaManagedInstanceServiceRolePolicy`. Override if the preview account uses a different managed policy name. |

## Known gaps

1. **CloudFormation resource types.** aws-cdk-lib 2.273 ships L2 constructs for
   `AWS::Lambda::CapacityProvider` and the function's `CapacityProviderConfig`, and
   `capacityProvider.addFunction(fn, { perExecutionEnvironmentMaxConcurrency })` wires those
   together. `AcceleratorConfig.AcceleratorMemorySize` is the GPU preview piece not yet in the
   shipped CFN schema, so the stack sets it with a raw property override on the `CfnFunction`.
   `cdk synth` warns with `F3002 (Additional properties are not allowed)` for this one override;
   the deploy fails against a non-preview account until the schema catches up.
2. **GPU base image.** The preview docs say Lambda publishes GPU base images with CUDA, PyTorch,
   `awslambdaric`, and a prebuilt SOCI index. Their ECR path is not public yet, so this Dockerfile
   uses `pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime` and installs `awslambdaric` explicitly.
   To switch, replace the `FROM` line and drop the `pip install awslambdaric` step.
3. **SOCI / lazy loading.** Startup optimization (Seekable OCI, Run:ai Model Streamer) is not
   wired up. With the model baked in and no SOCI index, a new execution environment's cold start
   will include pulling several GB from ECR.
4. **Instance entrypoint must be an absolute path to a regular executable file.** LMI GPU
   rejects `ImageConfig.EntryPoint[0]` if it is a bare command (`python`), a PATH-resolved name,
   or a symlink. The Dockerfile writes a tiny `/var/task/bootstrap` shell script and the stack
   points the function at that script via `toLambdaDockerImageCode({ entrypoint, cmd,
   workingDirectory })`. Setting the same props as top-level Function properties does not work;
   CDK reads them from the code object's `bind()` result.
5. **Rollback deletes logs.** `AWS::Lambda::Managed-Instances` function failures (e.g.
   `RuntimeInitError`) often leave no events in CloudWatch, so the log group is set to
   `RemovalPolicy.RETAIN` and the deploy uses `--no-rollback`. Delete the retained log groups
   by hand after a teardown.
6. **GPU capacity varies by family/AZ.** LMI's capacity provider picks the latest-generation
   instance type with the smallest qualifying GPU memory and does not fall back to an older
   family in the same `instance_types` list. In a region where `g6.xlarge` is out of stock,
   adding `g4dn.xlarge` alongside it does not help — Lambda keeps retrying the newer family.
   The default (`g4dn` + 16 GB) is the most available choice; `g4dn` is not offered in
   `us-west-2d`, so pass `-c availability_zones=us-west-2a,us-west-2b,us-west-2c` when using
   that default. The stack also sets `minExecutionEnvironments=1` so the first publish does
   not ask for three instances at once.
7. **Model download uses `git clone` + Git-LFS.** The pytorch base image's `huggingface_hub` 1.x
   pulls in `httpx2`, whose Brotli/zstandard content-encoding decoders fail with
   "TypeError: process() takes no keyword arguments" against the bundled decompressor wheels.
   `scripts/prepare_model.py` clones the adapter and the base model with `git`/`git-lfs`
   instead and materializes the base model into `HF_HOME` so `transformers.AutoConfig` /
   `AutoModelForCausalLM` resolve offline (`HF_HUB_OFFLINE=1`).

## Deleting it

```bash
npx cdk destroy                    # the default model
npx cdk destroy -c model=<Hub id>  # any other model you deployed
```

This deletes everything the deploy created: the function, versions, alias, capacity provider,
ECR repository, CodeBuild project and roles. The build log groups (`/aws/codebuild/*`) stay.
