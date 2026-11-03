# Serving on Amazon EKS

This guide hosts a strands decider model on an Amazon EKS Auto Mode cluster in your AWS account,
on Graviton (arm64) CPU nodes. One AWS CDK stack creates the cluster, copies the model to S3,
builds the image and runs `strands-decider serve`
([docs/inference.md, Serve](../../docs/inference.md#serve)), with an autoscaler that adds pods
and nodes under load.

To follow it you need:

- An AWS account, and credentials that can deploy CloudFormation stacks that create IAM roles.
- Node.js 22 or later. It runs the AWS CDK through `npx`.
- The AWS CLI v2 and `kubectl`, to test the service.
- A region that offers the default instance type, `m9g.4xlarge`, or another type in
  `instance_types` ([Configuration](#configuration)). To check:
  `aws ec2 describe-instance-type-offerings --region <region> --location-type availability-zone --filters Name=instance-type,Values=m9g.4xlarge --query 'InstanceTypeOfferings[].Location'`

Install the CDK app's dependencies, and bootstrap CDK once for each account and region:

```bash
cd deploy/eks
npm install
npx cdk bootstrap aws://<account>/<region>
```

## 1. Deploy

Set `admin_principal` in `cdk.json` to the ARN of the IAM role or user you run `kubectl` as, so
that every later deploy keeps it. For the role of your current credentials (with an IAM user,
use the `Arn` that `aws sts get-caller-identity` prints):

```bash
aws iam list-roles --output text --query \
  "Roles[?RoleName=='$(aws sts get-caller-identity --query Arn --output text | cut -d/ -f2)'].Arn"
```

```bash
npx cdk deploy
```

This takes about 20 minutes, most of it creating the cluster. The first deploy finishes when a
pod is ready; later ones return once Kubernetes has the change.

## 2. Test it

```bash
aws eks update-kubeconfig --name strands-decider-2b-qwen3-5-v1-2610-eks --region <region>
```

In a second terminal, run `kubectl -n strands-decider port-forward svc/strands-decider 8080`, then:

```bash
curl -s -X POST localhost:8080/v1/systemone -H 'content-type: application/json' \
  -d '{"state": "Help! My payouts have been failing for 3 days!",
       "questions": {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}}}'
```

```json
{"model":"strands-decider-2B-qwen3.5-v1-2610","answers":{"is_urgent":{"type":"noul","noul":0.8581}},"usage":{"input_tokens":86,"output_tokens":1},"latency_ms":380.88}
```

Inside the cluster, call `http://strands-decider.strands-decider.svc.cluster.local:8080`. For
callers elsewhere in the VPC, set `internal_nlb` to `true` in `cdk.json`: the stack adds an
internal Network Load Balancer on port 80, and prints its URL as `InternalNlbUrl`. Run at least 2
pods behind it (`min_replicas`). The server has no authentication of its own, so the stack never
makes it internet-facing, and the load balancer accepts connections only from the VPC's
addresses.

## Debugging

```bash
kubectl -n strands-decider get pods,nodes
kubectl -n strands-decider rollout status deploy/strands-decider        # after a change
kubectl -n strands-decider logs deploy/strands-decider -c fetch-model   # the copy from S3
kubectl -n strands-decider logs deploy/strands-decider                  # the server
```

A pod that stays `Pending` has no node: check that `instance_types` are offered in the region and
have at least 32 GiB of memory. `kubectl get nodeclaims` shows why Auto Mode could not launch one.
If no pod is ready 10 minutes into the first deploy, the deploy fails and CloudFormation deletes
the stack; `npx cdk deploy --no-rollback` keeps it for debugging. `kubectl` answering
`Unauthorized` means `admin_principal` is missing or wrong.

## What `cdk deploy` creates

One CloudFormation stack per model, named after it with `-eks` at the end
(`strands-decider-2b-qwen3-5-v1-2610-eks` for the default). It holds:

1. A VPC in two zones, with one NAT gateway and an S3 gateway endpoint.
2. An EKS Auto Mode cluster, with a public API endpoint that needs IAM authentication, and the
   metrics-server add-on for the autoscaler, which Auto Mode runs on a small node of its own.
3. A node pool that launches only the Graviton types in `instance_types` (Auto Mode's built-in
   general-purpose pool is x86 only). It adds nodes for pods that do not fit, and removes nodes
   2 minutes after they are empty.
4. A private S3 bucket with the model and its base model, copied from the Hugging Face Hub by
   [`../sagemaker/scripts/fetch-model.ts`](../sagemaker/scripts/fetch-model.ts) in AWS CodeBuild.
5. The image, built from `Dockerfile` on an arm64 CodeBuild host and published to an ECR
   repository in the stack. It holds the package but no model weights.
6. A Deployment whose pods copy the model from S3 into memory (they can read only that bucket,
   through EKS Pod Identity), then run `strands-decider serve --device cpu` with
   `HF_HUB_OFFLINE=1`. A new pod on a new node is ready in about 45 seconds.
7. A Service, and a HorizontalPodAutoscaler that adds pods when their CPU use passes 60%.

## Configuration

Every setting is a CDK context value. Change it in `cdk.json`, or pass `-c key=value` to
`cdk deploy`:

| Setting | Default | Meaning |
| --- | --- | --- |
| `model` | `StrandsAgents/strands-decider-2B-qwen3.5-v1-2610` | The strands-decider model, a Hugging Face Hub id. It also names the stack and the cluster. Pods are sized for the 2B models; only this one has been tested. Larger models do not fit. |
| `model_revision` | `main` | The model's revision. Use a release's date tag (`YYMMDD`) to pin it. |
| `instance_types` | `m9g.4xlarge` | Graviton types for the node pool, comma-separated, each with at least 32 GiB of memory. Not every region offers m9g; `m8g.4xlarge` is more widely available. |
| `threads` | `16` | Torch threads per pod. Each pod requests 85% of this many vCPUs, rounded down, and 16 GiB of memory. |
| `min_replicas`, `max_replicas` | `1`, `4` | The autoscaler's bounds. With `min_replicas` 1, a node replacement leaves no pod for about 45 seconds; use 2 to avoid that. |
| `admin_principal` | empty | The IAM role or user ARN to give `kubectl` access ([Deploy](#1-deploy)) |
| `internal_nlb` | `false` | Add an internal Network Load Balancer |
| `serve_args` | empty | More `strands-decider serve` options |

## Performance

Measured with the default model and settings in eu-west-1, on one pod (16 threads on an
m9g.4xlarge):

| Request | Latency |
| --- | --- |
| One noul question (the test above) | about 400 ms |
| A five-way choice and a noul over a short message | about 1.2 s |

On CPU the torso runs in fp32, and `serve` runs requests that arrive together side by side on
the same cores, which slows them all down. One pod answers about 0.8 of the 1.2 s requests per
second when they come one at a time, and 0.24 per second with 8 in flight. So scale with pods:
set `min_replicas` for your load, and the autoscaler adds pods up to `max_replicas`. It scales on
CPU, so one caller sending requests back to back keeps 2 pods running.

## Deleting it

The cluster, its nodes and the NAT gateway cost money for every hour the stack exists. When you
no longer need it:

```bash
npx cdk destroy                    # the default model
npx cdk destroy -c model=<Hub id>  # any other model you deployed
```

This deletes everything `cdk deploy` created except the CloudWatch log groups of the build and
Lambda functions. It takes up to 45 minutes, mostly waiting for AWS Lambda to release the network
interfaces of the function that runs `kubectl` in the VPC.
