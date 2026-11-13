# Deploying Strands Decider

There are a number of ways to host Strands Decider yourself.

## Hosting options

### SageMaker JumpStart

Host Strands Decider behind a SageMaker model endpoint.

- **Pros:** dedicated host and auto-scaling.
- **Cons:** pay for compute.

| Instance type | Cost / hr | Single-question latency | Max concurrency |
| --- | --- | --- | --- |
| Basic CPU | _TBD_ | _TBD_ | _TBD_ |
| Trainium | _TBD_ | _TBD_ | _TBD_ |
| GPU | _TBD_ | _TBD_ | _TBD_ |

_Cost, latency, and concurrency numbers coming soon from benchmarks._

### Lambda

Deploy a Lambda function that responds with the model response.

- **Pros:** pay per request.
- **Cons:** slow cold start, quantized models.

### Lambda managed instances

Deploy to a GPU Lambda function.

- **Pros:** pay per request (you pay for instances).
- **Cons:** cold start when scaled to zero (where supported).

## Also supported

- **EKS** — deploy alongside your existing cluster.
- **GCP** — deploy to Google Cloud.
- **Azure** — deploy to Azure.

_How-to guides for these targets coming soon._

## Choosing a model version

Compare Strands Decider model variants across features, supported deployment
targets, image support, and benchmarks.

| Model | Base | Params | Image support | Deployment targets | Benchmark score |
| --- | --- | --- | --- | --- | --- |
| _Placeholder_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| _Placeholder_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| _Placeholder_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

_Model comparison data coming soon._
