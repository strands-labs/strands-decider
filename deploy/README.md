# Deploying strands decider

You can deploy the `strands-decider` pip server (`strands-decider serve`,
[docs/inference.md, Serve](../docs/inference.md#serve)) as-is to any compute. Below is a list of
recommended ways to host this model yourself.

## Your machine

Run `strands-decider serve` on Linux with CUDA, on a Mac, or on CPU. See
[docs/inference.md, Environments for serving](../docs/inference.md#environments-for-serving).

## Amazon SageMaker

A real-time endpoint in your AWS account, on a GPU or CPU instance. One CDK stack copies the model
to a private S3 bucket, builds the image in AWS CodeBuild and creates the endpoint; `cdk destroy`
removes it. See
[sagemaker/README.md](sagemaker/README.md).

## Amazon EKS

An EKS Auto Mode cluster in your AWS account, on Graviton CPU nodes, with an autoscaler that adds
pods and nodes under load. One CDK stack creates the cluster, copies the model to S3 and builds the
image in AWS CodeBuild; `cdk destroy` removes it. See [eks/README.md](eks/README.md).
