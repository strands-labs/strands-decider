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

## Choosing an AWS host

How SageMaker, Bedrock AgentCore Runtime and Lambda MicroVMs compare on speed and start-up, and what
to watch for on each: [aws-hosts.md](aws-hosts.md).
