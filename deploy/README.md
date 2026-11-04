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

## AWS Lambda

Serverless, on arm64 CPUs: no GPU, no instance to reserve, nothing standing between requests.
One CDK stack builds an image with the 2B Qwen3.5 model in it, quantized for llama.cpp, in AWS
CodeBuild, and creates a function behind an IAM-authenticated Function URL; requests take
seconds. See [lambda/README.md](lambda/README.md).
