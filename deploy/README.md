# Deploying strands decider

You can deploy the `strands-decider` pip server (`strands-decider serve`,
[designs/inference.md, Serve](../designs/inference.md#serve)) as-is to any compute. Below is a list of
recommended ways to host this model yourself.

## Your machine

Run `strands-decider serve` on Linux with CUDA, on a Mac, or on CPU. See
[designs/inference.md, Environments for serving](../designs/inference.md#environments-for-serving).

## Amazon SageMaker

A real-time endpoint in your AWS account, on a GPU or CPU instance. One CDK stack copies the model
to a private S3 bucket, builds the image in AWS CodeBuild and creates the endpoint; `cdk destroy`
removes it. See
[sagemaker/README.md](sagemaker/README.md).

## AWS Lambda

Serverless, on arm64 CPUs: no GPU, no instance to reserve, nothing standing between requests.
One CDK stack builds an image with the 2B Qwen3.5 model in it, quantized for llama.cpp, in AWS
CodeBuild, and creates a function behind an IAM-authenticated Function URL; requests take
seconds. With `-c vision=true`, it answers over images too. See [lambda/README.md](lambda/README.md).

## AWS Lambda Managed Instances GPU (preview)

Serverless on GPU-accelerated Lambda (G4dn/G5/G6/G6e/G6f/Gr6/Gr6f). One CDK stack builds a torch
+ CUDA image with the model baked in, publishes it to an ECR repository in the stack, creates a
Lambda capacity provider, and attaches it to a GPU-accelerated Event Function (direct `Invoke`,
SQS, SNS, S3, EventBridge — no Function URL). Preview access and an EC2 GPU vCPU quota are
required. See [lambda-gpu/README.md](lambda-gpu/README.md).
