import { readFileSync } from "fs";
import * as path from "path";
import { CfnOutput, IgnoreMode, RemovalPolicy, Stack, StackProps } from "aws-cdk-lib";
import * as codebuild from "aws-cdk-lib/aws-codebuild";
import * as ecrAssets from "aws-cdk-lib/aws-ecr-assets";
import * as iam from "aws-cdk-lib/aws-iam";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as sagemaker from "aws-cdk-lib/aws-sagemaker";
import { Construct } from "constructs";
import { ContainerImageBuild, NodejsBuild } from "@cdklabs/deploy-time-build";
import { NeuronCompile } from "./neuron-compile";

/**
 * The stack and endpoint name for a model's Hugging Face Hub id: its name, lowercased, with
 * every character a SageMaker endpoint name does not allow replaced by a hyphen.
 * StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 -> strands-decider-2b-qwen3-5-v1-2610.
 */
export function endpointNameFor(model: string): string {
  const name = (model.split("/").pop() ?? "")
    .toLowerCase()
    .replace(/[^a-z0-9-]+/g, "-")
    .replace(/^-+|-+$/g, "");
  if (!/^[a-z0-9](-*[a-z0-9]){0,62}$/.test(name)) {
    throw new Error(`${model}: cannot make a SageMaker endpoint name (1 to 63 letters, digits, hyphens) from it`);
  }
  return name;
}
const GPU_AMI_VERSION = "al2023-ami-sagemaker-inference-gpu-4-1";

const REPO_ROOT = path.join(__dirname, "..", "..", "..");

// The versions scripts/fetch-model.ts runs with in CodeBuild: the ones package.json pins, so the
// build and a local `npm run build` agree.
const PACKAGE = JSON.parse(readFileSync(path.join(__dirname, "..", "package.json"), "utf8"));
const FETCH_DEPENDENCIES = ["@huggingface/hub", "tsx"].map(
  (name) => `${name}@${(PACKAGE.devDependencies[name] as string).replace(/^[\^~]/, "")}`,
);

// The build uploads this directory to S3; keep only what the image needs, as
// Dockerfile.dockerignore does for a plain docker build.
const ASSET_EXCLUDE = [
  "*",
  "!pyproject.toml",
  "!README.md",
  "!LICENSE",
  "!src/",
  "!deploy/sagemaker/Dockerfile",
  "!deploy/sagemaker/Dockerfile.neuron",
  "**/__pycache__",
  "**/*.egg-info",
];

/**
 * The image's device for a SageMaker instance type: `cuda` on NVIDIA GPU families
 * (ml.g4dn, ml.g5, ml.g6, ml.p4d, ...), `neuron` on one-chip Inferentia2 (ml.inf2.xlarge,
 * ml.inf2.8xlarge), `cpu` on x86 CPU families. Graviton (ml.c7g, ml.m6gd, ...), Inferentia1,
 * Trainium and the larger inf2 types fail: the server warms each NeuronCore in turn, about 12
 * minutes each, and their 12 or 24 cores would outlast SageMaker's 60-minute startup limit.
 */
export function deviceFor(instanceType: string): "cuda" | "cpu" | "neuron" {
  const family = instanceType.replace(/^ml\./, "").split(".")[0];
  if (family === "inf2") {
    if (!/^ml\.inf2\.(xlarge|8xlarge)$/.test(instanceType)) {
      throw new Error(
        `${instanceType}: of Inferentia2, only ml.inf2.xlarge and ml.inf2.8xlarge (one chip, two ` +
          "NeuronCores) are supported; warming more cores outlasts the 60-minute startup limit",
      );
    }
    return "neuron";
  }
  if (/^(inf|trn)\d/.test(family)) {
    throw new Error(`${instanceType}: of Inferentia and Trainium, only ml.inf2 is supported`);
  }
  if (/^[a-z]+\d+[a-z]*g[a-z]*$/.test(family) && !/^g\d/.test(family)) {
    throw new Error(`${instanceType}: Graviton (arm64) instances are not supported; the image is x86`);
  }
  return /^(g|p)\d/.test(family) ? "cuda" : "cpu";
}

/**
 * Whether an instance type has local NVMe storage. SageMaker keeps the model data there, and
 * rejects VolumeSizeInGB for these types; the others get an EBS volume of the size set below.
 * GPU families from ml.g4dn on, and families with a `d` (ml.m5d, ml.c6id, ...), have it.
 */
export function hasInstanceStorage(instanceType: string): boolean {
  const family = instanceType.replace(/^ml\./, "").split(".")[0];
  return /^(g4dn|g5|g6|g6e|g7|g7e|p4d|p4de|p5|p5e|p5en|p6)/.test(family) || /^[a-z]+\d+[a-z]*d[a-z]*$/.test(family);
}

// Room for the largest model and its base model (about 52 GB for the 26B-A4B), with margin.
const MODEL_VOLUME_GB = 128;

/**
 * The strands-decider SageMaker endpoint (README.md): a private bucket with the model, the image,
 * the execution role, the SageMaker model, the endpoint config and the endpoint. Settings are CDK
 * context values (cdk.json, or -c key=value).
 */
export interface DeciderEndpointStackProps extends StackProps {
  /** The endpoint name, from endpointNameFor(model). */
  readonly endpointName: string;
}

/** `compile_vpc` and `compile_subnets` (comma-separated), both or neither. */
export function compileNetwork(
  vpcId: string | undefined,
  subnets: string | undefined,
): { vpcId: string; subnetIds: string[] } | undefined {
  const subnetIds = (subnets ?? "").split(",").map((s) => s.trim()).filter(Boolean);
  if (!vpcId && subnetIds.length === 0) {
    return undefined;
  }
  if (!vpcId || subnetIds.length === 0) {
    throw new Error("compile_vpc and compile_subnets go together: the VPC and subnets in it to compile in");
  }
  return { vpcId, subnetIds };
}

export class DeciderEndpointStack extends Stack {
  constructor(scope: Construct, id: string, props: DeciderEndpointStackProps) {
    super(scope, id, props);
    const endpointName = props.endpointName;
    const ctx = (key: string) => this.node.tryGetContext(key);
    const modelId: string = ctx("model");
    const instanceType: string = ctx("instance_type");
    const device = deviceFor(instanceType);

    // The model, copied from the Hugging Face Hub into a private bucket during `cdk deploy`.
    // SageMaker copies it from there to /opt/ml/model on each instance (ModelDataSource below).
    const bucket = new s3.Bucket(this, "Models", {
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      encryption: s3.BucketEncryption.S3_MANAGED,
      enforceSSL: true,
      removalPolicy: RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
    });
    // A CodeBuild job during `cdk deploy` (deploy-time-build's NodejsBuild) runs
    // scripts/fetch-model.ts, which writes the model and its base model to model/, and
    // syncs that folder to s3://<bucket>/<endpoint>/<model_revision>/. It runs again when the model
    // or its revision changes; the revision is in the path, so the SageMaker model changes with it.
    const modelPrefix = `${endpointName}/${ctx("model_revision")}/`;
    const modelData = new NodejsBuild(this, "ModelData", {
      assets: [{ path: path.join(__dirname, "..", "scripts"), extractPath: "fetch" }],
      nodejsVersion: 22,
      computeType: codebuild.ComputeType.LARGE, // 128 GB of disk for the largest models
      buildEnvironment: { MODEL: modelId, MODEL_REVISION: ctx("model_revision") },
      buildCommands: [
        `npm install --no-save --no-audit --no-fund ${FETCH_DEPENDENCIES.join(" ")}`,
        'npx tsx fetch-model.ts "$MODEL" "$MODEL_REVISION" model',
      ],
      outputSourceDirectory: "model",
      destinationBucket: bucket,
      destinationKeyPrefix: modelPrefix,
    });

    const role = new iam.Role(this, "ExecutionRole", {
      assumedBy: new iam.ServicePrincipal("sagemaker.amazonaws.com"),
    });
    role.addToPolicy(
      new iam.PolicyStatement({
        actions: ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
        resources: [`arn:aws:logs:${this.region}:${this.account}:log-group:/aws/sagemaker/*`],
      }),
    );
    role.addToPolicy(new iam.PolicyStatement({ actions: ["cloudwatch:PutMetricData"], resources: ["*"] }));
    bucket.grantRead(role);

    // Builds the image in AWS CodeBuild during `cdk deploy` (deploy-time-build) and publishes it
    // to an ECR repository that belongs to the stack. The image holds no model weights.
    // Neuron has its own image (Dockerfile.neuron): torch-neuronx needs a newer glibc and Python
    // than the CUDA image's.
    const image = new ContainerImageBuild(this, "Image", {
      directory: REPO_ROOT,
      file: device === "neuron" ? "deploy/sagemaker/Dockerfile.neuron" : "deploy/sagemaker/Dockerfile",
      exclude: ASSET_EXCLUDE,
      ignoreMode: IgnoreMode.DOCKER,
      buildArgs: { DEVICE: device },
      platform: ecrAssets.Platform.LINUX_AMD64,
    });
    image.repository.grantPull(role);

    // On Neuron the torso runs as graphs compiled ahead of time, by the same image. They are
    // compiled once, on a temporary instance, into a copy of the model data (neuron-compile.ts).
    const checkpoint = modelId.split("/").pop()!;
    let servedPrefix = modelPrefix;
    let compile: NeuronCompile | undefined;
    if (device === "neuron") {
      servedPrefix = `${modelPrefix.replace(/\/$/, "")}-neuron-${image.imageTag}/`;
      compile = new NeuronCompile(this, "NeuronCompile", {
        bucket,
        sourcePrefix: modelPrefix,
        targetPrefix: servedPrefix,
        repository: image.repository,
        imageUri: image.imageUri,
        checkpoint,
        instanceType: instanceType.replace(/^ml\./, ""),
        network: compileNetwork(ctx("compile_vpc"), ctx("compile_subnets")),
      });
      compile.node.addDependency(modelData);
    }
    const serveArgs = [device === "neuron" ? "--neuron-cache /opt/ml/model/neuron-cache" : "", ctx("serve_args") ?? ""]
      .filter(Boolean)
      .join(" ");

    const model = new sagemaker.CfnModel(this, "Model", {
      executionRoleArn: role.roleArn,
      primaryContainer: {
        image: image.imageUri,
        modelDataSource: {
          s3DataSource: {
            s3Uri: bucket.s3UrlForObject(servedPrefix),
            s3DataType: "S3Prefix",
            compressionType: "None",
          },
        },
        environment: {
          // scripts/fetch-model.ts's layout: /opt/ml/model/<model name>/ and /opt/ml/model/base/.
          DECIDER_CHECKPOINT: `/opt/ml/model/${checkpoint}`,
          DECIDER_DEVICE: device,
          DECIDER_ARGS: serveArgs,
        },
      },
    });
    // The role's policy must exist before SageMaker checks that it can pull the image, and the
    // model must be in S3 before SageMaker copies it.
    model.node.addDependency(role, modelData, ...(compile ? [compile] : []));

    const config = new sagemaker.CfnEndpointConfig(this, "EndpointConfig", {
      productionVariants: [
        {
          variantName: "AllTraffic",
          modelName: model.attrModelName,
          instanceType,
          initialInstanceCount: Number(ctx("instance_count")),
          initialVariantWeight: 1,
          // GPU hosts run Amazon Linux 2023 (NVIDIA driver 580, which runs the image's CUDA 12.6).
          // SageMaker has no AL2023 option for CPU hosts, so they keep its default AMI. The
          // `inference_ami_version` context value overrides both, for GPU families outside the
          // AL2023 AMI's compatibility list (ml.p2, ml.p3, ml.p6e-gb200).
          inferenceAmiVersion: ctx("inference_ami_version") || (device === "cuda" ? GPU_AMI_VERSION : undefined),
          volumeSizeInGb: hasInstanceStorage(instanceType) ? undefined : MODEL_VOLUME_GB,
          // Copying up to about 52 GB from S3, then loading it (about 100 GB of float32 weights
          // on CPU for the largest model), takes minutes. On Neuron, loading every compiled graph
          // takes longer.
          modelDataDownloadTimeoutInSeconds: 3600,
          containerStartupHealthCheckTimeoutInSeconds: device === "neuron" ? 3600 : 1800,
        },
      ],
    });

    const endpoint = new sagemaker.CfnEndpoint(this, "Endpoint", {
      endpointName,
      endpointConfigName: config.attrEndpointConfigName,
    });

    new CfnOutput(this, "EndpointName", { value: endpoint.attrEndpointName });
    new CfnOutput(this, "Device", { value: device });
    new CfnOutput(this, "ModelDataS3Uri", { value: bucket.s3UrlForObject(servedPrefix) });
    new CfnOutput(this, "InvokeUrl", {
      value: `https://runtime.sagemaker.${this.region}.amazonaws.com/endpoints/${endpointName}/invocations`,
    });
  }
}
