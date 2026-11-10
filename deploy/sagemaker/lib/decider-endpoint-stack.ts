import { readFileSync } from "fs";
import * as path from "path";
import { CfnMapping, CfnOutput, IgnoreMode, RemovalPolicy, Stack, StackProps, Token } from "aws-cdk-lib";
import * as codebuild from "aws-cdk-lib/aws-codebuild";
import * as ecrAssets from "aws-cdk-lib/aws-ecr-assets";
import * as iam from "aws-cdk-lib/aws-iam";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as sagemaker from "aws-cdk-lib/aws-sagemaker";
import { Construct } from "constructs";
import { ContainerImageBuild, NodejsBuild } from "@cdklabs/deploy-time-build";

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

/**
 * The AWS Deep Learning Container for strands-decider (GPU): published in every region below, rebuilt
 * weekly with security patches. `-v1` moves with every compatible release; pin `-v1.0.0` (or later) to
 * keep one image. https://gallery.ecr.aws/deep-learning-containers/strands-decider
 */
const DLC_REPOSITORY = "strands-decider";
const DLC_DEFAULT_TAG = "serve-sagemaker-cuda-v1";

/**
 * The Deep Learning Containers registry account in each region, from
 * https://aws.github.io/deep-learning-containers/reference/region_availability/
 */
const DLC_ACCOUNTS: Record<string, string> = {
  "af-south-1": "626614931356",
  "ap-east-1": "871362719292",
  "ap-east-2": "975050140332",
  "ap-northeast-1": "763104351884",
  "ap-northeast-2": "763104351884",
  "ap-northeast-3": "364406365360",
  "ap-south-1": "763104351884",
  "ap-south-2": "772153158452",
  "ap-southeast-1": "763104351884",
  "ap-southeast-2": "763104351884",
  "ap-southeast-3": "907027046896",
  "ap-southeast-4": "457447274322",
  "ap-southeast-5": "550225433462",
  "ap-southeast-6": "633930458069",
  "ap-southeast-7": "590183813437",
  "ca-central-1": "763104351884",
  "ca-west-1": "204538143572",
  "cn-north-1": "727897471807",
  "cn-northwest-1": "727897471807",
  "eu-central-1": "763104351884",
  "eu-central-2": "380420809688",
  "eu-north-1": "763104351884",
  "eu-south-1": "692866216735",
  "eu-south-2": "503227376785",
  "eu-west-1": "763104351884",
  "eu-west-2": "763104351884",
  "eu-west-3": "763104351884",
  "il-central-1": "780543022126",
  "me-central-1": "914824155844",
  "me-south-1": "217643126080",
  "me-west-1": "124382933778",
  "mx-central-1": "637423239942",
  "sa-east-1": "763104351884",
  "sa-west-1": "434457397001",
  "us-east-1": "763104351884",
  "us-east-2": "763104351884",
  "us-west-1": "763104351884",
  "us-west-2": "763104351884",
};

/** Where the endpoint's image comes from: the published DLC (GPU), or a build of ./Dockerfile. */
export type ImageSource = "dlc" | "build";

/**
 * GPU families the DLC runs on: its CUDA kernels are built for compute capability 8.0 to 9.0
 * (A10G, A100, L4, L40S, H100, H200). Older GPUs (ml.g4dn's T4, ml.p3's V100) and newer ones use
 * the Dockerfile.
 */
const DLC_GPU_FAMILIES = /^(g5|g6|g6e|p4d|p4de|p5|p5e|p5en)$/;

function dlcRunsOn(instanceType: string): boolean {
  return DLC_GPU_FAMILIES.test(instanceType.replace(/^ml\./, "").split(".")[0]);
}

const DLC_FAMILIES_TEXT = "ml.g5, ml.g6, ml.g6e, ml.p4d, ml.p4de, ml.p5, ml.p5e, ml.p5en";

/**
 * `dlc` when every instance type the endpoint may use is one the DLC supports, `build` (the
 * Dockerfile) otherwise, unless set.
 */
export function imageSourceFor(setting: string | undefined, instanceTypes: string[]): ImageSource {
  const source = setting || (instanceTypes.every(dlcRunsOn) ? "dlc" : "build");
  if (source !== "dlc" && source !== "build") {
    throw new Error(`image=${source}: use "dlc" (the published GPU image) or "build" (build ./Dockerfile)`);
  }
  const unsupported = instanceTypes.filter((t) => !dlcRunsOn(t));
  if (source === "dlc" && unsupported.length > 0) {
    throw new Error(
      `image=dlc: the Deep Learning Container runs on ${DLC_FAMILIES_TEXT}; ` +
        `use -c image=build for ${unsupported.join(", ")}`,
    );
  }
  return source;
}

/**
 * The instance types to try, in order: `instance_type`, then `fallback_instance_types` (a comma
 * list, up to 4), used when SageMaker has no capacity for the first. They must agree on the device,
 * the image and whether they have instance storage, since the endpoint config holds one of each.
 */
export function instancePoolsFor(instanceType: string, fallbacks: unknown): string[] {
  const list = Array.isArray(fallbacks) ? fallbacks.map(String) : String(fallbacks ?? "").split(",");
  const pools = [instanceType, ...list.map((t) => t.trim()).filter(Boolean)];
  for (const t of pools) {
    if (!/^ml\.[a-z0-9-]+\.[a-z0-9]+$/.test(t)) {
      throw new Error(`${t}: not a SageMaker instance type such as ml.g6.xlarge`);
    }
  }
  if (pools.length > 5) {
    throw new Error(`fallback_instance_types: at most 4 (SageMaker allows 5 instance pools), got ${pools.length - 1}`);
  }
  if (new Set(pools).size !== pools.length) {
    throw new Error(`fallback_instance_types: the instance types must be distinct: ${pools.join(", ")}`);
  }
  for (const t of pools.slice(1)) {
    if (deviceFor(t) !== deviceFor(instanceType)) {
      throw new Error(`fallback_instance_types: ${t} runs on ${deviceFor(t)}, ${instanceType} on ${deviceFor(instanceType)}`);
    }
    if (hasInstanceStorage(t) !== hasInstanceStorage(instanceType)) {
      throw new Error(`fallback_instance_types: ${t} and ${instanceType} differ in instance storage`);
    }
  }
  return pools;
}

/**
 * `max_concurrency`: a whole number, 0 for no cap. Empty means 1 on GPU (a burst of large
 * requests can otherwise run the GPU out of memory) and no cap on CPU, as before.
 */
export function maxConcurrencyFor(setting: unknown, device: "cuda" | "cpu"): number {
  const text = String(setting ?? "").trim();
  if (text === "") return device === "cuda" ? 1 : 0;
  if (!/^\d+$/.test(text)) {
    throw new Error(`max_concurrency=${setting}: a whole number of requests, or 0 for no cap`);
  }
  return Number(text);
}

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
  "**/__pycache__",
  "**/*.egg-info",
];

/**
 * The image's device for a SageMaker instance type: `cuda` on NVIDIA GPU families
 * (ml.g4dn, ml.g5, ml.g6, ml.p4d, ...), `cpu` on x86 CPU families. The image is x86 and has
 * no Neuron support, so Graviton (ml.c7g, ml.m6gd, ...) and Inferentia or Trainium types fail.
 */
export function deviceFor(instanceType: string): "cuda" | "cpu" {
  const family = instanceType.replace(/^ml\./, "").split(".")[0];
  if (/^(inf|trn)\d/.test(family)) {
    throw new Error(`${instanceType}: Inferentia and Trainium instances are not supported`);
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
        resources: [`arn:${this.partition}:logs:${this.region}:${this.account}:log-group:/aws/sagemaker/*`],
      }),
    );
    role.addToPolicy(new iam.PolicyStatement({ actions: ["cloudwatch:PutMetricData"], resources: ["*"] }));
    bucket.grantRead(role);

    const instancePools = instancePoolsFor(instanceType, ctx("fallback_instance_types"));
    const imageSource = imageSourceFor(ctx("image"), instancePools);
    const imageUri = imageSource === "dlc" ? this.dlcImage(role, ctx("dlc_image_tag") || DLC_DEFAULT_TAG) : this.buildImage(role, device);

    // Serving options, as the image takes them: the DLC maps DECIDER_* variables to `serve`
    // options and splits DECIDER_ARGS like a shell command line (quotes group); the Dockerfile's
    // image takes DECIDER_ARGS only, split on spaces (no quoting).
    const maxConcurrency = String(maxConcurrencyFor(ctx("max_concurrency"), device));
    const deviceMap: string = ctx("device_map") ?? "";
    if (deviceMap && device !== "cuda") {
      throw new Error(`device_map=${deviceMap}: splits a model across GPUs; ${instanceType} has none`);
    }
    const serveArgs: string = ctx("serve_args") ?? "";
    const servingEnvironment: Record<string, string> =
      imageSource === "dlc"
        ? {
            DECIDER_MAX_CONCURRENCY: maxConcurrency,
            ...(deviceMap ? { DECIDER_DEVICE_MAP: deviceMap } : {}),
            DECIDER_ARGS: serveArgs,
          }
        : {
            DECIDER_ARGS: [
              maxConcurrency !== "0" ? `--max-concurrency ${maxConcurrency}` : "",
              deviceMap ? `--device-map ${deviceMap}` : "",
              serveArgs,
            ].filter(Boolean).join(" "),
          };

    const model = new sagemaker.CfnModel(this, "Model", {
      executionRoleArn: role.roleArn,
      primaryContainer: {
        image: imageUri,
        modelDataSource: {
          s3DataSource: {
            s3Uri: bucket.s3UrlForObject(modelPrefix),
            s3DataType: "S3Prefix",
            compressionType: "None",
          },
        },
        environment: {
          // scripts/fetch-model.ts's layout: /opt/ml/model/<model name>/ and /opt/ml/model/base/.
          DECIDER_CHECKPOINT: `/opt/ml/model/${modelId.split("/").pop()}`,
          DECIDER_DEVICE: device,
          ...servingEnvironment,
        },
      },
    });
    // The role's policy must exist before SageMaker checks that it can pull the image, and the
    // model must be in S3 before SageMaker copies it.
    model.node.addDependency(role, modelData);

    const config = new sagemaker.CfnEndpointConfig(this, "EndpointConfig", {
      productionVariants: [
        {
          variantName: "AllTraffic",
          modelName: model.attrModelName,
          // With fallbacks, SageMaker tries the instance types in order (instance pools).
          ...(instancePools.length > 1
            ? { instancePools: instancePools.map((t, i) => ({ instanceType: t, priority: i + 1 })) }
            : { instanceType }),
          initialInstanceCount: Number(ctx("instance_count")),
          initialVariantWeight: 1,
          // GPU hosts run Amazon Linux 2023 (NVIDIA driver 580, which runs the DLC's CUDA 12.8 and
          // the Dockerfile image's CUDA 12.6).
          // SageMaker has no AL2023 option for CPU hosts, so they keep its default AMI. The
          // `inference_ami_version` context value overrides both, for GPU families outside the
          // AL2023 AMI's compatibility list (ml.p2, ml.p3, ml.p6e-gb200).
          inferenceAmiVersion: ctx("inference_ami_version") || (device === "cuda" ? GPU_AMI_VERSION : undefined),
          volumeSizeInGb: hasInstanceStorage(instanceType) ? undefined : MODEL_VOLUME_GB,
          // Copying up to about 52 GB from S3, then loading it (about 100 GB of float32 weights
          // on CPU for the largest model), takes minutes.
          modelDataDownloadTimeoutInSeconds: 3600,
          containerStartupHealthCheckTimeoutInSeconds: 1800,
        },
      ],
    });

    const endpoint = new sagemaker.CfnEndpoint(this, "Endpoint", {
      endpointName,
      endpointConfigName: config.attrEndpointConfigName,
    });

    new CfnOutput(this, "EndpointName", { value: endpoint.attrEndpointName });
    new CfnOutput(this, "Device", { value: device });
    new CfnOutput(this, "ImageUri", { value: imageUri });
    new CfnOutput(this, "ModelDataS3Uri", { value: bucket.s3UrlForObject(modelPrefix) });
    new CfnOutput(this, "InvokeUrl", {
      value: `https://runtime.sagemaker.${this.region}.${this.urlSuffix}/endpoints/${endpointName}/invocations`,
    });
  }

  /** The published Deep Learning Container in this stack's region, pullable by `role`. */
  private dlcImage(role: iam.IRole, tag: string): string {
    const region = this.region;
    let account: string;
    if (Token.isUnresolved(region)) {
      const mapping = new CfnMapping(this, "DlcAccounts", {
        mapping: Object.fromEntries(Object.entries(DLC_ACCOUNTS).map(([r, a]) => [r, { account: a }])),
      });
      account = mapping.findInMap(region, "account");
    } else {
      const known = DLC_ACCOUNTS[region];
      if (!known) {
        throw new Error(`${region}: no Deep Learning Containers registry; use -c image=build, or a region from DLC_ACCOUNTS`);
      }
      account = known;
    }
    role.addToPrincipalPolicy(
      new iam.PolicyStatement({
        actions: ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer", "ecr:BatchCheckLayerAvailability"],
        resources: [`arn:${this.partition}:ecr:${region}:${account}:repository/${DLC_REPOSITORY}`],
      }),
    );
    role.addToPrincipalPolicy(new iam.PolicyStatement({ actions: ["ecr:GetAuthorizationToken"], resources: ["*"] }));
    return `${account}.dkr.ecr.${region}.${this.urlSuffix}/${DLC_REPOSITORY}:${tag}`;
  }

  /**
   * Builds ./Dockerfile in AWS CodeBuild during `cdk deploy` (deploy-time-build) and publishes it to
   * an ECR repository that belongs to the stack. The image holds no model weights.
   */
  private buildImage(role: iam.IRole, device: "cuda" | "cpu"): string {
    const image = new ContainerImageBuild(this, "Image", {
      directory: REPO_ROOT,
      file: "deploy/sagemaker/Dockerfile",
      exclude: ASSET_EXCLUDE,
      ignoreMode: IgnoreMode.DOCKER,
      buildArgs: { DEVICE: device },
      platform: ecrAssets.Platform.LINUX_AMD64,
    });
    image.repository.grantPull(role);
    return image.imageUri;
  }

}
