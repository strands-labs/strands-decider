import * as path from "path";
import { Annotations, CfnOutput, Duration, FileSystem, IgnoreMode, RemovalPolicy, Size, Stack, StackProps } from "aws-cdk-lib";
import * as codebuild from "aws-cdk-lib/aws-codebuild";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as ecrAssets from "aws-cdk-lib/aws-ecr-assets";
import * as iam from "aws-cdk-lib/aws-iam";
import * as lambda from "aws-cdk-lib/aws-lambda";
import * as logs from "aws-cdk-lib/aws-logs";
import { Construct } from "constructs";
import { ContainerImageBuild } from "@cdklabs/deploy-time-build";

/**
 * The stack name for a model's Hugging Face Hub id: its name, lowercased, with every character
 * other than a letter, digit or hyphen replaced by a hyphen, and `-lambda-gpu` after it.
 */
export function stackNameFor(model: string): string {
  const name = (model.split("/").pop() ?? "")
    .toLowerCase()
    .replace(/[^a-z0-9-]+/g, "-")
    .replace(/^-+|-+$/g, "");
  if (!/^[a-z][a-z0-9-]{0,116}$/.test(name)) {
    throw new Error(`${model}: cannot make a CloudFormation stack name (a letter, then letters, digits, hyphens) from it`);
  }
  return `${name}-lambda-gpu`;
}

/** AcceleratorConfig.AcceleratorMemorySize valid values, in GB (per the preview docs). */
const VALID_ACCELERATOR_MEMORY_GB = [3, 6, 12, 16, 24, 48];

/** GPU EC2 families Lambda Managed Instances supports (preview docs). `g4dn.metal` is unsupported. */
const SUPPORTED_GPU_FAMILIES = ["g4dn", "g5", "g6", "gr6", "g6e", "g6f", "gr6f"];

function parseInstanceTypes(raw: unknown): ec2.InstanceType[] {
  const list = Array.isArray(raw) ? raw.map(String) : String(raw ?? "").split(",");
  const types = list.map((t) => t.trim()).filter(Boolean);
  if (types.length === 0) {
    throw new Error("instance_types: give one or more GPU EC2 instance types (comma list)");
  }
  for (const t of types) {
    if (!/^[a-z0-9]+\.[a-z0-9]+$/.test(t)) {
      throw new Error(`${t}: not an EC2 instance type such as g6.xlarge (no ml. prefix)`);
    }
    if (t === "g4dn.metal") {
      throw new Error(`g4dn.metal: not supported by Lambda Managed Instances`);
    }
    const family = t.split(".")[0];
    if (!SUPPORTED_GPU_FAMILIES.includes(family)) {
      throw new Error(
        `${t}: Lambda Managed Instances GPU supports ${SUPPORTED_GPU_FAMILIES.join(", ")}; not ${family}`,
      );
    }
  }
  return types.map((t) => new ec2.InstanceType(t));
}

function wholeNumber(value: unknown, key: string, min = 0): number {
  const n = Number(value);
  if (!Number.isInteger(n) || n < min) {
    throw new Error(`${key} must be a whole number >= ${min}, not ${JSON.stringify(value)}`);
  }
  return n;
}

// Tell CDK's version-hash scanner about the LMI GPU accelerator property, which aws-cdk-lib 2.273
// does not know yet. `true` locks the version: a change to AcceleratorConfig publishes a new one.
lambda.Function.classifyVersionProperty("AcceleratorConfig", true);

const REPO_ROOT = path.join(__dirname, "..", "..", "..");

const ASSET_EXCLUDE = [
  "*",
  "!pyproject.toml",
  "!README.md",
  "!LICENSE",
  "!src/",
  "!deploy/lambda-gpu/Dockerfile",
  "!deploy/lambda-gpu/function/",
  "!deploy/lambda-gpu/scripts/",
  "**/__pycache__",
  "**/*.egg-info",
];

/**
 * The strands-decider GPU-accelerated Lambda Event Function (README.md): a VPC, a Lambda capacity
 * provider over the configured GPU instance types, the image built with the model in it, the
 * function attached to the capacity provider with AcceleratorConfig, and its `live` alias.
 * Settings are CDK context values (cdk.json, or `-c key=value`).
 *
 * Preview: Lambda Managed Instances GPU is a preview feature. The capacity provider and
 * CapacityProviderConfig are shipped in aws-cdk-lib 2.273; `AcceleratorConfig` is set with a raw
 * CfnFunction property override, because it is not yet in the L2 construct or the shipped CFN
 * schema. The deploying account must have access to the preview GPU resource type, or
 * `cdk deploy` fails at create time.
 */
export class DeciderGpuFunctionStack extends Stack {
  constructor(scope: Construct, id: string, props?: StackProps) {
    super(scope, id, props);
    const ctx = (key: string) => this.node.tryGetContext(key);

    const instanceTypes = parseInstanceTypes(ctx("instance_types"));
    const acceleratorMemoryGb = wholeNumber(ctx("accelerator_memory_gb"), "accelerator_memory_gb", 3);
    if (!VALID_ACCELERATOR_MEMORY_GB.includes(acceleratorMemoryGb)) {
      throw new Error(
        `accelerator_memory_gb must be one of ${VALID_ACCELERATOR_MEMORY_GB.join(", ")}; got ${acceleratorMemoryGb}`,
      );
    }
    const maxVcpus = wholeNumber(ctx("max_vcpus"), "max_vcpus", 0);
    const perEnvMaxConcurrency = wholeNumber(ctx("per_env_max_concurrency"), "per_env_max_concurrency", 1);
    if (perEnvMaxConcurrency > 48) {
      throw new Error(`per_env_max_concurrency must be at most 48; got ${perEnvMaxConcurrency}`);
    }
    const memorySizeMb = wholeNumber(ctx("memory_size_mb"), "memory_size_mb", 2048);
    if (memorySizeMb > 32768) {
      throw new Error(`memory_size_mb must be <= 32768; got ${memorySizeMb}`);
    }
    const ephemeralStorageMb = wholeNumber(ctx("ephemeral_storage_mb"), "ephemeral_storage_mb", 512);
    if (ephemeralStorageMb > 32768) {
      throw new Error(`ephemeral_storage_mb must be <= 32768; got ${ephemeralStorageMb}`);
    }
    const serverMaxConcurrency = ctx("server_max_concurrency");
    const serverMaxConcurrencyInt =
      serverMaxConcurrency === undefined || serverMaxConcurrency === null || serverMaxConcurrency === ""
        ? undefined
        : wholeNumber(serverMaxConcurrency, "server_max_concurrency", 1);

    // VPC: the capacity provider launches EC2 instances into subnets you give it. A simple
    // two-AZ VPC with public subnets is enough for a scaffold — public instances reach ECR and
    // the Hugging Face Hub directly, with no NAT gateway. Use `-c vpc_id=` and `-c subnet_ids=`
    // to point the capacity provider at an existing VPC instead (not yet wired up in this
    // scaffold; adapt in place if needed).
    // GPU capacity in a single AZ runs out often. In us-west-2, us-west-2a was out of g6.xlarge
    // during multiple deploy attempts. Pass `-c availability_zones=us-west-2b,us-west-2c,us-west-2d`
    // to skip a problem AZ; otherwise CDK picks the first `maxAzs` in alphabetical order.
    const azsOverride = ctx("availability_zones");
    const availabilityZones = azsOverride
      ? String(azsOverride).split(",").map((s) => s.trim()).filter(Boolean)
      : undefined;
    const vpc = new ec2.Vpc(this, "Vpc", {
      maxAzs: availabilityZones ? undefined : 4,
      availabilityZones,
      natGateways: 0,
      subnetConfiguration: [
        { name: "Public", subnetType: ec2.SubnetType.PUBLIC, cidrMask: 24 },
      ],
    });

    const capacityProviderSg = new ec2.SecurityGroup(this, "CapacityProviderSg", {
      vpc,
      description: "Lambda Managed Instances compute for the strands-decider GPU function",
      allowAllOutbound: true,
    });

    // Build the image in CodeBuild during `cdk deploy` (deploy-time-build) and publish it to an
    // ECR repository that belongs to the stack. The build bakes the strands-decider checkpoint
    // into the image (up to 32 GB), so a new execution environment loads from local disk.
    const buildArgs = {
      MODEL: ctx("model"),
      MODEL_REVISION: ctx("model_revision"),
    };
    const image = new ContainerImageBuild(this, "Image", {
      directory: REPO_ROOT,
      file: "deploy/lambda-gpu/Dockerfile",
      exclude: ASSET_EXCLUDE,
      ignoreMode: IgnoreMode.DOCKER,
      buildArgs,
      platform: ecrAssets.Platform.LINUX_AMD64,
    });
    // deploy-time-build builds on a SMALL instance by default. The build downloads several GB of
    // model weights and installs torch+CUDA (about 10 GB of layers), so give it a LARGE one (16 GB).
    const builds = this.node.children.filter(
      (c): c is codebuild.Project => c instanceof codebuild.Project && c.node.id.startsWith("ContainerImageBuildAmd64"),
    );
    if (builds.length !== 1) throw new Error(`expected deploy-time-build's amd64 project, found ${builds.length}`);
    (builds[0].node.defaultChild as codebuild.CfnProject).addPropertyOverride(
      "Environment.ComputeType",
      codebuild.ComputeType.LARGE,
    );

    const logGroup = new logs.LogGroup(this, "Logs", {
      retention: logs.RetentionDays.TWO_WEEKS,
      // RETAIN so logs survive a stack rollback; the preview service's failures are not always
      // visible in CloudFormation events, and losing the function's CloudWatch logs to rollback
      // makes debugging impossible. Delete the log group by hand after a successful teardown.
      removalPolicy: RemovalPolicy.RETAIN,
    });

    const executionRole = new iam.Role(this, "ExecutionRole", {
      assumedBy: new iam.ServicePrincipal("lambda.amazonaws.com"),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName("service-role/AWSLambdaBasicExecutionRole"),
      ],
    });
    image.repository.grantPull(executionRole);
    logGroup.grantWrite(executionRole);

    // The capacity provider. The L2 construct auto-creates an operator role with
    // AWSLambdaManagedEC2ResourceOperator attached when `operatorRole` is unset.
    const capacityProvider = new lambda.CapacityProvider(this, "CapacityProvider", {
      capacityProviderName: `${this.stackName}-cp`.slice(0, 64),
      vpc: vpc as unknown as ec2.IVpc,
      securityGroups: [capacityProviderSg],
      subnets: vpc.publicSubnets,
      instanceTypeFilter: lambda.InstanceTypeFilter.allow(instanceTypes),
      architectures: [lambda.Architecture.X86_64],
      ...(maxVcpus > 0 ? { maxVCpuCount: maxVcpus } : {}),
    } as any);

    const fn = new lambda.DockerImageFunction(this, "Function", {
      // LMI GPU validates ImageConfig.EntryPoint is an absolute path to an executable; without
      // these, Lambda reads the Docker manifest's ENTRYPOINT ("python") and rejects it as
      // non-absolute. These overrides have to go on the DockerImageCode, not the Function —
      // CDK's DockerImageFunction reads cmd/entrypoint/workingDirectory from the code object's
      // bind() result, so passing them as Function props silently drops them.
      code: image.toLambdaDockerImageCode({
        entrypoint: ["/var/task/bootstrap"],
        cmd: ["handler.handler"],
        workingDirectory: "/var/task",
      }),
      architecture: lambda.Architecture.X86_64,
      memorySize: memorySizeMb,
      timeout: Duration.minutes(15),
      // L2 caps ephemeral storage at 10 GB; LMI GPU allows up to 32 GB — values above 10,240 MB
      // are applied below via a CfnFunction property override on EphemeralStorage.Size.
      ephemeralStorageSize: Size.mebibytes(Math.min(ephemeralStorageMb, 10240)),
      role: executionRole,
      logGroup,
      environment: {
        // `strands-decider serve --max-concurrency N` caps concurrent forwards on the model.
        // Unset means no cap; the server queues on its GPU lock.
        ...(serverMaxConcurrencyInt !== undefined
          ? { DECIDER_MAX_CONCURRENCY: String(serverMaxConcurrencyInt) }
          : {}),
      },
      currentVersionOptions: { removalPolicy: RemovalPolicy.DESTROY },
    });

    // Attach the function to the capacity provider: this sets
    // CapacityProviderConfig.LambdaManagedInstancesCapacityProviderConfig on the CfnFunction.
    // minExecutionEnvironments defaults to 3; a first publish then tries to launch 3 GPU
    // instances at once, which often fails with InsufficientCapacity on GPU families. Keep it
    // at 1 and let traffic scale the function up.
    capacityProvider.addFunction(fn, {
      perExecutionEnvironmentMaxConcurrency: perEnvMaxConcurrency,
      latestPublishedScalingConfig: { minExecutionEnvironments: 1 },
    });

    const cfnFn = fn.node.defaultChild as lambda.CfnFunction;
    // AcceleratorConfig is the GPU preview piece; not yet in the shipped CFN schema, so the
    // override is the way to set it. Acknowledge the "unexpected property" warning that CDK's
    // bundled schema validator emits for the same reason.
    cfnFn.addPropertyOverride("AcceleratorConfig", {
      AcceleratorMemorySize: acceleratorMemoryGb,
    });
    Annotations.of(fn).acknowledgeWarning(
      "CloudFormation-Validate::F3002",
      "AcceleratorConfig is the Lambda Managed Instances GPU preview property, not yet in the shipped CFN schema",
    );
    Annotations.of(fn).acknowledgeInfo("Annotation::aws-lambda:Function.CapacityProviderConfiguration");
    if (ephemeralStorageMb > 10240) {
      cfnFn.addPropertyOverride("EphemeralStorage.Size", ephemeralStorageMb);
    }

    // Publish a new version when the image can differ (the image tag is unknown at synth time,
    // so CDK's own hash would miss image-only changes).
    fn.invalidateVersionBasedOn(
      FileSystem.fingerprint(REPO_ROOT, { exclude: ASSET_EXCLUDE, ignoreMode: IgnoreMode.DOCKER }) +
        JSON.stringify(buildArgs) +
        JSON.stringify({ acceleratorMemoryGb, perEnvMaxConcurrency, memorySizeMb, ephemeralStorageMb, serverMaxConcurrencyInt }),
    );
    Annotations.of(fn).acknowledgeWarning(
      "@aws-cdk/aws-lambda:snapStartRequirePublish",
      "the alias below publishes the version",
    );
    const alias = new lambda.Alias(this, "Live", { aliasName: "live", version: fn.currentVersion });

    new CfnOutput(this, "FunctionArn", { value: alias.functionArn });
    new CfnOutput(this, "CapacityProviderArn", { value: capacityProvider.capacityProviderArn });
    new CfnOutput(this, "LogGroup", { value: logGroup.logGroupName });
    new CfnOutput(this, "InstanceTypes", { value: instanceTypes.map((t) => t.toString()).join(",") });
    new CfnOutput(this, "AcceleratorMemoryGb", { value: String(acceleratorMemoryGb) });
    new CfnOutput(this, "VpcId", { value: vpc.vpcId });
  }
}
