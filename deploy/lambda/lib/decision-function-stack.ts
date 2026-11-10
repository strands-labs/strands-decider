import * as path from "path";
import {
  Annotations,
  CfnOutput,
  CustomResource,
  Duration,
  FileSystem,
  IgnoreMode,
  RemovalPolicy,
  Stack,
  StackProps,
} from "aws-cdk-lib";
import * as codebuild from "aws-cdk-lib/aws-codebuild";
import * as ecrAssets from "aws-cdk-lib/aws-ecr-assets";
import * as iam from "aws-cdk-lib/aws-iam";
import * as lambda from "aws-cdk-lib/aws-lambda";
import * as logs from "aws-cdk-lib/aws-logs";
import { Construct } from "constructs";
import { ContainerImageBuild } from "@cdklabs/deploy-time-build";

const REPO_ROOT = path.join(__dirname, "..", "..", "..");

// The build uploads this directory to S3; keep only what the image needs, as
// decision/Dockerfile.dockerignore does for a plain docker build.
const ASSET_EXCLUDE = [
  "*",
  "!pyproject.toml",
  "!README.md",
  "!LICENSE",
  "!src/",
  "!deploy/lambda/decision/",
  "**/__pycache__",
  "**/*.egg-info",
];

/**
 * The most tokens a request may forward by default, the state once plus every question. The
 * slowest request under it, 4 images and an 8,000-token state with the shortest questions after it,
 * takes about 640 s on Graviton2, inside the 15-minute timeout; at 24,576 it does not finish
 * (README.md, Decision checkpoints, Limits).
 */
export const DECISION_MAX_REQUEST_TOKENS = 16384;

/** A true or false context value, from cdk.json or `-c key=value`. */
function flag(value: unknown, key: string): boolean {
  if (String(value) !== "true" && String(value) !== "false") {
    throw new Error(`${key} must be true or false, not ${JSON.stringify(value)}`);
  }
  return String(value) === "true";
}

/**
 * A decision checkpoint (docs/spec.md) as a Lambda function (README.md, Decision checkpoints): the
 * image, built from the checkpoint with its decoder and vision tower converted for llama.cpp, the
 * function, its `live` alias and that alias's IAM-authenticated Function URL. Settings are CDK
 * context values (cdk.json, or -c key=value): `checkpoint`, a Hugging Face Hub id or an s3:// URI
 * of a checkpoint directory; `checkpoint_revision`; `snapstart`; and `max_request_tokens`, the most
 * tokens one request may forward (DECISION_MAX_REQUEST_TOKENS by default).
 */
export class DecisionFunctionStack extends Stack {
  constructor(scope: Construct, id: string, props?: StackProps) {
    super(scope, id, props);
    const ctx = (key: string) => this.node.tryGetContext(key);
    const checkpoint = String(ctx("checkpoint") ?? "");
    if (!checkpoint) throw new Error("set the decision checkpoint: -c checkpoint=<Hub id or s3://bucket/prefix/>");
    const snapStart = flag(ctx("snapstart") ?? false, "snapstart");
    const maxRequestTokens = Number(ctx("max_request_tokens") ?? DECISION_MAX_REQUEST_TOKENS);
    if (!Number.isInteger(maxRequestTokens) || maxRequestTokens < 0) {
      throw new Error(`max_request_tokens must be a whole number, 0 for no limit, not ${ctx("max_request_tokens")}`);
    }

    // Builds the image in AWS CodeBuild during `cdk deploy` (deploy-time-build) and publishes it to
    // an ECR repository that belongs to the stack. The build downloads the checkpoint and converts
    // it with llama.cpp's own converter, so the image holds the model.
    const buildArgs = { CHECKPOINT: checkpoint, CHECKPOINT_REVISION: String(ctx("checkpoint_revision") ?? "main") };
    const image = new ContainerImageBuild(this, "Image", {
      directory: REPO_ROOT,
      file: "deploy/lambda/decision/Dockerfile",
      exclude: ASSET_EXCLUDE,
      ignoreMode: IgnoreMode.DOCKER,
      buildArgs,
      // An s3:// checkpoint is read with the CodeBuild project's role, through the container
      // credentials endpoint; its path goes in as a BuildKit secret, so no layer or image setting
      // holds it.
      buildSecrets: { aws_container_credentials_uri: "env=AWS_CONTAINER_CREDENTIALS_RELATIVE_URI" },
      platform: ecrAssets.Platform.LINUX_ARM64,
    });
    const s3 = /^s3:\/\/([^/]+)\/?(.*)$/.exec(checkpoint);
    if (s3) {
      // Read-only access to the checkpoint's own prefix; nothing changes on the bucket.
      const [bucket, prefix] = [s3[1], s3[2].replace(/\/?$/, "/").replace(/^\/$/, "")];
      image.grantPrincipal.addToPrincipalPolicy(
        new iam.PolicyStatement({ actions: ["s3:GetObject"], resources: [`arn:aws:s3:::${bucket}/${prefix}*`] }),
      );
      image.grantPrincipal.addToPrincipalPolicy(
        new iam.PolicyStatement({
          actions: ["s3:ListBucket"],
          resources: [`arn:aws:s3:::${bucket}`],
          conditions: { StringLike: { "s3:prefix": [`${prefix}*`] } },
        }),
      );
    }
    // deploy-time-build builds on a SMALL instance. The conversion holds the checkpoint on disk and
    // its BF16 GGUF in memory (9.3 GB at E2B), so its project gets an XLARGE one, with a name that
    // starts with the stack's, and its log group belongs to the stack, as does the build trigger's,
    // so that `cdk destroy` deletes them.
    const projects = this.node.children.filter(
      (c): c is codebuild.Project => c instanceof codebuild.Project && c.node.id.startsWith("ContainerImageBuildArm64"),
    );
    if (projects.length !== 1) throw new Error(`expected deploy-time-build's arm64 project, found ${projects.length}`);
    const project = projects[0].node.defaultChild as codebuild.CfnProject;
    const projectName = `${this.stackName}-image-build`.slice(0, 150);
    project.addPropertyOverride("Environment.ComputeType", "BUILD_GENERAL1_XLARGE");
    project.addPropertyOverride("Name", projectName);
    const buildLogs = new logs.LogGroup(this, "BuildLogs", {
      logGroupName: `/aws/codebuild/${projectName}`,
      retention: logs.RetentionDays.TWO_WEEKS,
      removalPolicy: RemovalPolicy.DESTROY,
    });
    const triggers = this.node
      .findAll()
      .filter(
        (c): c is lambda.Function =>
          c instanceof lambda.Function && c.node.id.startsWith("DeployTimeBuildCustomResourceHandler"),
      );
    if (triggers.length !== 1) throw new Error(`expected deploy-time-build's trigger function, found ${triggers.length}`);
    const triggerLogs = new logs.LogGroup(this, "TriggerLogs", {
      logGroupName: `/aws/lambda/${triggers[0].functionName}`,
      retention: logs.RetentionDays.TWO_WEEKS,
      removalPolicy: RemovalPolicy.DESTROY,
    });
    (image.node.findChild("Resource") as CustomResource).node.addDependency(buildLogs, triggerLogs);

    const logGroup = new logs.LogGroup(this, "Logs", {
      retention: logs.RetentionDays.TWO_WEEKS,
      removalPolicy: RemovalPolicy.DESTROY,
    });
    const fn = new lambda.DockerImageFunction(this, "Function", {
      code: image.toLambdaDockerImageCode(),
      architecture: lambda.Architecture.ARM_64,
      // The most Lambda gives, and the only size with 6 vCPUs.
      memorySize: 10240,
      timeout: Duration.minutes(15),
      snapStart: snapStart ? lambda.SnapStartConf.ON_PUBLISHED_VERSIONS : undefined,
      logGroup,
      environment: { DECIDER_MAX_REQUEST_TOKENS: String(maxRequestTokens) },
      // Each deploy that changes the function publishes a version, and CloudFormation deletes the
      // one it replaces: Lambda bills a version's SnapStart snapshot for as long as it exists.
      currentVersionOptions: { removalPolicy: RemovalPolicy.DESTROY },
    });
    // The image is built during the deploy, so its tag is not known when CDK hashes the function's
    // settings; hashing what the build reads publishes a new version whenever the image can differ.
    fn.invalidateVersionBasedOn(
      FileSystem.fingerprint(REPO_ROOT, { exclude: ASSET_EXCLUDE, ignoreMode: IgnoreMode.DOCKER }) +
        JSON.stringify(buildArgs),
    );
    Annotations.of(fn).acknowledgeWarning("@aws-cdk/aws-lambda:snapStartRequirePublish", "the alias below publishes the version");
    const alias = new lambda.Alias(this, "Live", { aliasName: "live", version: fn.currentVersion });
    const url = alias.addFunctionUrl({ authType: lambda.FunctionUrlAuthType.AWS_IAM });

    new CfnOutput(this, "FunctionUrl", { value: url.url });
    new CfnOutput(this, "FunctionArn", { value: alias.functionArn });
    new CfnOutput(this, "LogGroup", { value: logGroup.logGroupName });
    new CfnOutput(this, "BuildLogGroup", { value: buildLogs.logGroupName });
    new CfnOutput(this, "SnapStart", { value: String(snapStart) });
  }
}
