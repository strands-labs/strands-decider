import * as path from "path";
import { Annotations, CfnOutput, Duration, FileSystem, IgnoreMode, RemovalPolicy, Stack, StackProps } from "aws-cdk-lib";
import * as codebuild from "aws-cdk-lib/aws-codebuild";
import * as ecrAssets from "aws-cdk-lib/aws-ecr-assets";
import * as lambda from "aws-cdk-lib/aws-lambda";
import * as logs from "aws-cdk-lib/aws-logs";
import { Construct } from "constructs";
import { ContainerImageBuild } from "@cdklabs/deploy-time-build";

/**
 * The stack name for a model's Hugging Face Hub id: its name, lowercased, with every character
 * other than a letter, digit or hyphen replaced by a hyphen, and `-lambda` after it.
 * StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 -> strands-decider-2b-qwen3-5-v1-2610-lambda.
 */
export function stackNameFor(model: string): string {
  const name = (model.split("/").pop() ?? "")
    .toLowerCase()
    .replace(/[^a-z0-9-]+/g, "-")
    .replace(/^-+|-+$/g, "");
  if (!/^[a-z][a-z0-9-]{0,120}$/.test(name)) {
    throw new Error(`${model}: cannot make a CloudFormation stack name (a letter, then letters, digits, hyphens) from it`);
  }
  return `${name}-lambda`;
}

/** A true or false context value, from cdk.json or `-c key=value`. */
function flag(value: unknown, key: string): boolean {
  if (String(value) !== "true" && String(value) !== "false") {
    throw new Error(`${key} must be true or false, not ${JSON.stringify(value)}`);
  }
  return String(value) === "true";
}

const REPO_ROOT = path.join(__dirname, "..", "..", "..");

// The build uploads this directory to S3; keep only what the image needs, as
// Dockerfile.dockerignore does for a plain docker build.
const ASSET_EXCLUDE = [
  "*",
  "!pyproject.toml",
  "!README.md",
  "!LICENSE",
  "!src/",
  "!deploy/lambda/Dockerfile",
  "!deploy/lambda/function/",
  "!deploy/lambda/scripts/",
  "**/__pycache__",
  "**/*.egg-info",
];

/**
 * The strands-decider Lambda function (README.md): the image, built with the model in it, the
 * function, its `live` alias and that alias's IAM-authenticated Function URL. Settings are CDK
 * context values (cdk.json, or -c key=value).
 */
export class DeciderFunctionStack extends Stack {
  constructor(scope: Construct, id: string, props?: StackProps) {
    super(scope, id, props);
    const ctx = (key: string) => this.node.tryGetContext(key);
    const snapStart = flag(ctx("snapstart"), "snapstart");
    const maxRequestTokens = Number(ctx("max_request_tokens"));
    if (!Number.isInteger(maxRequestTokens) || maxRequestTokens < 0) {
      throw new Error(`max_request_tokens must be a whole number, 0 for no limit, not ${ctx("max_request_tokens")}`);
    }

    // Builds the image in AWS CodeBuild during `cdk deploy` (deploy-time-build) and publishes it
    // to an ECR repository that belongs to the stack. The build downloads the model, folds its
    // LoRA into the torso and converts it, so the image holds the model.
    const buildArgs = { MODEL: ctx("model"), MODEL_REVISION: ctx("model_revision") };
    const image = new ContainerImageBuild(this, "Image", {
      directory: REPO_ROOT,
      file: "deploy/lambda/Dockerfile",
      exclude: ASSET_EXCLUDE,
      ignoreMode: IgnoreMode.DOCKER,
      buildArgs,
      platform: ecrAssets.Platform.LINUX_ARM64,
    });
    // deploy-time-build builds on a SMALL instance, 4 GB on arm64. Folding the LoRA in takes
    // about 8 GB, so its project, which it adds to the stack, gets a LARGE one (16 GB).
    const builds = this.node.children.filter(
      (c): c is codebuild.Project => c instanceof codebuild.Project && c.node.id.startsWith("ContainerImageBuildArm64"),
    );
    if (builds.length !== 1) throw new Error(`expected deploy-time-build's arm64 project, found ${builds.length}`);
    (builds[0].node.defaultChild as codebuild.CfnProject).addPropertyOverride(
      "Environment.ComputeType",
      codebuild.ComputeType.LARGE,
    );

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
      environment: {
        DECIDER_STRICT_WINDOW: String(flag(ctx("strict_window"), "strict_window")),
        DECIDER_MAX_REQUEST_TOKENS: String(maxRequestTokens),
      },
      // Each deploy that changes the function publishes a version, and CloudFormation deletes the
      // one it replaces: Lambda bills a version's SnapStart snapshot for as long as it exists.
      currentVersionOptions: { removalPolicy: RemovalPolicy.DESTROY },
    });
    // The image's tag is known only once it is built, so the version's hash does not change with
    // it: add the build's inputs, and a new image publishes a new version.
    fn.invalidateVersionBasedOn(
      FileSystem.fingerprint(REPO_ROOT, { exclude: ASSET_EXCLUDE, ignoreMode: IgnoreMode.DOCKER }) +
        JSON.stringify(buildArgs),
    );
    Annotations.of(fn).acknowledgeWarning(
      "@aws-cdk/aws-lambda:snapStartRequirePublish",
      "the alias below publishes the version",
    );
    const alias = new lambda.Alias(this, "Live", { aliasName: "live", version: fn.currentVersion });
    const url = alias.addFunctionUrl({ authType: lambda.FunctionUrlAuthType.AWS_IAM });

    new CfnOutput(this, "FunctionUrl", { value: url.url });
    new CfnOutput(this, "FunctionArn", { value: alias.functionArn });
    new CfnOutput(this, "LogGroup", { value: logGroup.logGroupName });
    new CfnOutput(this, "SnapStart", { value: String(snapStart) });
  }
}
