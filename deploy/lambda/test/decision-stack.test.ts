// The decision checkpoint's stack, synthesized: its Function URL's auth, the build role's S3 grant,
// the request limit, and the refusals (README.md, Decision checkpoints). Run: npm test.
import { strict as assert } from "node:assert";
import { test } from "node:test";
import { App } from "aws-cdk-lib";
import { Match, Template } from "aws-cdk-lib/assertions";
import { DeciderFunctionStack } from "../lib/decider-function-stack";
import { DecisionFunctionStack, decisionStackName, parseS3Checkpoint } from "../lib/decision-function-stack";

const ENV = { account: "123456789012", region: "us-east-1" };

function synth(context: Record<string, unknown>): Template {
  const app = new App({ context });
  return Template.fromStack(new DecisionFunctionStack(app, "decision-test", { env: ENV }));
}

const S3 = { checkpoint: "s3://my-bucket/models/e2b/" };

test("the Function URL needs SigV4 and grants no one access", () => {
  const t = synth(S3);
  t.resourceCountIs("AWS::Lambda::Url", 1);
  t.hasResourceProperties("AWS::Lambda::Url", { AuthType: "AWS_IAM" });
  t.resourceCountIs("AWS::Lambda::Permission", 0);
});

test("the build reads only the checkpoint's prefix, in the stack's partition", () => {
  const t = synth(S3);
  const s3 = Object.values(t.findResources("AWS::IAM::Policy"))
    .flatMap((p: any) => p.Properties.PolicyDocument.Statement)
    .filter((s: any) => [s.Action].flat().some((a: string) => a.startsWith("s3:")));
  const checkpoint = s3.filter((s: any) => JSON.stringify(s.Resource).includes("my-bucket"));
  assert.equal(checkpoint.length, 2);
  const get = checkpoint.find((s: any) => s.Action === "s3:GetObject");
  assert.deepEqual(get.Resource, { "Fn::Join": ["", ["arn:", { Ref: "AWS::Partition" }, ":s3:::my-bucket/models/e2b/*"]] });
  const list = checkpoint.find((s: any) => s.Action === "s3:ListBucket");
  assert.deepEqual(list.Condition, { StringLike: { "s3:prefix": ["models/e2b/*"] } });
  // Every other S3 grant is CDK's own, on its asset bucket.
  for (const s of s3.filter((x: any) => !checkpoint.includes(x))) {
    assert.match(JSON.stringify(s.Resource), /cdk-hnb659fds-assets/);
  }
  t.hasResourceProperties("Custom::CDKContainerImageBuild", {
    buildCommand: Match.stringLikeRegexp("--secret id=aws_container_credentials_uri"),
  });
});

test("the build pushes only to the stack's own repository", () => {
  const t = synth(S3);
  const ecr = Object.values(t.findResources("AWS::IAM::Policy"))
    .flatMap((p: any) => p.Properties.PolicyDocument.Statement)
    .filter((s: any) => [s.Action].flat().some((a: string) => a.startsWith("ecr:")));
  const repo = Object.keys(t.findResources("AWS::ECR::Repository"));
  assert.equal(repo.length, 1);
  for (const s of ecr) {
    if (s.Action === "ecr:GetAuthorizationToken") assert.equal(s.Resource, "*"); // ECR allows no other
    else assert.deepEqual(s.Resource, { "Fn::GetAtt": [repo[0], "Arn"] });
  }
});

test("a Hub checkpoint's build gets no credentials and no S3 grant", () => {
  const t = synth({ checkpoint: "StrandsAgents/example-e2b" });
  const build = Object.values(t.findResources("Custom::CDKContainerImageBuild"))[0] as any;
  assert.doesNotMatch(build.Properties.buildCommand, /--secret/);
  assert.doesNotMatch(JSON.stringify(t.findResources("AWS::IAM::Policy")), /StrandsAgents/);
});

test("the request limit is 16,384 by default, and set by max_request_tokens", () => {
  synth(S3).hasResourceProperties("AWS::Lambda::Function", {
    Environment: { Variables: { DECIDER_MAX_REQUEST_TOKENS: "16384" } },
  });
  synth({ ...S3, max_request_tokens: 8192 }).hasResourceProperties("AWS::Lambda::Function", {
    Environment: { Variables: { DECIDER_MAX_REQUEST_TOKENS: "8192" } },
  });
});

test("the checkpoint digest goes to the build", () => {
  const digest = "a".repeat(64);
  synth({ ...S3, checkpoint_digest: digest }).hasResourceProperties("Custom::CDKContainerImageBuild", {
    buildCommand: Match.stringLikeRegexp(`CHECKPOINT_DIGEST=${digest}`),
  });
  assert.throws(() => synth({ ...S3, checkpoint_digest: "abc" }), /checkpoint_digest must be a sha256/);
});

test("SnapStart is refused", () => {
  assert.throws(() => synth({ ...S3, snapstart: true }), /snapstart=true is not supported/);
  synth({ ...S3, snapstart: false });
});

test("an s3:// checkpoint needs a real bucket and prefix", () => {
  assert.deepEqual(parseS3Checkpoint("s3://my-bucket/a/b"), { bucket: "my-bucket", prefix: "a/b/" });
  assert.equal(parseS3Checkpoint("StrandsAgents/x"), undefined);
  for (const bad of ["s3://my-bucket", "s3://my-bucket/", "s3://my-bucket//", "s3://my-bucket/a*", "s3://my-bucket/a?b", "s3://My_Bucket/a", "s3://b/a"]) {
    assert.throws(() => parseS3Checkpoint(bad), Error, bad);
  }
  assert.throws(() => synth({}), /set the decision checkpoint/);
});

test("checkpoints of one name in different places get different stacks", () => {
  const a = decisionStackName("s3://b1/teamA/e2b/");
  const b = decisionStackName("s3://b1/teamB/e2b/");
  assert.notEqual(a, b);
  assert.match(a, /^e2b-[0-9a-f]{8}-lambda$/);
  assert.equal(decisionStackName("s3://b1/teamA/e2b"), a);
  assert.equal(decisionStackName("StrandsAgents/strands-decider-E2B-gemma4-v2"), "strands-decider-e2b-gemma4-v2-lambda");
});

test("v1's stack keeps its defaults", () => {
  const app = new App({ context: { model: "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610", model_revision: "main", snapstart: false } });
  const t = Template.fromStack(new DeciderFunctionStack(app, "v1-test", { env: ENV }));
  t.hasResourceProperties("AWS::Lambda::Function", { Environment: { Variables: { DECIDER_MAX_REQUEST_TOKENS: "24576" } } });
  t.hasResourceProperties("AWS::Lambda::Url", { AuthType: "AWS_IAM" });
});
