// DecisionFunctionStack, synthesized: its Function URL's auth, the build role's S3 grant,
// the request limit, and the refusals (README.md, Model checkpoints with a decision_config.json).
// Run: npm test.
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

const S3 = { model: "s3://my-bucket/models/e2b/" };

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
  const t = synth({ model: "StrandsAgents/example-e2b" });
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
  synth({ ...S3, model_digest: digest }).hasResourceProperties("Custom::CDKContainerImageBuild", {
    buildCommand: Match.stringLikeRegexp(`MODEL_DIGEST=${digest}`),
  });
  assert.throws(() => synth({ ...S3, model_digest: "abc" }), /model_digest must be a sha256/);
});

test("the CPU build is Graviton2's by default, every Arm tier with cpu=variants", () => {
  const command = (t: Template) => (Object.values(t.findResources("Custom::CDKContainerImageBuild"))[0] as any).Properties.buildCommand;
  assert.match(command(synth(S3)), /NATIVE=--lambda\b(?!-)/);
  assert.match(command(synth(S3)), /KLEIDIAI=0/);
  assert.match(command(synth({ ...S3, cpu: "variants", kleidiai: "true" })), /NATIVE=--lambda-variants/);
  assert.match(command(synth({ ...S3, cpu: "variants", kleidiai: "true" })), /KLEIDIAI=1/);
  assert.throws(() => synth({ ...S3, cpu: "native" }), /cpu must be baseline or variants/);
  assert.throws(() => synth({ ...S3, kleidiai: "yes" }), /kleidiai must be true or false/);
  assert.match(command(synth(S3)), /GRAVITON_SVE=0/);
  assert.match(command(synth({ ...S3, cpu: "variants", graviton_sve: "true" })), /GRAVITON_SVE=1/);
  assert.throws(() => synth({ ...S3, graviton_sve: "on" }), /graviton_sve must be true or false/);
  assert.match(command(synth(S3)), /DECODER_QUANT=Q8_0/);
  assert.match(command(synth({ ...S3, decoder_quant: "Q4_0" })), /DECODER_QUANT=Q4_0/);
  assert.throws(() => synth({ ...S3, decoder_quant: "Q2_K" }), /decoder_quant must be/);
});

test("v1's settings are refused, whatever their value", () => {
  for (const [key, value] of Object.entries({
    snapstart: false,
    vision: false,
    image_long_side: 448,
    decoder_precision: "q8_0",
    vision_precision: "f16",
  })) {
    assert.throws(() => synth({ ...S3, [key]: value }), new RegExp(`${key} is only supported for v1 model checkpoints`));
  }
});

test("an s3:// checkpoint needs a real bucket and prefix", () => {
  assert.deepEqual(parseS3Checkpoint("s3://my-bucket/a/b"), { bucket: "my-bucket", prefix: "a/b/" });
  assert.equal(parseS3Checkpoint("StrandsAgents/x"), undefined);
  for (const bad of ["s3://my-bucket", "s3://my-bucket/", "s3://my-bucket//", "s3://my-bucket/a*", "s3://my-bucket/a?b", "s3://My_Bucket/a", "s3://b/a"]) {
    assert.throws(() => parseS3Checkpoint(bad), Error, bad);
  }
  assert.throws(() => synth({}), /set the model/);
});

test("checkpoints of one name in different places get different stacks", () => {
  const a = decisionStackName("s3://b1/teamA/e2b/");
  const b = decisionStackName("s3://b1/teamB/e2b/");
  assert.notEqual(a, b);
  assert.match(a, /^e2b-[0-9a-f]{8}-lambda$/);
  assert.equal(decisionStackName("s3://b1/teamA/e2b"), a);
  assert.equal(decisionStackName("StrandsAgents/strands-decider-E2B-gemma4-v2"), "strands-decider-e2b-gemma4-v2-lambda");
});

test("v1's stack keeps its defaults, and refuses model_digest", () => {
  const v1 = (context: Record<string, unknown>) =>
    Template.fromStack(new DeciderFunctionStack(new App({ context }), "v1-test", { env: ENV }));
  const model = "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610";
  assert.throws(() => v1({ model, model_digest: "a".repeat(64) }), /model_digest is not supported for v1 model checkpoints/);
  const t = v1({ model });
  t.hasResourceProperties("Custom::CDKContainerImageBuild", {
    buildCommand: Match.stringLikeRegexp("VISION=false.*DECODER_PRECISION=q8_0.*VISION_PRECISION=f16"),
  });
  t.hasResourceProperties("AWS::Lambda::Function", { Environment: { Variables: { DECIDER_MAX_REQUEST_TOKENS: "24576" } } });
  t.hasResourceProperties("AWS::Lambda::Url", { AuthType: "AWS_IAM" });
});
