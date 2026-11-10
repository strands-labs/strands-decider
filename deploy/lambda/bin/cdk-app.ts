#!/usr/bin/env node
import { App } from "aws-cdk-lib";
import { DeciderFunctionStack, stackNameFor } from "../lib/decider-function-stack";
import { CHECKPOINT_NEEDED, DecisionFunctionStack, decisionStackName } from "../lib/decision-function-stack";

const app = new App();
const ctx = (key: string) => app.node.tryGetContext(key);

// `decision=true` deploys a decision checkpoint (`checkpoint`; README.md, Decision checkpoints);
// otherwise a v1 model with a pointer head (`model`).
const decision = String(ctx("decision") ?? "false");
if (decision !== "true" && decision !== "false") {
  throw new Error(`decision must be true or false, not ${JSON.stringify(decision)}`);
}
// The account and region of the current credentials and CLI configuration.
const env = { account: process.env.CDK_DEFAULT_ACCOUNT, region: process.env.CDK_DEFAULT_REGION };

// One stack per model, named after it, so several models can be deployed side by side in one
// account and region, beside a SageMaker endpoint for the same model (deploy/sagemaker).
// `stack_name` names it explicitly.
if (decision === "true") {
  const checkpoint = String(ctx("checkpoint") ?? "");
  if (!checkpoint) throw new Error(CHECKPOINT_NEEDED);
  new DecisionFunctionStack(app, ctx("stack_name") ?? decisionStackName(checkpoint), { env });
} else {
  new DeciderFunctionStack(app, ctx("stack_name") ?? stackNameFor(ctx("model")), { env });
}
