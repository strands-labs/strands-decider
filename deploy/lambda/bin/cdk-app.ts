#!/usr/bin/env node
import { App } from "aws-cdk-lib";
import { stackNameFor } from "../lib/decider-function-stack";
import { DecisionFunctionStack, decisionStackName } from "../lib/decision-function-stack";
import { stackFor } from "../lib/stack-for";

async function main(): Promise<void> {
  const app = new App();
  const ctx = (key: string) => app.node.tryGetContext(key);
  // The account and region of the current credentials and CLI configuration.
  const env = { account: process.env.CDK_DEFAULT_ACCOUNT, region: process.env.CDK_DEFAULT_REGION };
  const model = String(ctx("model") ?? "");
  // The checkpoint says which stack deploys it (README.md, Which stack a model gets).
  const FunctionStack = await stackFor(model, String(ctx("model_revision") ?? "main"));
  // One stack per model, named after it, so several models can be deployed side by side in one
  // account and region, beside a SageMaker endpoint for the same model (deploy/sagemaker).
  // `stack_name` names it explicitly.
  const name = FunctionStack === DecisionFunctionStack ? decisionStackName(model) : stackNameFor(model);
  new FunctionStack(app, ctx("stack_name") ?? name, { env });
}

main().catch((e: Error) => {
  console.error(`error: ${e.message}`);
  process.exit(1);
});
