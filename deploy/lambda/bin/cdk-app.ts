#!/usr/bin/env node
import { App } from "aws-cdk-lib";
import { DeciderFunctionStack, stackNameFor } from "../lib/decider-function-stack";

const app = new App();

// One stack per model, named after it, so several models can be deployed side by side in one
// account and region, beside a SageMaker endpoint for the same model (deploy/sagemaker).
new DeciderFunctionStack(app, stackNameFor(app.node.tryGetContext("model")), {
  // The account and region of the current credentials and CLI configuration.
  env: { account: process.env.CDK_DEFAULT_ACCOUNT, region: process.env.CDK_DEFAULT_REGION },
});
